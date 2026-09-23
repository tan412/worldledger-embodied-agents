"""世界修订机制(Rosetta canonical world + diff 链 + 世界 patch 的回归,薄实现)。

世界是纯函数:world = f(机器人档案, 地面高度, 物体清单, 附加碰撞体)。
修订链只记录"改了什么"(patch 文档),每一版的身份 = 链式哈希
sha256(父版哈希 + patch JSON + 编译出的 MJCF 哈希);patch 幂等
(同名物体的 add/swap 是覆盖语义,重复应用结果不变)。

patch 操作:
  add_object    {name, kind(box/cylinder/sphere/static_box), size, pos, quat?,
                 mass?, friction?, rgba?}
  swap_object   同 add_object(语义:替换同名物体,保持位姿可省略字段继承)
  move_object   {name, pos}
  set_physics   {name, mass?/friction?}
  remove_object {name}
  enable_claw_collision {}   给夹爪连杆装碰撞胶囊(URDF 原样只有视觉 mesh,
                             无碰撞体,物理上抓不住东西 —— 此 patch 显式声明补上)
"""
from __future__ import annotations

import json
import copy
from pathlib import Path

from .hashing import sha256_bytes, sha256_json
from . import mjworld


class WorldAdmissionError(ValueError):
    """世界准入判负:补丁描述的物体违反物理常识,拒绝进入世界(fail-closed)。"""


# 材料密度表 g/cm³(准入锚点;数据来源:常识物性,供 material 声明与密度带校验)
MATERIAL_DENSITY = {"泡沫": 0.05, "木": 0.60, "塑料": 1.20, "橡胶": 1.10,
                    "铝": 2.70, "玻璃": 2.50, "钢": 7.85, "铜": 8.96, "铅": 11.34}
DENSITY_BAND = (0.02, 11.5)          # g/cm³:比最轻泡沫更轻/比铅更重的都拒收
FRICTION_BAND = (0.02, 2.0)
MAX_DIM_M = 0.5            # 可抓取物体单边上限:更大的"物体"多半是台面/家具写错了
MAX_STATIC_DIM_M = 2.0     # 静态支撑面(台面)单边上限:真实桌面 0.6~1.2m,不受物体带约束


def _volume_m3(kind: str, size) -> float:
    s = list(size)
    if kind in ("box", "static_box"):
        return s[0] * s[1] * s[2]
    if kind == "cylinder":
        return 3.14159265 * s[0] ** 2 * s[1]
    return 4.18879 * s[0] ** 3                 # sphere


def admit_object(name: str, spec: dict) -> None:
    """物体准入(第一层根治):质量必须与几何、材质自洽,非物理的世界内容
    在提交补丁时就被拒绝,而不是三天后在成品视频里被人眼发现。
    教训案例:1.6cm 截面 300g 方柱 = 密度 18 g/cm³(钨级),曾进过提议清单。"""
    kind = spec.get("kind")
    size = spec.get("size")
    if not kind or not size:
        return                                 # 不完整补丁由编译层报错
    cap = MAX_STATIC_DIM_M if kind == "static_box" else MAX_DIM_M
    if any(not (0 < float(v) <= cap) for v in size):
        raise WorldAdmissionError(f"{name}: 尺寸 {size} 超出 (0, {cap}m]")
    fr = spec.get("friction")
    if fr is not None and not (FRICTION_BAND[0] <= float(fr) <= FRICTION_BAND[1]):
        raise WorldAdmissionError(f"{name}: 摩擦系数 {fr} 超出 {FRICTION_BAND}")
    if kind == "static_box" or spec.get("mass") is None:
        return                                 # 静态台面/无质量声明不做密度校验
    vol = _volume_m3(kind, size)
    rho = float(spec["mass"]) / vol / 1000.0   # g/cm³
    mat = spec.get("material")
    if mat is not None:
        if mat not in MATERIAL_DENSITY:
            raise WorldAdmissionError(f"{name}: 未知材质 {mat!r}(可用: "
                                      f"{sorted(MATERIAL_DENSITY)})")
        ref = MATERIAL_DENSITY[mat]
        if not (0.7 * ref <= rho <= 1.3 * ref):
            raise WorldAdmissionError(
                f"{name}: 质量 {spec['mass']}kg 隐含密度 {rho:.2f} g/cm³,"
                f"与声明材质 {mat}(ρ={ref})偏差超 30%")
    elif not (DENSITY_BAND[0] <= rho <= DENSITY_BAND[1]):
        raise WorldAdmissionError(
            f"{name}: 质量 {spec['mass']}kg / 体积 {vol*1e6:.1f}cm³ 隐含密度 "
            f"{rho:.2f} g/cm³,超出物理带 {DENSITY_BAND}(泡沫~铅);"
            f"请声明 material 或给出自洽质量")


def apply_patches(base_state: dict, patches: list) -> dict:
    """把 patch 序列应用到世界状态(纯数据,幂等)。"""
    state = {"objects": copy.deepcopy(base_state.get("objects", {})),
             "claw_collision": bool(base_state.get("claw_collision", False))}
    for p in patches:
        op = p["op"]
        if op in ("add_object", "swap_object"):
            prev = state["objects"].get(p["name"], {})
            spec = {**prev, **{k: v for k, v in p.items() if k not in ("op",)}}
            admit_object(p["name"], spec)          # 准入:非物理内容拒收
            state["objects"][p["name"]] = spec
        elif op == "move_object":
            state["objects"][p["name"]]["pos"] = p["pos"]
        elif op == "set_physics":
            state["objects"][p["name"]].update(
                {k: v for k, v in p.items() if k in ("mass", "friction")})
            admit_object(p["name"], state["objects"][p["name"]])
        elif op == "remove_object":
            state["objects"].pop(p["name"], None)
        elif op == "enable_claw_collision":
            state["claw_collision"] = True
        else:
            raise ValueError(f"未知 patch 操作: {op}")
    return state


class WorldRevisionStore:
    """world@000 起步的修订链;每版落一份收据,可从收据完整复现。"""

    def __init__(self, out_dir: Path, profile, model_u, ground_height: float):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.profile = profile
        self.model_u = model_u
        self.ground = ground_height
        self.revisions = []          # [{rev, parent_hash, patches, state, xml_hash, chain_hash}]
        base_info = self._compile({"objects": {}, "claw_collision": False})
        h = sha256_bytes(base_info.xml.encode())
        self.revisions.append({"rev": "world@000", "parent_hash": None, "patches": [],
                               "state": {"objects": {}, "claw_collision": False},
                               "xml_hash": h, "chain_hash": h})
        self._write_receipt()

    def _compile(self, state: dict):
        return mjworld.build_world(
            self.model_u, self.profile, self.ground, free_root=True,
            objects=list(state["objects"].values()) and
            [{"name": n, **s} for n, s in state["objects"].items()],
            claw_collision=state["claw_collision"])

    def derive(self, patches: list, label: str = None):
        """从当前最新版打 patch 出新版;返回 (rev_doc, WorldInfo)。"""
        parent = self.revisions[-1]
        state = apply_patches(parent["state"], patches)
        info = self._compile(state)
        xml_hash = sha256_bytes(info.xml.encode())
        chain = sha256_json({"parent": parent["chain_hash"],
                             "patches": patches, "xml": xml_hash})
        rev = {"rev": label or f"world@{len(self.revisions):03d}",
               "parent_hash": parent["chain_hash"], "patches": patches,
               "state": state, "xml_hash": xml_hash, "chain_hash": chain}
        self.revisions.append(rev)
        self._write_receipt()
        return rev, info

    def _write_receipt(self):
        (self.dir / "world-revisions.json").write_text(json.dumps({
            "schema": "organoid-kernel.world-revisions.v1",
            "base": {"profile": self.profile.name, "ground_height_m": self.ground},
            "note": "世界=纯函数(档案+地面+物体清单);修订身份=链式哈希;patch 幂等",
            "revisions": self.revisions,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
