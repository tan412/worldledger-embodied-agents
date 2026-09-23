"""URDF → MuJoCo 世界(自研编译,不依赖上游 organoid):

  * fixed 关节子连杆并入父刚体(几何与惯量随之复合)—— 脚部角点球因此归到脚刚体;
  * 自由根策略加 freejoint,固定基策略根刚体直接挂 worldbody;
  * 地面平面按标定高度放置;脚部球 geom 设 margin/gap(接触判定的余量);
  * 邻接过滤:运动树距离 ≤ depth 的刚体对 exclude,消资产建模伪影;
  * 全部 collision geom 命名 collision::<body>::<i>,支撑与自碰撞按名字归属。
"""
from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .fk import URDFModel, Geom, _rpy_to_mat

GROUND_PENETRATION_LIMIT_M = 0.010
SELF_COLLISION_MIN_DEPTH_M = 0.001


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 → wxyz 四元数(MJCF 约定)。"""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


@dataclass
class WorldInfo:
    xml: str
    joint_order: list                    # 可动关节名(qpos 段顺序,自由根后)
    foot_geoms: dict = field(default_factory=dict)    # geom 名 -> (side, radius)
    robot_geom_prefix: str = "collision::"
    free_root: bool = True
    counts: dict = field(default_factory=dict)


def _merge_fixed(model: URDFModel):
    """fixed 子连杆归并:返回 body 树(仅可动关节分界)与每个 body 的 (geom, T复合, 源link)。"""
    # 每个 link 归属的 body(沿 fixed 链回溯到最近的可动关节子链接或根)
    parent_joint = {j.child: j for j in model.joints}

    def owner_and_T(link: str):
        T = np.eye(4)
        node = link
        while node in parent_joint and parent_joint[node].type == "fixed":
            j = parent_joint[node]
            T = j.T_origin @ T
            node = j.parent
        return node, T

    body_geoms: dict = {}
    body_inertia: dict = {}
    for name, link in model.links.items():
        owner, T = owner_and_T(name)
        for g in link.geoms:
            body_geoms.setdefault(owner, []).append((g, T @ g.T, name))
        if link.mass > 0:
            com_local = T @ np.array([*link.com, 1.0])
            body_inertia.setdefault(owner, []).append(
                (link.mass, com_local[:3], link.inertia,
                 T[:3, :3] @ _rpy_to_mat(link.inertia_rpy)))
    return body_geoms, body_inertia


def _geom_xml(parent_el, g: Geom, T: np.ndarray, name: str, assets: dict,
              foot_margin: float = None, group: str = None,
              friction: str = None) -> None:
    pos = T[:3, 3]
    quat = _mat_to_quat(T[:3, :3])
    attrs = {"name": name, "pos": " ".join(f"{v:.8g}" for v in pos),
             "quat": " ".join(f"{v:.8g}" for v in quat),
             "friction": friction or "0.8 0.02 0.002", "condim": "3"}
    if g.kind == "box":
        attrs.update(type="box", size=" ".join(f"{v/2:.8g}" for v in g.size))
    elif g.kind == "sphere":
        attrs.update(type="sphere", size=f"{g.size[0]:.8g}")
        if foot_margin is not None:
            attrs.update(margin=repr(float(foot_margin)), gap="0.001")
    elif g.kind == "cylinder":
        attrs.update(type="cylinder", size=f"{g.size[0]:.8g} {g.size[1]/2:.8g}")
    elif g.kind == "mesh":
        mesh_file = Path(g.mesh)
        if not mesh_file.exists():
            return                      # 缺失 mesh 由视觉验证记账,物理侧跳过
        key = mesh_file.stem
        assets[key] = str(mesh_file)
        attrs.update(type="mesh", mesh=key)
    if group is not None:
        attrs["group"] = group
    ET.SubElement(parent_el, "geom", attrs)


# 夹爪连杆的碰撞胶囊(URDF 只有视觉 mesh;fromto/半径按 STL 包络实测)。
# bar-1/bar-3 是四连杆的两根平行曲柄,bar-2 才是指板(在 build_world 里从
# 厂商焊死的 fixed 恢复为铰链,碰撞用 STL 凸包)
CLAW_CAPSULES = {name: ((0, 0, 0.005), (0, 0, -0.052), 0.010)
                 for name in ("l_f_bar-1", "l_f_bar-3", "l_b_bar-1", "l_b_bar-3",
                              "r_f_bar-1", "r_f_bar-3", "r_b_bar-1", "r_b_bar-3",
                              "l_f_bar-2", "l_b_bar-2", "r_f_bar-2", "r_b_bar-2")}
CLAW_JOINTS = ("l_f_bar-1_joint", "l_f_bar-3", "l_b_bar-1", "l_b_bar-3",
               "r_f_bar-1_joint", "r_f_bar-3", "r_b_bar-1", "r_b_bar-3",
               "l_f_bar-2", "l_b_bar-2", "r_f_bar-2", "r_b_bar-2")


def _stl_vertices(path) -> np.ndarray:
    """读 STL 顶点(二进制/ASCII 双支持),供指板碰撞盒按视觉网格贴合。"""
    import re
    import struct
    b = Path(path).read_bytes()
    if b[:5] == b"solid" and b"facet" in b[:2000]:
        vs = re.findall(rb"vertex\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)", b)
        return np.array(vs, dtype=float)
    n = struct.unpack_from("<I", b, 80)[0]
    dt = np.dtype([("n", "<3f4"), ("v", "<9f4"), ("attr", "<u2")])
    rec = np.frombuffer(b, dtype=dt, count=n, offset=84)
    return rec["v"].reshape(-1, 3).astype(float)


def build_world(model: URDFModel, profile, ground_height: float = 0.0,
                free_root: bool = True, with_floor: bool = True,
                objects: list = None, claw_collision: bool = False,
                with_visual_meshes: bool = False, cameras: list = None,
                hide_visual_of: tuple = (), full_body_actuators: bool = False,
                claw_closed_chain: bool = False, weld_body: str = None,
                root_pos: tuple = None, root_quat_wxyz: tuple = None,
                smooth_friction_links: tuple = (),
                omit_wrist_marker_ball: bool = False) -> WorldInfo:
    """with_visual_meshes: 视觉 mesh 以非碰撞 geom(group 1)渲染,碰撞体挪 group 3
    (仅显示层,物理不变);cameras: [{name,pos,xyaxes,fovy}] 复现数据自带相机视角。
    claw_closed_chain: URDF 表达不了的平行四连杆闭链,以 equality 关节耦合补上
    (f_bar-1 为主动,同侧其余三杆 1:±1 从动 —— 平行指板假设,入收据);此时爪执行器
    只挂主动关节。weld_body: 该连杆经 mocap 体 weld 约束走运动学轨迹(混合驱动:
    腕部轨迹精确、约束力有限,机械臂其余部分由伺服跟随)。root_pos/root_quat_wxyz:
    固定根时根刚体在世界系的安放位姿。smooth_friction_links: 这些连杆的碰撞几何
    按光滑塑料取 μ=0.3(指板橡胶垫仍为 1.5)—— 材质假设,入收据;腕端中央标记柱/
    前臂筒若沿用默认 μ=0.8,松爪后轻物会被摩擦楔在口袋里带走(实测)。
    omit_wrist_marker_ball: 不给腕端 5mm 标记球生成碰撞体。证据:球位于口袋正中
    (局部 (0,0,-0.17)),任何居中深握物体在几何上必与其穿插;而真实 episode 反复
    完成此类抓取 —— 数据证伪了"刚性球形障碍"的碰撞近似(实物应为柔性标定触点),
    按证据禁用并计入 counts.omitted_marker_balls。"""
    # Representation fixes must not leak into the caller's parsed model.
    # Otherwise compiling a claw-enabled variant changes later variants.
    model = copy.deepcopy(model)
    if claw_collision:
        # 机构修正:厂商 URDF 把四连杆的指板(bar-2)焊死在曲柄 bar-1 上
        # (revolute 被导出成 fixed,axis 标签还留着)。焊死的指板随曲柄同转,
        # 合爪呈剪刀 X 型 —— 与实物平动指板不符。恢复为铰链,闭链 equality
        # 反向耦合(bar-2 = -bar-1)让指板保持与掌面平行
        for _js in model.children.values():
            for _j in _js:
                if _j.type == "fixed" and _j.child.endswith("_bar-2"):
                    _j.type = "revolute"
                    _j.axis = (0.0, 1.0, 0.0)
                    _j.lower, _j.upper = -0.698, 0.698
    body_geoms, body_inertia = _merge_fixed(model)
    foot_bodies = dict(profile.foot_bodies or {})
    margin = float(profile.foot_contact_margin or 0.003)
    robot_friction = float(getattr(profile, "robot_friction", 0.8))
    floor_friction = float(getattr(profile, "floor_friction", 0.8))
    claw_friction = float(getattr(profile, "claw_friction", 1.5))
    claw_joint_damping = float(getattr(profile, "claw_joint_damping", 1.5))
    claw_joint_armature = float(getattr(profile, "claw_joint_armature", 0.005))
    claw_actuator_kp = float(getattr(profile, "claw_actuator_kp", 60.0))
    claw_actuator_force_limit = float(
        getattr(profile, "claw_actuator_force_limit", 2.0))
    collision_mesh_prefixes = tuple(
        getattr(profile, "collision_mesh_body_prefixes", ()) or ())

    root = ET.Element("mujoco", {"model": profile.name})
    ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true",
                                     "balanceinertia": "true"})
    opt_attrs = {"gravity": "0 0 -9.81"}
    if full_body_actuators:
        opt_attrs["integrator"] = "implicitfast"   # 隐式阻尼积分,稳住刚性 PD
    ET.SubElement(root, "option", opt_attrs)
    asset_el = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    if with_floor:
        ET.SubElement(world, "geom", {"name": "floor", "type": "plane",
                                      "pos": f"0 0 {ground_height:.8g}",
                                      "size": "20 20 0.1", "condim": "3",
                                      "friction": f"{floor_friction} 0.02 0.002"})

    assets: dict = {}
    foot_geoms: dict = {}
    counts = {"collision": 0, "visual_skipped": 0, "foot_contact": 0}
    joint_order = []
    joint_by_name = {}
    body_parent: dict = {}               # 邻接过滤用

    def emit_body(link_name: str, parent_el, joint=None):
        body = ET.SubElement(parent_el, "body", {"name": link_name})
        if joint is None and free_root:
            ET.SubElement(body, "freejoint", {"name": "root"})
        if joint is None and not free_root:
            if root_pos is not None:
                body.set("pos", " ".join(f"{v:.8g}" for v in root_pos))
            if root_quat_wxyz is not None:
                body.set("quat", " ".join(f"{v:.8g}" for v in root_quat_wxyz))
        if joint is not None:
            joint_by_name[joint.name] = joint
            body.set("pos", " ".join(f"{v:.8g}" for v in joint.T_origin[:3, 3]))
            body.set("quat", " ".join(f"{v:.8g}" for v in _mat_to_quat(joint.T_origin[:3, :3])))
            jattrs = {"name": joint.name, "axis": " ".join(f"{v:.8g}" for v in joint.axis)}
            if joint.type == "prismatic":
                jattrs["type"] = "slide"
            else:
                jattrs["type"] = "hinge"
            if joint.type != "continuous" and (joint.lower or joint.upper):
                jattrs["range"] = f"{joint.lower:.8g} {joint.upper:.8g}"
                jattrs["limited"] = "true"
            if joint.damping:
                jattrs["damping"] = f"{joint.damping:.8g}"
            if joint.friction:
                jattrs["frictionloss"] = f"{joint.friction:.8g}"
            if claw_collision and joint.name in CLAW_JOINTS:
                # The restored claw hinges have no vendor dynamics entry.
                # Keep that explicit approximation, but never overwrite a
                # source damping value if one exists.
                jattrs.setdefault("damping", str(claw_joint_damping))
                jattrs["armature"] = str(claw_joint_armature)
            ET.SubElement(body, "joint", jattrs)
            joint_order.append(joint.name)
        # 惯量:合并后的分量逐个交给 MJCF(inertial 允许多源时须复合;这里以主 link 为准,
        # 零质量 body 由 compiler balanceinertia 兜底)
        parts = body_inertia.get(link_name, [])
        if parts:
            # 正经的惯量合成:各部件张量旋到刚体系,平行轴定理搬到合成质心,求和后
            # 以 fullinertia 原样交给 MJCF(此前按大小排序 diaginertia 会把轴张冠李戴,
            # 运动学重放无感,全动力学直接数值爆炸)
            mass = sum(p[0] for p in parts)
            com = sum(p[0] * np.asarray(p[1]) for p in parts) / mass
            I_total = np.zeros((3, 3))
            for pm, pcom, in6, R in parts:
                ixx, iyy, izz, ixy, ixz, iyz = in6
                I_local = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
                I_rot = R @ I_local @ R.T
                r = np.asarray(pcom) - com
                I_total += I_rot + pm * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
            # 防退化:特征值下限
            w, V = np.linalg.eigh(I_total)
            w = np.maximum(w, 1e-7)
            I_total = V @ np.diag(w) @ V.T
            ET.SubElement(body, "inertial", {
                "mass": f"{mass:.8g}", "pos": " ".join(f"{v:.8g}" for v in com),
                "fullinertia": " ".join(f"{v:.8g}" for v in
                                        (I_total[0, 0], I_total[1, 1], I_total[2, 2],
                                         I_total[0, 1], I_total[0, 2], I_total[1, 2]))})
        idx = 0
        if claw_collision and link_name in CLAW_CAPSULES:
            emitted = False
            if link_name.endswith("bar-2"):
                # 指板(bar-2)与其焊接的指尖(fingers)碰撞 = 各自 STL 板体贴合盒,
                # 按源 link 命名(claw::*_bar-2 / claw::*_fingers)。教训:指尖才是
                # 真机的夹取面,漏装碰撞时指尖在视频里交叉互穿而判据全瞎;凸包会
                # 填平 L 型凹角虚胖假性互撞;手调薄盒面差 ~4mm 造成"沉进指板"假象
                for g, T, src in body_geoms.get(link_name, []):
                    if (g.visual and g.kind == "mesh" and Path(g.mesh).exists()
                            and tuple(g.scale) == (1.0, 1.0, 1.0)):
                        v = _stl_vertices(g.mesh) @ T[:3, :3].T + T[:3, 3]
                        xs = v[:, 0]
                        x_in = xs.min() if abs(xs.min()) > abs(xs.max()) else xs.max()
                        in_slab = np.abs(xs - x_in) <= 0.013
                        # 内侧板体 + 其余臂段各一盒:碰撞覆盖整个视觉网格,
                        # 物体才不会在"无碰撞区"里视觉虚穿
                        parts = [(f"claw::{src}", v[in_slab])]
                        if (~in_slab).sum() > 50:
                            parts.append((f"claw::{src}::arm", v[~in_slab]))
                        for nm_, pts_ in parts:
                            lo, hi = pts_.min(0), pts_.max(0)
                            center = (lo + hi) / 2
                            half = np.maximum((hi - lo) / 2, 0.002)
                            cattr = {"name": nm_, "type": "box",
                                     "pos": " ".join(f"{c:.6g}" for c in center),
                                     "size": " ".join(f"{c:.6g}" for c in half),
                                     "friction": f"{claw_friction} 0.1 0.002", "condim": "6",
                                     # 负 solref = 绝对刚度形式(30kN/m),与物体质量
                                     # 无关 —— 时间常数形式下 10g 轻物接触刚度随
                                     # 质量缩水,力钳位处压出 4mm 假形变(用户抓出)
                                     "solref": "-30000 -300", "mass": "0.01"}
                            if with_visual_meshes:
                                cattr["group"] = "3"
                            ET.SubElement(body, "geom", cattr)
                            counts["collision"] += 1
                        emitted = True
            if not emitted:
                cattr = {"name": f"claw::{link_name}", "type": "box",
                         "size": "0.004 0.010 0.028", "pos": "0 0 -0.0235",
                         "friction": f"{claw_friction} 0.1 0.002", "condim": "6",
                         "solref": "0.01 1", "mass": "0.02"}
                if with_visual_meshes:
                    cattr["group"] = "3"
                ET.SubElement(body, "geom", cattr)
                counts["collision"] += 1
        for g, T, src in body_geoms.get(link_name, []):
            if g.visual:
                if (g.kind == "mesh" and Path(g.mesh).exists()
                        and any(link_name.startswith(p)
                                for p in collision_mesh_prefixes)):
                    _geom_xml(
                        body, g, T,
                        f"collision::{link_name}::visual::{counts['visual_skipped']}",
                        assets,
                        group=("3" if with_visual_meshes else None),
                        friction=(f"0.3 0.02 0.002"
                                  if link_name in smooth_friction_links
                                  else f"{robot_friction} 0.02 0.002"))
                    counts["collision"] += 1
                if (with_visual_meshes and g.kind == "mesh" and Path(g.mesh).exists()
                        and src not in hide_visual_of):
                    key = Path(g.mesh).stem
                    assets[key] = str(g.mesh)
                    ET.SubElement(body, "geom", {
                        "name": f"visual::{link_name}::{counts['visual_skipped']}",
                        "type": "mesh", "mesh": key,
                        "pos": " ".join(f"{v:.8g}" for v in T[:3, 3]),
                        "quat": " ".join(f"{v:.8g}" for v in _mat_to_quat(T[:3, :3])),
                        "contype": "0", "conaffinity": "0", "group": "1",
                        "rgba": "0.82 0.83 0.85 1"})
                counts["visual_skipped"] += 1
                continue
            if (omit_wrist_marker_ball and g.kind == "sphere"
                    and g.size[0] <= 0.006
                    and link_name in ("zarm_l7_link", "zarm_r7_link")):
                counts["omitted_marker_balls"] = counts.get("omitted_marker_balls", 0) + 1
                idx += 1
                continue
            name = f"collision::{link_name}::{idx}"
            is_foot = link_name in foot_bodies and g.kind == "sphere"
            eff_margin = None if full_body_actuators else (margin if is_foot else None)
            _geom_xml(body, g, T, name, assets, eff_margin,
                      group=("3" if with_visual_meshes else None),
                      friction=(f"0.3 0.02 0.002" if link_name in smooth_friction_links
                                else f"{robot_friction} 0.02 0.002"))
            if is_foot:
                foot_geoms[name] = (foot_bodies[link_name], g.size[0])
                counts["foot_contact"] += 1
            counts["collision"] += 1
            idx += 1
        for j in model.children.get(link_name, []):
            if j.type == "fixed":
                # 已归并;但其子树上可能还有可动关节(fixed 链中转)
                for jj in _movable_descendants(model, j.child):
                    body_parent[jj.child] = link_name
                    emit_body_chain(jj, body, link_name)
                continue
            body_parent[j.child] = link_name
            emit_body(j.child, body, j)
        return body

    def _movable_descendants(model, link):
        """穿过 fixed 链找直接可动子关节(fixed 的复合位姿并进 joint T_origin)。"""
        out = []
        for j in model.children.get(link, []):
            if j.type == "fixed":
                for jj in _movable_descendants(model, j.child):
                    merged = type(jj)(name=jj.name, type=jj.type, parent=link, child=jj.child,
                                      T_origin=j.T_origin @ jj.T_origin, axis=jj.axis,
                                      lower=jj.lower, upper=jj.upper,
                                      velocity=jj.velocity, effort=jj.effort,
                                      damping=jj.damping, friction=jj.friction)
                    out.append(merged)
            else:
                out.append(j)
        return out

    def emit_body_chain(joint, parent_el, parent_name):
        emit_body(joint.child, parent_el, joint)

    emit_body(model.root_link, world)

    # 邻接过滤:运动树距离 ≤ depth 的 body 对不做自碰撞判定
    depth = int(profile.adjacent_filter_depth or 3)
    contact = ET.SubElement(root, "contact")
    names = list(body_parent.keys()) + [model.root_link]
    def dist(a, b):
        # 沿 parent 链找最近公共祖先距离
        pa, chain = a, {a: 0}
        d = 0
        while pa in body_parent:
            pa = body_parent[pa]
            d += 1
            chain[pa] = d
        pb, d2 = b, 0
        while True:
            if pb in chain:
                return chain[pb] + d2
            if pb not in body_parent:
                return None
            pb = body_parent[pb]
            d2 += 1
    seen = set()
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if (a, b) in seen:
                continue
            seen.add((a, b))
            d = dist(a, b)
            if d is not None and d <= depth:
                ET.SubElement(contact, "exclude", {"body1": a, "body2": b})

    # 自由物体与静态台面(世界修订的 patch 产物)
    for spec in (objects or []):
        kind = spec.get("kind", "box")
        name = spec.get("name", "object")
        pos = " ".join(f"{v:.5g}" for v in spec.get("pos", (0, 0, 0)))
        rgba = " ".join(str(v) for v in spec.get("rgba", (0.85, 0.55, 0.2, 1)))
        fric = spec.get("friction", 0.9)
        size = spec.get("size", (0.03,))
        gattr = {"friction": f"{fric} 0.02 0.002", "condim": "4", "rgba": rgba,
                 "name": f"object::{name}"}
        if spec.get("soft"):
            gattr.update(solref="0.02 0.8", solimp="0.8 0.92 0.01",
                         friction=f"{fric} 0.1 0.005", condim="6")
        if kind in ("box", "static_box"):
            gattr.update(type="box", size=" ".join(f"{v/2:.5g}" for v in size))
        elif kind == "cylinder":
            gattr.update(type="cylinder", size=f"{size[0]:.5g} {size[1]/2:.5g}")
        elif kind == "sphere":
            gattr.update(type="sphere", size=f"{size[0]:.5g}")
        if kind == "static_box":
            gattr["pos"] = pos
            ET.SubElement(world, "geom", gattr)
            continue
        battr = {"name": f"object::{name}", "pos": pos}
        if spec.get("quat_wxyz"):
            battr["quat"] = " ".join(f"{v:.6g}" for v in spec["quat_wxyz"])
        obody = ET.SubElement(world, "body", battr)
        ET.SubElement(obody, "freejoint", {"name": f"objfree::{name}"})
        subs = spec.get("geoms")
        if subs:
            # 复合刚体(如底重立式文件袋:宽重底座 + 薄可夹上板)——
            # 单几何体表达不了"底稳顶薄"的真实物体,逐子几何声明,质量分布入收据
            for k2, sg in enumerate(subs):
                sk = sg.get("kind", "box")
                ssz = sg["size"]
                sattr = {"name": f"object::{name}::{k2}",
                         "pos": " ".join(f"{v:.5g}" for v in sg.get("pos", (0, 0, 0))),
                         "friction": f"{sg.get('friction', fric)} 0.02 0.002",
                         "condim": "4", "rgba": rgba,
                         "mass": f"{sg.get('mass', 0.1):.4g}"}
                if sg.get("soft", spec.get("soft")):
                    sattr.update(solref="0.02 0.8", solimp="0.8 0.92 0.01",
                                 friction=f"{sg.get('friction', fric)} 0.1 0.005",
                                 condim="6")
                if sk == "box":
                    sattr.update(type="box", size=" ".join(f"{v/2:.5g}" for v in ssz))
                elif sk == "cylinder":
                    sattr.update(type="cylinder", size=f"{ssz[0]:.5g} {ssz[1]/2:.5g}")
                elif sk == "sphere":
                    sattr.update(type="sphere", size=f"{ssz[0]:.5g}")
                ET.SubElement(obody, "geom", sattr)
            continue
        gattr["mass"] = f"{spec.get('mass', 0.2):.4g}"
        ET.SubElement(obody, "geom", gattr)

    # 夹爪位置执行器(力受限的位置伺服 —— 夹持力有界,不会把物体挤爆)
    if full_body_actuators:
        act_fb = ET.SubElement(root, "actuator")
        servo = getattr(profile, "fullbody_servo", {}) or {}
        def gains_of(jn):
            # 近端硬、远端软:末端小惯量关节配大 kp 会数值爆炸(踝/腕实测)
            if jn.startswith("leg"):
                idx = int(jn.split("_")[1][1])
                # 这是离线位置重放的控制器近似,不是已标定的真机增益。
                # 额定 effort 由下方 forcerange 硬约束,不能被增益选择绕过。
                cfg = servo.get("leg", {})
                return (cfg.get("kp", 1500), cfg.get("kv", 150),
                        cfg.get("proximal_force_limit", 300)
                        if idx <= 4 else cfg.get("distal_force_limit", 250))
            if jn.startswith("zarm"):
                idx = int(jn.split("_")[1][1])
                cfg = servo.get("zarm", {})
                return ((cfg.get("proximal_kp", 100), cfg.get("proximal_kv", 10),
                         cfg.get("proximal_force_limit", 60))
                        if idx <= 4 else
                        (cfg.get("distal_kp", 30), cfg.get("distal_kv", 3),
                         cfg.get("distal_force_limit", 20)))
            if jn.startswith("zhead"):
                cfg = servo.get("zhead", {})
                return (cfg.get("kp", 10), cfg.get("kv", 1),
                        cfg.get("force_limit", 5))
            if jn.startswith("waist"):
                cfg = servo.get("waist", {})
                return (cfg.get("kp", 300), cfg.get("kv", 30),
                        cfg.get("force_limit", 150))
            return None
        for jn in joint_order:
            g3 = gains_of(jn)
            if g3 is None:
                continue
            kp, kv, fr = g3
            effort = joint_by_name[jn].effort
            if effort is not None and effort > 0:
                fr = min(fr, effort)
            ET.SubElement(act_fb, "position", {
                "name": f"fb::{jn}", "joint": jn, "kp": str(kp), "kv": str(kv),
                "forcerange": f"-{fr} {fr}"})

    if claw_collision:
        act = ET.SubElement(root, "actuator")
        # 闭链耦合时只驱动主动关节(f_bar-1),其余三杆由 equality 从动
        drive_joints = (("l_f_bar-1_joint", "r_f_bar-1_joint")
                        if claw_closed_chain else CLAW_JOINTS)
        for jn in drive_joints:
            # forcerange ±2N·m ≈ 指尖 40N 夹持力(小型夹爪物理量级;此前 ±20
            # 等效 ~380N,指令过驱压出 5mm 假穿透 —— 真机是电机限流+亚毫米形变)
            ET.SubElement(act, "position", {
                "name": f"act::{jn}", "joint": jn, "kp": str(claw_actuator_kp),
                "forcerange": f"-{claw_actuator_force_limit} {claw_actuator_force_limit}",
                "ctrlrange": "-0.698 0.698"})

    if claw_closed_chain or weld_body:
        eq = ET.SubElement(root, "equality")
    if claw_closed_chain:
        # 四连杆闭链:bar-1/bar-3 双平行曲柄同转(1:1),指板 bar-2 反向耦合
        # (bar-2 = -曲柄角)保持与掌面平行 —— 平动指板,与实物一致。
        # solref 0.002:刚性连杆语义 —— 0.004 时持球载荷实测把闭链撑开 ~0.11 rad
        for side in ("l", "r"):
            master = f"{side}_f_bar-1_joint"
            for slave, coef in ((f"{side}_f_bar-3", "1"),
                                (f"{side}_b_bar-1", "-1"), (f"{side}_b_bar-3", "-1"),
                                (f"{side}_f_bar-2", "-1"), (f"{side}_b_bar-2", "1")):
                ET.SubElement(eq, "joint", {
                    "joint1": slave, "joint2": master,
                    "polycoef": f"0 {coef} 0 0 0", "solref": "0.002 1"})
    if weld_body:
        ET.SubElement(world, "body", {"name": "mocap::wrist", "mocap": "true"})
        ET.SubElement(eq, "weld", {
            "body1": "mocap::wrist", "body2": weld_body,
            "relpose": "0 0 0 1 0 0 0", "solref": "0.004 1"})

    if claw_collision:
        pair_el = contact          # 已有 <contact> 元素(exclude 用的同一个)
        # 指板/指尖 f×b 全配对(尖端交叉必被接触挡住);曲柄(bar-1/bar-3)
        # 在实物中本就互相穿插布置,给曲柄加接触对会卡死机构
        for side in ("l", "r"):
            for fa in ("bar-2", "fingers"):
                for bb in ("bar-2", "fingers"):
                    ET.SubElement(pair_el, "pair", {
                        "geom1": f"claw::{side}_f_{fa}",
                        "geom2": f"claw::{side}_b_{bb}",
                        "condim": "1", "margin": "0.001"})

    for cam in (cameras or []):
        ET.SubElement(world, "camera", {
            "name": cam["name"],
            "pos": " ".join(f"{v:.6g}" for v in cam["pos"]),
            "xyaxes": " ".join(f"{v:.6g}" for v in cam["xyaxes"]),
            "fovy": f"{cam.get('fovy', 60):.4g}"})

    for key, path in sorted(assets.items()):
        ET.SubElement(asset_el, "mesh", {"name": key, "file": path})

    return WorldInfo(xml=ET.tostring(root, encoding="unicode"),
                     joint_order=joint_order, foot_geoms=foot_geoms,
                     free_root=free_root, counts=counts)


def compile_model(info: WorldInfo):
    import mujoco
    return mujoco.MjModel.from_xml_string(info.xml)
