"""URDF 解析与正向运动学:运动学预检、Blender 摆位、MuJoCo 世界共用的几何底座。"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MOVABLE = ("revolute", "continuous", "prismatic")


def _rpy_to_mat(rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]])


def _origin_T(el) -> np.ndarray:
    T = np.eye(4)
    if el is None:
        return T
    xyz = [float(v) for v in el.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in el.get("rpy", "0 0 0").split()]
    T[:3, :3] = _rpy_to_mat(rpy)
    T[:3, 3] = xyz
    return T


def _axis_rot(axis, angle) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / (np.linalg.norm(a) or 1.0)
    x, y, z = a
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    T_origin: np.ndarray
    axis: tuple = (1.0, 0.0, 0.0)
    lower: float = 0.0
    upper: float = 0.0
    velocity: float = 0.0
    effort: float | None = None
    damping: float = 0.0
    friction: float = 0.0


@dataclass
class Geom:
    kind: str            # box / sphere / cylinder / mesh
    T: np.ndarray        # 相对所属 link 原点
    size: tuple = ()     # box: (x,y,z) 全尺寸; sphere: (r,); cylinder: (r, len)
    mesh: str = ""       # mesh 文件名(已解析为绝对路径)
    scale: tuple = (1.0, 1.0, 1.0)
    visual: bool = False


@dataclass
class Link:
    name: str
    mass: float = 0.0
    com: tuple = (0.0, 0.0, 0.0)
    inertia: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)   # ixx iyy izz ixy ixz iyz
    inertia_rpy: tuple = (0.0, 0.0, 0.0)
    geoms: list = field(default_factory=list)


@dataclass
class URDFModel:
    path: Path
    root_link: str
    links: dict            # name -> Link
    joints: list           # 文档序全部 Joint
    movable: list          # 文档序可动 Joint
    children: dict         # parent link -> [Joint]

    @property
    def movable_names(self):
        return [j.name for j in self.movable]

    def limits(self) -> dict:
        return {j.name: (j.lower, j.upper, j.velocity) for j in self.movable}

    def fk(self, qpos: dict, base_T: np.ndarray = None) -> dict:
        """全部 link 的世界位姿(4x4)。qpos: joint name -> 角度/位移。"""
        T = {self.root_link: np.eye(4) if base_T is None else base_T}
        stack = [self.root_link]
        while stack:
            parent = stack.pop()
            for j in self.children.get(parent, []):
                Tj = T[parent] @ j.T_origin
                q = float(qpos.get(j.name, 0.0))
                if j.type in ("revolute", "continuous"):
                    R = np.eye(4)
                    R[:3, :3] = _axis_rot(j.axis, q)
                    Tj = Tj @ R
                elif j.type == "prismatic":
                    P = np.eye(4)
                    P[:3, 3] = np.asarray(j.axis, dtype=float) * q
                    Tj = Tj @ P
                T[j.child] = Tj
                stack.append(j.child)
        return T


def resolve_mesh(filename: str, mesh_dir: Path) -> str:
    """package:// 与相对路径都落到资产目录;找不到原样返回(缺失由消费方记录)。"""
    name = filename.split("/")[-1]
    cand = mesh_dir / name
    if cand.exists():
        return str(cand)
    if filename.startswith(("package://", "model://")):
        tail = filename.split("//", 1)[1].split("/", 1)[-1]
        cand = mesh_dir.parent / tail
        if cand.exists():
            return str(cand)
    return filename


def load_urdf(path: Path, mesh_dir: Path = None) -> URDFModel:
    path = Path(path)
    mesh_dir = Path(mesh_dir) if mesh_dir else path.parent / "meshes"
    root = ET.parse(path).getroot()

    links = {}
    for el in root.findall("link"):
        link = Link(name=el.get("name"))
        inertial = el.find("inertial")
        if inertial is not None:
            mass = inertial.find("mass")
            link.mass = float(mass.get("value", 0)) if mass is not None else 0.0
            io = inertial.find("origin")
            link.com = tuple(float(v) for v in (io.get("xyz", "0 0 0") if io is not None
                                                else "0 0 0").split())
            ine = inertial.find("inertia")
            if ine is not None:
                link.inertia = tuple(float(ine.get(k, 0)) for k in
                                     ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"))
            link.inertia_rpy = tuple(float(v) for v in
                                     (io.get("rpy", "0 0 0") if io is not None
                                      else "0 0 0").split())
        for tag, visual in (("collision", False), ("visual", True)):
            for c in el.findall(tag):
                g = c.find("geometry")
                if g is None:
                    continue
                T = _origin_T(c.find("origin"))
                if g.find("box") is not None:
                    size = tuple(float(v) for v in g.find("box").get("size").split())
                    link.geoms.append(Geom("box", T, size=size, visual=visual))
                elif g.find("sphere") is not None:
                    link.geoms.append(Geom("sphere", T,
                                           size=(float(g.find("sphere").get("radius")),),
                                           visual=visual))
                elif g.find("cylinder") is not None:
                    cy = g.find("cylinder")
                    link.geoms.append(Geom("cylinder", T,
                                           size=(float(cy.get("radius")),
                                                 float(cy.get("length"))), visual=visual))
                elif g.find("mesh") is not None:
                    m = g.find("mesh")
                    scale = tuple(float(v) for v in m.get("scale", "1 1 1").split())
                    link.geoms.append(Geom("mesh", T, visual=visual, scale=scale,
                                           mesh=resolve_mesh(m.get("filename", ""), mesh_dir)))
        links[link.name] = link

    joints, children, has_parent = [], {}, set()
    for el in root.findall("joint"):
        axis_el = el.find("axis")
        limit = el.find("limit")
        dynamics = el.find("dynamics")
        j = Joint(
            name=el.get("name"), type=el.get("type"),
            parent=el.find("parent").get("link"), child=el.find("child").get("link"),
            T_origin=_origin_T(el.find("origin")),
            axis=tuple(float(v) for v in (axis_el.get("xyz", "1 0 0") if axis_el is not None
                                          else "1 0 0").split()),
            lower=float(limit.get("lower", 0)) if limit is not None else 0.0,
            upper=float(limit.get("upper", 0)) if limit is not None else 0.0,
            velocity=float(limit.get("velocity", 0)) if limit is not None else 0.0,
            effort=(float(limit.get("effort")) if limit is not None
                    and limit.get("effort") is not None else None),
            damping=(float(dynamics.get("damping", 0))
                     if dynamics is not None else 0.0),
            friction=(float(dynamics.get("friction", 0))
                      if dynamics is not None else 0.0))
        joints.append(j)
        children.setdefault(j.parent, []).append(j)
        has_parent.add(j.child)

    roots = [n for n in links if n not in has_parent]
    return URDFModel(path=path, root_link=roots[0], links=links, joints=joints,
                     movable=[j for j in joints if j.type in MOVABLE],
                     children=children)
