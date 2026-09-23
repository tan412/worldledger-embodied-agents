"""在 Blender 内执行:读 spec(mesh 路径 + 4x4 世界矩阵),导入摆位、取景、渲染。

用法(由 visual_blender.py 调起): blender --background --python blender_render.py -- spec.json
产物:spec["image"] PNG + 同名 .result.json(导入计数与画幅覆盖)。
"""
import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix


def main():
    spec_path = Path(sys.argv[sys.argv.index("--") + 1])
    spec = json.loads(spec_path.read_text())

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    imported = 0
    for item in spec["items"]:
        path = item["mesh"]
        try:
            if path.lower().endswith(".stl"):
                try:
                    bpy.ops.wm.stl_import(filepath=path)      # Blender 4.x
                except AttributeError:
                    bpy.ops.import_mesh.stl(filepath=path)    # Blender 3.x
            elif path.lower().endswith((".dae",)):
                bpy.ops.wm.collada_import(filepath=path)
            elif path.lower().endswith((".obj",)):
                bpy.ops.wm.obj_import(filepath=path)
            else:
                continue
        except Exception:
            continue
        obj = bpy.context.selected_objects[0] if bpy.context.selected_objects else None
        if obj is None:
            continue
        M = Matrix([item["matrix"][0:4], item["matrix"][4:8],
                    item["matrix"][8:12], item["matrix"][12:16]])
        obj.matrix_world = M
        sx, sy, sz = item.get("scale", [1, 1, 1])
        obj.scale = (obj.scale[0] * sx, obj.scale[1] * sy, obj.scale[2] * sz)
        imported += 1

    # 相机:包围盒取景(斜前方 45°),灯光一盏太阳光
    xs, ys, zs = [], [], []
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            world = obj.matrix_world @ Matrix.Translation(corner).to_translation()
            xs.append(world.x); ys.append(world.y); zs.append(world.z)
    if xs:
        cx, cy, cz = (max(xs)+min(xs))/2, (max(ys)+min(ys))/2, (max(zs)+min(zs))/2
        size = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs), 0.5)
        cam_data = bpy.data.cameras.new("cam")
        cam = bpy.data.objects.new("cam", cam_data)
        scene.collection.objects.link(cam)
        cam.location = (cx + size*1.8, cy - size*1.8, cz + size*0.9)
        direction = Matrix.Translation((cx, cy, cz)).to_translation() - cam.location
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        scene.camera = cam
        sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
        scene.collection.objects.link(sun)
        sun.rotation_euler = (0.6, 0.2, 0.4)

    # 引擎按本机 Blender 版本能力选:EEVEE Next(4.2+)→ EEVEE(≤4.1)→ Workbench
    avail = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys()
    for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH"):
        if engine in avail:
            scene.render.engine = engine
            break
    scene.render.resolution_x, scene.render.resolution_y = 1280, 720
    scene.render.filepath = spec["image"]
    bpy.ops.render.render(write_still=True)

    # 画幅覆盖:所有包围盒角点投到相机 NDC,统计在画幅内的比例
    in_frame = total = 0
    if xs and scene.camera:
        from bpy_extras.object_utils import world_to_camera_view
        for obj in scene.objects:
            if obj.type != "MESH":
                continue
            for corner in obj.bound_box:
                co = obj.matrix_world @ Matrix.Translation(corner).to_translation()
                ndc = world_to_camera_view(scene, scene.camera, co)
                total += 1
                if 0 <= ndc.x <= 1 and 0 <= ndc.y <= 1 and ndc.z > 0:
                    in_frame += 1
    result = {"imported": imported,
              "in_frame_ratio": round(in_frame / total, 4) if total else None}
    spec_path.with_name(spec_path.name.replace(".spec.json", ".result.json")).write_text(
        json.dumps(result), encoding="utf-8")


main()
