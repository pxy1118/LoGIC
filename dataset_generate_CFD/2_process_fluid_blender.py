# 使用方法   blender --background --python 2_process_fluid_blender.py


import bpy
import os
import sys

# ================= 配置区域 =================
# 脚本假设在 dataset_generate 目录下运行，或者 dataset_ml 与脚本在同级目录
# 如果需要指定绝对路径，请修改下面的 DATASET_DIR
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
DATASET_DIR = os.path.join(CURRENT_DIR, "dataset_ml")

FLUID_BOX_SIZE = (23.8, 23.8, 25.0)  # 24x24x25
FLUID_BOX_CENTER = (12.0, 12.0, 12.0) # 中心坐标
SOLID_FILENAME = "model_solid.stl"
OUTPUT_FILENAME = "model_fluid.stl"
# ============================================

def clear_scene():
    """清空场景中的所有对象"""
    if bpy.context.active_object and bpy.context.active_object.mode == 'EDIT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def process_sample(sample_dir):
    stl_path = os.path.join(sample_dir, SOLID_FILENAME)
    output_path = os.path.join(sample_dir, OUTPUT_FILENAME)
    
    if os.path.exists(output_path):
        print(f"  [跳过] 流体模型已存在: {output_path}")
        return

    if not os.path.exists(stl_path):
        print(f"  [跳过] 找不到固体模型: {stl_path}")
        return

    # 1. 导入 Solid STL
    # 优先尝试 Blender 4.2+ 新版 API
    try:
        bpy.ops.wm.stl_import(filepath=stl_path)
    except (AttributeError, RuntimeError):
        try:
            # 回退到旧版 API
            bpy.ops.import_mesh.stl(filepath=stl_path)
        except Exception as e:
            print(f"  [错误] 导入失败: {e}")
            return

    # 选中导入的对象（假设是选中的第一个）
    if not bpy.context.selected_objects:
        # 有时新版导入可能选中了 Collection，或者没有 active object
        # 尝试选择 scene 中的所有 mesh
        for obj in bpy.context.scene.objects:
             if obj.type == 'MESH':
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
                break
       
    solid_obj = bpy.context.selected_objects[0]
    solid_obj.name = "TPMS_Solid"
    
    # 2. 创建流体域 Box (24x24x25)
    # 位置逻辑：
    # Solid 是 24x24x24，坐标范围 [0, 24]
    # Box 是 24x24x25，需要上下各留 0.5mm，即 Z 范围 [-0.5, 24.5]
    # Box 的 Z 中心 = (-0.5 + 24.5) / 2 = 12.0
    # Box 的 X,Y 中心 = 24 / 2 = 12.0
    # 所以 Box Center = (12, 12, 12)
    
    bpy.ops.mesh.primitive_cube_add(
        size=1.0, 
        location=FLUID_BOX_CENTER
    ) 
    fluid_box = bpy.context.active_object
    fluid_box.name = "Fluid_Domain"
    fluid_box.dimensions = FLUID_BOX_SIZE
    
    # 应用缩放变换，确保布尔运算准确
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # 3. 添加 Boolean Modifier (Difference)
    bool_mod = fluid_box.modifiers.new(name="Boolean_Diff", type='BOOLEAN')
    bool_mod.operation = 'DIFFERENCE'
    bool_mod.object = solid_obj
    bool_mod.solver = 'EXACT' # 'FAST' 有时更快但可能产生非流形几何，'EXACT' 更稳健

    # 4. 应用 Modifier
    bpy.context.view_layer.objects.active = fluid_box
    bpy.ops.object.modifier_apply(modifier="Boolean_Diff")

    # 5. 删除原来的 Solid 对象，只保留流体域
    bpy.data.objects.remove(solid_obj, do_unlink=True)

    # 6. 导出 Fluid STL
    try:
        # 优先尝试新版 API
        bpy.ops.wm.stl_export(filepath=output_path, export_selected_objects=True)
    except (AttributeError, RuntimeError):
        try:
            # 回退到旧版 API
            bpy.ops.export_mesh.stl(filepath=output_path, selection_only=True)
        except Exception as e:
            print(f"  [错误] 导出失败: {e}")

    print(f"  [完成] 导出流体域: {output_path}")

def main():
    print("=== 开始批量生成流体域 STL (Blender Script) ===")
    
    if not os.path.exists(DATASET_DIR):
        print(f"错误: 数据集目录不存在: {DATASET_DIR}")
        return

    # 遍历 dataset_ml 下的所有子文件夹
    subdirs = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    subdirs.sort() # 按名称排序，通常是 0000, 0001...
    
    print(f"找到 {len(subdirs)} 个样本待处理，路径: {DATASET_DIR}")
    
    count = 0
    import time
    start_time = time.time()

    for subdir in subdirs:
        sample_dir = os.path.join(DATASET_DIR, subdir)
        # 简单过滤，只处理包含数字的目录（dataset_ml/0000）
        print(f"处理样本: {subdir} ...")
        
        clear_scene()
        process_sample(sample_dir)
        count += 1
        
    print(f"\n=== 全部处理完成! 共处理 {count} 个样本，耗时: {time.time() - start_time:.2f}s ===")

if __name__ == "__main__":
    main()
