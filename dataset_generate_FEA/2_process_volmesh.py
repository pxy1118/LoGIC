# 使用方法：在3-matic中运行此脚本，以批量为数据集中的remeshed fluid STL生成高质量体积网格，并导出为Abaqus .inp文件（model.inp）
# 本脚本参考官方Femur例程（wrap → smooth → remesh → auto_fix → create_volume_mesh），在您的remeshed表面网格基础上进一步优化
# 优势：
# - wrap：闭合微小间隙、修复潜在缺陷
# - smooth：平滑表面，提高网格质量
# - adaptive remesh：统一边长、进一步优化表面
# - auto_fix：自动修复剩余网格问题（强烈推荐，用于确保create_volume_mesh成功率）
# - 已验证有效的导出方式（part.name = "model" + output_directory）

import os

try:
    import trimatic
except ImportError:
    print("Warning: 'trimatic' module not found. Please run this script inside Materialise 3-matic.")
    class DummyTrimatic:
        def new_project(self): pass
        def delete(self, entities): pass
        def get_parts(self): return []
        def import_part_stl(self, path): return None
        def duplicate(self, entities): return []
        def wrap(self, **kwargs): return None
        def smooth(self, **kwargs): pass
        def adaptive_remesh(self, **kwargs): return None
        def auto_fix(self, **kwargs): pass
        def create_volume_mesh(self, **kwargs): pass
        def export_abaqus(self, **kwargs): pass
    trimatic = DummyTrimatic()

# ================= Configuration =================
# 自动获取当前脚本所在目录作为根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = SCRIPT_DIR
DATASET_DIR = os.path.join(ROOT_DIR, "dataset_fea")
INPUT_FILENAME = "model_solid.stl"   
OUTPUT_FILENAME = "model.inp"
FORCE_REGENERATE = False  # 强制重新生成

# 处理链参数（请根据模型实际情况调整，参考官方例程）
WRAP_GAP_CLOSING_DISTANCE = 0.1     # mm，闭合间隙距离（太大可能改变几何）
WRAP_SMALLEST_DETAIL = 0.2           # mm，保留的最小特征尺寸
# SMOOTH_FACTOR = 0.2                  # 已禁用平滑
REMESH_TARGET_EDGE_LENGTH = 0.3       # mm，表面网格目标边长（建议比volume稍小）
PRESERVE_CONTOURS = True             # 是否保留表面轮廓（fluid域建议False以保持尖锐特征，可测试）
MAXIMUM_EDGE_LENGTH_VOLUME = 0.6     # mm，体积网格最大边长（建议为表面边长的1.5~2倍）
# ============================================

def process_sample(sample_dir):
    stl_path = os.path.normpath(os.path.join(sample_dir, INPUT_FILENAME))
    output_path = os.path.normpath(os.path.join(sample_dir, OUTPUT_FILENAME))
    
    if not FORCE_REGENERATE and os.path.exists(output_path):
        print(f"  [Skipped] 已存在: {output_path}")
        return

    if not os.path.exists(stl_path):
        print(f"  [Skipped] 未找到输入文件: {stl_path}")
        return

    print("--------------------------------------------------")
    print(f"Processing Folder: {os.path.basename(sample_dir)}")
    print(f"  Input: {stl_path}")
    print(f"  Output will be: {output_path}")

    # 1. 清空项目
    try:
        if hasattr(trimatic, 'new_project'):
            trimatic.new_project()
        parts = trimatic.get_parts()
        if parts:
            trimatic.delete(parts)
    except:
        pass

    # 2. 导入remeshed表面网格
    imported = trimatic.import_part_stl(stl_path)
    base_part = imported[0] if isinstance(imported, (list, tuple)) and imported else imported
    if base_part is None:
        print("  [Error] 导入STL失败")
        return
    base_part.name = "Base_Remeshed"
    print("  已导入基础表面网格")

    # 3. Duplicate → Wrap（闭合间隙）
    print("  Step 1: Wrap（闭合间隙）...")
    wrapped_part = trimatic.duplicate(base_part)
    wrapped_part = wrapped_part[0] if isinstance(wrapped_part, (list, tuple)) else wrapped_part
    wrapped_part = trimatic.wrap(
        entities=wrapped_part,
        gap_closing_distance=WRAP_GAP_CLOSING_DISTANCE,
        smallest_detail=WRAP_SMALLEST_DETAIL
    )
    if wrapped_part:
        wrapped_part.name = "Wrapped"
        print("  [Success] Wrap完成")
    else:
        print("  [Error] Wrap失败，跳过后续步骤")
        return

    # 4. (Skipped) Smooth
    # print("  Step 2: Smooth（平滑）...")
    # smoothed_part = trimatic.duplicate(wrapped_part)
    # smoothed_part = smoothed_part[0] if isinstance(smoothed_part, (list, tuple)) else smoothed_part
    # smoothed_part.name = "Smoothed"
    # trimatic.smooth(entities=smoothed_part, smooth_factor=SMOOTH_FACTOR)
    # print("  [Success] Smooth完成")

    # 5. Duplicate → Adaptive Remesh（再次优化表面网格）
    print(f"  Step 3: Adaptive Remesh（目标边长 {REMESH_TARGET_EDGE_LENGTH} mm）...")
    remeshed_part = trimatic.duplicate(wrapped_part) # 使用 wrapped_part 替代 smoothed_part
    remeshed_part = remeshed_part[0] if isinstance(remeshed_part, (list, tuple)) else remeshed_part
    remeshed_part.name = "Remeshed_Final"
    remeshed_part = trimatic.adaptive_remesh(
        entities=remeshed_part,
        target_triangle_edge_length=REMESH_TARGET_EDGE_LENGTH,
        preserve_surface_contours=PRESERVE_CONTOURS
    )
    if remeshed_part:
        print("  [Success] Adaptive Remesh完成")
    else:
        print("  [Error] Remesh失败，可能需要检查输入")

    # 6. Duplicate → Auto Fix（自动修复网格问题，强烈推荐）
    print("  Step 4: Auto Fix（修复网格缺陷）...")
    # 这里原来的逻辑是 remeshed_part or smoothed_part，现在 smoothed_part 没了
    input_for_fix = remeshed_part if remeshed_part else wrapped_part
    fixed_part = trimatic.duplicate(input_for_fix)
    fixed_part = fixed_part[0] if isinstance(fixed_part, (list, tuple)) else fixed_part
    fixed_part.name = "Fixed"
    trimatic.auto_fix(entities=fixed_part)
    print("  [Success] Auto Fix完成")

    # 7. Duplicate → Create Volume Mesh
    print(f"  Step 5: Create Volume Mesh（max edge {MAXIMUM_EDGE_LENGTH_VOLUME} mm）...")
    vol_part = trimatic.duplicate(fixed_part)
    vol_part = vol_part[0] if isinstance(vol_part, (list, tuple)) else vol_part
    vol_part.name = "Mesh-archive"  # 导出时文件名为 Mesh-archive.inp
    try:
        trimatic.create_volume_mesh(
            part=vol_part,
            maximum_edge_length=MAXIMUM_EDGE_LENGTH_VOLUME,
            # 如需二次单元：element_order='quadratic'
        )
        print("  [Success] 体积网格创建完成")
    except Exception as e:
        print("  [Error] 体积网格创建失败: " + str(e))
        return

    # 8. 导出Abaqus .inp
    print("  Step 6: 导出Abaqus .inp ...")
    try:
        trimatic.export_abaqus_single_output(
            [vol_part], 
            output_path, 
            False, 
            trimatic.TypeOfElement.element_type_C3D10, 
            None, 
            trimatic.FaceSplitMethod.split_by_part
        )
        print(f"  [Success] 已导出: {output_path}")
    except Exception as e:
        print("  [Error] 导出失败: " + str(e))

def main():
    print("=== 3-matic 批量高质量体积网格生成（参考官方Femur例程增强版） ===")
    print("数据集路径: " + DATASET_DIR)
    
    if not os.path.exists(DATASET_DIR):
        print("Error: 数据集目录不存在")
        return

    subdirs = [os.path.join(DATASET_DIR, d) for d in os.listdir(DATASET_DIR) 
               if os.path.isdir(os.path.join(DATASET_DIR, d))]
    subdirs.sort()
    
    print(f"发现 {len(subdirs)} 个样本")
    
    for sample_dir in subdirs:
        process_sample(sample_dir)

    print("\n=== 批量处理完成 ===")

if __name__ == "__main__":
    main()