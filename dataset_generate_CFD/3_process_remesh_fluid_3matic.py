#使用方法：在3-matic中运行此脚本以批量处理数据集中的STL文件。
import os
import sys
import glob

# Try to import trimatic (works if running inside 3-matic)
# If running externally, you might need rpyc connection, but this script assumes
# it is being executed by the 3-matic Python interpreter.
try:
    import trimatic
except ImportError:
    print("Warning: 'trimatic' module not found. Please run this script inside Materialise 3-matic.")
    print("Usage: 3-matic.exe -run_script \"<path_to_this_script>\"")
    # For coding assistance, we'll define a dummy trimatic to prevent linter errors if viewed in IDE
    class DummyTrimatic:
        def open_project(self, path): pass
        def import_part_stl(self, path): return []
        def adaptive_remesh(self, entities, **kwargs): pass
        def uniform_remesh(self, entities, **kwargs): pass
        def export_stl_binary(self, entities, **kwargs): pass
    trimatic = DummyTrimatic()

# ================= Configuration =================
# Define paths (Using absolute paths is safer for 3-matic)
ROOT_DIR = r"D:\Workplace\Papers\TPMS\dataset_generate"
DATASET_DIR = os.path.join(ROOT_DIR, "dataset_ml")
INPUT_FILENAME = "model_fluid.stl"
OUTPUT_FILENAME = "model_fluid_remeshed.stl"

# Remesh Parameters (Adjust these based on your mesh requiremens)
# Target triangle edge length in mm
TARGET_EDGE_LENGTH = 0.15
# ============================================

def process_sample(sample_dir):
    stl_path = os.path.join(sample_dir, INPUT_FILENAME)
    output_path = os.path.join(sample_dir, OUTPUT_FILENAME)
    
    if os.path.exists(output_path):
        print("  [Skipped] Remeshed STL already exists: " + output_path)
        return

    if not os.path.exists(stl_path):
        print("  [Skipped] Source STL not found: " + stl_path)
        return

    print("--------------------------------------------------")
    print("Processing: " + sample_dir)

    # 1. Initialize
    # As requested: Do not load template. 
    # To prevent memory accumulation in batch, we try to clear parts or create new project
    try:
        # Generic attempt to start fresh in 3-matic
        if hasattr(trimatic, 'new_project'):
            trimatic.new_project()
        elif hasattr(trimatic, 'delete') and hasattr(trimatic, 'get_parts'):
            parts = trimatic.get_parts()
            if parts:
                trimatic.delete(parts)
    except:
        pass

    # 2. Import Fluid STL
    imported_parts = trimatic.import_part_stl(stl_path)
    
    # Handle API differences: newer versions might return a single object instead of a list
    part = None
    if isinstance(imported_parts, (list, tuple)):
        if len(imported_parts) > 0:
            part = imported_parts[0]
    else:
        # If not a list, assume it is the part object, provided it is not None
        if imported_parts is not None:
            part = imported_parts

    if part is None:
        print("  [Error] Failed to import STL.")
        return
    
    part.name = "Fluid_Domain"
    print("  Imported part: " + part.name)

    # 3. Open Inspection Page (Simulated)
    # Note: Logic operations don't require the UI page to be active, 
    # but we can activate the Remesh menu context if needed.
    # trimatic.activate_remesh_page() 
    
    # 4. Adaptive Remesh (First Pass)
    print("  Step 1: Adaptive Remesh...")
    # Using parameters from user image: 0.2mm, Preserve contours=False
    try:
        trimatic.adaptive_remesh(
            entities=[part], 
            target_triangle_edge_length=TARGET_EDGE_LENGTH,
            preserve_surface_contours=False
        )
    except Exception as e:
        print("  [Error] Adaptive Remesh 1 failed: " + str(e))

    # 5. Uniform Remesh (Second Pass)
    print("  Step 2: Uniform Remesh...")
    # Using parameters from user image: 0.2mm, Preserve contours=False
    try:
        trimatic.uniform_remesh(
            entities=[part], 
            target_triangle_edge_length=TARGET_EDGE_LENGTH,
            preserve_surface_contours=False
        )
    except Exception as e:
        print("  [Error] Uniform Remesh failed: " + str(e))

    # 6. Adaptive Remesh (Third Pass)
    print("  Step 3: Adaptive Remesh...")
    try:
        trimatic.adaptive_remesh(
            entities=[part], 
            target_triangle_edge_length=TARGET_EDGE_LENGTH,
            preserve_surface_contours=False
        )
    except Exception as e:
        print("  [Error] Adaptive Remesh 2 failed: " + str(e))

    # 7. Export Processed STL
    # Strategy: Rename part to match desired filename and export to directory
    # This avoids keyword argument issues if the API signature varies
    target_name = os.path.splitext(OUTPUT_FILENAME)[0] # e.g., "model_fluid_remeshed"
    part.name = target_name
    
    try:
        # 1. Try generic export using part name + directory
        trimatic.export_stl_binary(
            entities=[part], 
            output_directory=sample_dir
        )
        print("  [Success] Exported to: " + output_path)
    except Exception as e1:
        # 2. If that fails (e.g. missing arg), try old style with filename pos/kw
        # print("  [Info] Method 1 failed (" + str(e1) + "), trying Method 2...")
        try:
            # Try positional args: entities, dir, filename
            trimatic.export_stl_binary([part], sample_dir, OUTPUT_FILENAME)
            print("  [Success] Exported to: " + output_path)
        except Exception as e2:
             print("  [Error] Export failed: " + str(e2))


def main():
    print("=== Batch Remeshing Script for 3-matic ===")
    print("Dataset Directory: " + DATASET_DIR)
    
    if not os.path.exists(DATASET_DIR):
        print("Error: Dataset directory does not exist.")
        return

    # Find sample directories (assuming dataset_ml/0000 etc.)
    # Using os.listdir to avoid dependency issues in restricted python environments
    subdirs = []
    if os.path.exists(DATASET_DIR):
        for d in os.listdir(DATASET_DIR):
            full_path = os.path.join(DATASET_DIR, d)
            if os.path.isdir(full_path):
                subdirs.append(full_path)
    
    subdirs.sort()
    
    print("Found " + str(len(subdirs)) + " samples.")
    
    for sample_dir in subdirs:
        process_sample(sample_dir)

    print("\n=== Batch Processing Completed ===")

if __name__ == "__main__":
    main()
