import os
import shutil

# ================= Configuration =================
ROOT_DIR = r"D:\Workplace\Papers\TPMS\dataset_generate_FEA"
DATASET_DIR = os.path.join(ROOT_DIR, "dataset_fea")
TARGET_FILENAME = "model.inp"
DELETE_FAILED = True  # 如果设为 True，将删除未生成 inp 的文件夹
# ================================================

def check_inp_files():
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory not found at {DATASET_DIR}")
        return

    print(f"Checking for '{TARGET_FILENAME}' in: {DATASET_DIR}\n")
    
    subdirs = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    subdirs.sort()
    
    total_samples = len(subdirs)
    success_count = 0
    missing_samples = []

    for subdir in subdirs:
        inp_path = os.path.join(DATASET_DIR, subdir, TARGET_FILENAME)
        if os.path.exists(inp_path):
            success_count += 1
        else:
            missing_samples.append(subdir)

    print(f"Summary:")
    print(f"  Total directories found: {total_samples}")
    print(f"  Successfully generated:  {success_count}")
    print(f"  Missing (failed):        {len(missing_samples)}")
    
    if missing_samples:
        print("\nMissing Samples (First 10):")
        for s in missing_samples[:10]:
            print(f"  - {s}")
        if len(missing_samples) > 10:
            print(f"  ... and {len(missing_samples) - 10} more.")
        
        if DELETE_FAILED:
            print(f"\n[Cleaning] Deleting {len(missing_samples)} failed folders...")
            for s in missing_samples:
                folder_path = os.path.join(DATASET_DIR, s)
                try:
                    shutil.rmtree(folder_path)
                except Exception as e:
                    print(f"  [Error] Failed to delete {s}: {e}")
            print("[OK] Cleanup complete.")
        else:
            print("\n(Note: Set DELETE_FAILED = True in the script if you want to remove these folders automatically.)")
    else:
        print("\nAll folders have successfully generated the .inp file!")

if __name__ == "__main__":
    check_inp_files()
