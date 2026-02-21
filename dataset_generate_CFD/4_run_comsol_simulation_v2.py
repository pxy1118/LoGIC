import mph
import os
import pandas as pd
import time
import sys
import gc

# ================= 配置区域 =================
# 1. 路径配置
# 脚本所在目录 (也是 dataset_ml 的父目录)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(ROOT_DIR, "dataset_ml")
TEMPLATE_PATH = os.path.join(ROOT_DIR, "fluid.mph")
# 仿真使用的输入文件名 (3-matic 导出的文件)
STL_FILENAME = "model_fluid_remeshed.stl" 

# 2. COMSOL 模型内部标签配置
COMP_TAG = "comp1"
MESH_TAG = "mesh1"
IMPORT_NODE_TAG = "imp1" 
STUDY_TAG = "std1"
EVALUATION_TAG = "gev1"  # 全局计算 1
TABLE_TAG = "tbl1"       # 表格 1
PLOT_VEL_TAG = "pg1"     # 速度图
PLOT_PRE_TAG = "pg2"     # 压力图

# 3. 结果提取配置
OUTPUT_CSV = "simulation_results.csv"

# ============================================

def export_plot_image(model, plot_tag, output_path, label="图像"):
    """
    导出指定绘图组的图像
    """
    temp_export_tag = f"img_export_{plot_tag}"
    try:
        # 如果已存在先删除（避免冲突）
        try:
            model.java.result().export().remove(temp_export_tag)
        except:
            pass

        # 创建导出节点 (Image3D 或 Image2D，取决于绘图类型，这里假设 Image3D 通用性较好，
        # 或者直接使用 Image，COMSOL API 中通常是 "Image" 对应 2D/3D)
        # 注意: COMSOL API 可能是 "Image2D", "Image3D" 或 "Image". 
        # 安全起见，对于三维模型通常用 "Image3D"，但也可能仅仅是 "Image"。
        # 尝试创建一个通用的图像导出
        exp = model.java.result().export().create(temp_export_tag, "Image3D")
        
        # 配置导出
        exp.set("sourceobject", plot_tag)
        exp.set("filename", output_path)
        # 修正: size 属性应为 "manualweb", "manualprint" 或 "current"
        # 使用 "manualweb" 以允许自定义分辨率 (width/height)
        exp.set("size", "manualweb")
        exp.set("unit", "px")
        # 修正: 显式转换为字符串，避免 Java 重载歧义 (Ambiguous overloads)
        exp.set("width", "1024")
        exp.set("height", "768")
        # 设为 automatic current view
        exp.set("view", "auto") 
        
        # 执行
        exp.run()
        print(f"    已保存{label}: {os.path.basename(output_path)}")
        
        # 清理
        model.java.result().export().remove(temp_export_tag)
        
    except Exception as e:
        print(f"    [警告] 导出{label}失败: {e}")

def process_sample(client, sample_dir, sample_id):
    # 0. 检查是否已完成
    # 依据: 如果 velocity.png 和 pressure.png 都存在，且 simulation_results.csv 中包含该ID (可选)，则跳过
    # 为了简化，这里只检查图片是否存在，或者您可以检查 CSV
    vel_path = os.path.join(sample_dir, "velocity.png")
    pre_path = os.path.join(sample_dir, "pressure.png")
    
    if os.path.exists(vel_path) and os.path.exists(pre_path):
        print(f"  [跳过] 样本 {sample_id} 结果图已存在。")
        # 尝试返回一个空结果或标记，以便主循环知道怎么处理
        # 但为了让 CSV 数据完整，其实最好是读取 CSV 中该行的数据并返回
        # 这里直接返回 None 让主循环跳过写入
        return None

    stl_path = os.path.join(sample_dir, STL_FILENAME)
    if not os.path.exists(stl_path):
        print(f"  [跳过] STL文件不存在: {stl_path}")
        return None

    print(f"--------------------------------------------------")
    print(f"处理样本: {sample_id}")
    
    start_time = time.time()
    
    # 加载模型
    model = client.load(TEMPLATE_PATH)
    
    try:
        # === 核心操作流程 ===
        
        # 1. 设置 STL 路径
        try:
            # 确保路径格式正确 (使用绝对路径 + 正斜杠，避免 Windows 反斜杠转义问题)
            stl_path_safe = os.path.abspath(stl_path).replace("\\", "/")
            
            mesh = model.java.component(COMP_TAG).mesh(MESH_TAG)
            imp_node = mesh.feature(IMPORT_NODE_TAG)
            print(f"  [1/3] 设置导入文件: {stl_path_safe}")
            
            # 强制设置文件名
            imp_node.set("filename", stl_path_safe)
            
            # 关键为了确保重新导入：先清除旧网格?
            # 很多时候仅仅 set filename 不足以让 COMSOL 认为状态已改变，
            # 特别是如果它认为这个 import 节点已经 built 且 up-to-date。
            # 我们尝试先运行 import 节点 (如果它支持 independent run) 或者清除网格。
            
        except Exception as e_node:
            raise Exception(f"无法访问导入节点 {COMP_TAG}/{MESH_TAG}/{IMPORT_NODE_TAG}: {e_node}")
        
        # 2. 重建网格
        print("  [2/3] 重建网格 (Import & Meshing)...")
        # 显式清除网格，强制重新构建
        # 注意: 某些 API 版本可能是 clear() 或 clearMesh()
        # 这里尝试直接运行导入节点，然后运行整体
        try:
            # 尝试导入数据 (force re-read)
            mphtag = model.java.component(COMP_TAG).mesh(MESH_TAG).feature(IMPORT_NODE_TAG)
            mphtag.run() 
        except:
            pass

        # 运行整个网格序列
        model.java.component(COMP_TAG).mesh(MESH_TAG).run()
        
        # 检查网格统计信息以验证是否更新 (例如单元数量)
        try:
            stats = model.java.component(COMP_TAG).mesh(MESH_TAG).stat()
            # 获取四面体单元数 (或总单元数)
            elem_count = stats.getNumElem()
            print(f"    当前网格单元数: {elem_count}")
        except:
             print("    (无法获取网格统计信息)")

        print("  网格构建完成!")
        
        # 3. 求解
        print("  [3/3] 开始求解 (Run Study)...")
        
        # --- 关键修改: 强制清除旧解 (Force Clear Solution) ---
        try:
            sols = model.java.sol().tags()
            for s_tag in sols:
                model.java.sol(s_tag).clearSolution()
                print(f"    清除旧解: {s_tag}")
        except Exception as e_clear:
            print(f"    [提示] 清除旧解时忽略: {e_clear}")
        
        model.java.study(STUDY_TAG).run()
        print("  求解成功!")

        # 3.1 结果源更新 & 检查
        sol_dset = "dset1"
        try:
            # 尝试确保 dset1 指向最新的 sol1
            # 有时 dset 可能指向了 "None" 或其他
            ds = model.java.result().dataset(sol_dset)
            ds.set("solution", "sol1") 
        except:
            pass
            
        # 4. 结果提取 (改为直接计算，绕过表格缓存)
        print("  [4/4] 提取评估结果 (Direct Evaluation)...")
        results = {"sample_id": sample_id}
        
        try:
            # 内部函数: 安全转换标量
            def safe_extract(v):
                # 如果是 numpy 数组或标量
                if hasattr(v, 'item'):
                    try:
                        return v.item()
                    except ValueError:
                        # 针对 non-scalar numpy array，尝试取第一个
                        if hasattr(v, '__getitem__'): return v[0]
                # 如果是列表/元组
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    return v[0]
                return v

            # 1. 渗透率 (kappa)
            # evaluate 返回值可能是 0-d array, 1-d array, list 或 float
            raw_kappa = model.evaluate("kappa")
            val_kappa = safe_extract(raw_kappa)
            results["渗透率 (m^2)"] = val_kappa
            print(f"    渗透率 (m^2): {val_kappa}")
            
            # 2. 压降 (dPdL)
            raw_dp = model.evaluate("dPdL")
            val_dp = safe_extract(raw_dp)
            results["压降 (N/m^3)"] = val_dp
            print(f"    压降 (N/m^3): {val_dp}")
            
        except Exception as e_res:
            print(f"    [警告] 直接评估失败 ({e_res})")
            # 备用方案: 尝试使用 evaluate 处理算子
            try:
                 print("    尝试备用方法...")
                 k = model.evaluate("abs(comp1.aveop1(p))") # 仅作示例
                 # ...
            except:
                 pass

        # 5. 导出图像


        # 5. 导出图像
        
        # 5.1 更新外壁 (surf1) 选择范围: 全选 - 入口 - 出口
        try:
            print("  [5/0] 更新外壁显示范围 (Remove Inlet/Outlet)...")
            surf = model.java.result().dataset("surf1")
            
            # 1. 全选
            surf.selection().all()
            
            # 2. 获取入口和出口的边界ID
            # 优先尝试从耦合算子获取 (因为它们被用于计算，标签明确)
            # 备选: 从物理场特征获取 (inl1, out1)
            inlet_ids = None
            outlet_ids = None
            
            try:
                # 尝试通过 Label 查找算子("入口", "出口")，或者已知标签 aveop1/aveop2
                # 这里使用硬编码标签，根据之前的 inspect 结果
                inlet_ids = model.java.component(COMP_TAG).cpl("aveop1").selection().entities()
                outlet_ids = model.java.component(COMP_TAG).cpl("aveop2").selection().entities()
            except:
                print("    [提示] 无法从 aveop 获取边界，尝试从物理场获取...")
            
            if inlet_ids is None or len(inlet_ids) == 0:
                 try:
                    inlet_ids = model.java.component(COMP_TAG).physics("spf").feature("inl1").selection().entities()
                 except: pass

            if outlet_ids is None or len(outlet_ids) == 0:
                 try:
                    outlet_ids = model.java.component(COMP_TAG).physics("spf").feature("out1").selection().entities()
                 except: pass
                 
            # 3. 剔除
            if inlet_ids is not None and len(inlet_ids) > 0:
                surf.selection().remove(inlet_ids)
                
            if outlet_ids is not None and len(outlet_ids) > 0:
                surf.selection().remove(outlet_ids)
                
        except Exception as e_surf:
            print(f"    [警告] 更新外壁选择失败: {e_surf}")

        print("  [5/5] 导出流场图像...")
        export_plot_image(model, PLOT_VEL_TAG, os.path.join(sample_dir, "velocity.png"), "流速图")
        export_plot_image(model, PLOT_PRE_TAG, os.path.join(sample_dir, "pressure.png"), "压力图")

        print(f"  样本耗时: {time.time() - start_time:.1f}s")
        return results

    except Exception as e:
        print(f"  [错误] 仿真失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        try:
            # 释放模型占用内存
            if 'model' in locals() and model:
                client.remove(model)
        except Exception as e_rm:
            print(f"    [警告] 释放模型内存失败: {e_rm}")
        
        # 强制Python垃圾回收
        gc.collect()

def main():
    print("=== COMSOL 批量仿真脚本 v2.0 ===")
    print(f"模板路径: {TEMPLATE_PATH}")
    print(f"数据目录: {DATASET_DIR}")
    
    if not os.path.exists(TEMPLATE_PATH):
        print("错误: 找不到模板文件。")
        return

    # 获取样本列表
    if not os.path.exists(DATASET_DIR):
        print("错误: 数据集目录不存在。")
        return

    subdirs = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
    # 简单过滤，确保是数字命名的样本文件夹
    subdirs = [d for d in subdirs if d.isdigit()]
    
    if not subdirs:
        print("未找到有效样本文件夹 (纯数字命名)。")
        return

    print(f"发现 {len(subdirs)} 个样本待处理。")
    print("正在连接/启动 COMSOL Server...")
    
    try:
        # cores=None 让 COMSOL 自行决定，或者设置为具体数字限制并发
        client = mph.start(cores=12)
    except Exception as e:
        print(f"启动失败: {e}")
        print("请检查 COMSOL 安装及是否在 PATH 环境变量中。")
        return

    print("COMSOL 准备就绪。")
    
    all_results = []
    
    # 策略更新: 如果 CSV 已存在，加载它，避免覆盖通过"跳过"逻辑导致的数据丢失
    csv_path = os.path.join(DATASET_DIR, OUTPUT_CSV)
    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            # 将 DataFrame 转回 dist 列表
            all_results = existing_df.to_dict('records')
            # 确保 sample_id 是字符串格式 (如果有前导零)
            for r in all_results:
                r['sample_id'] = f"{int(r['sample_id']):04d}"
            print(f"已加载现有结果: {len(all_results)} 条记录")
        except Exception as e:
            print(f"加载现有 CSV 失败: {e}")

    try:
        for idx, sample_id in enumerate(subdirs):
            sample_dir = os.path.join(DATASET_DIR, sample_id)
            
            # 检查是否已在 CSV 中存在 (可选的双重检查)
            already_processed = any(r['sample_id'] == sample_id for r in all_results)
            
            if already_processed:
                # 即使图片不存在，如果数据在 CSV 里了，我们也跳过仿真
                # (或者您可以选择重新跑以补全图片，这里假设一致性)
                # print(f"样本 {sample_id} 已记录在 CSV 中，跳过。")
                
                # 还有一种情况: CSV 有记录但没图片，想补图片? 
                # 这里 process_sample 里有图片检查。
                # 如果 process_sample 返回 None (因图片存在跳过)，我们不需要做任何事
                # 如果 process_sample 不返回 None (CSV有记录但没图片?) -> 这是一个边界情况
                # 简单起见，如果 CSV 有记录，我们就不跑 process_sample 了
                continue 
            
            # --- 执行仿真 ---
            res = process_sample(client, sample_dir, sample_id)
            
            if res:
                all_results.append(res)
                # 实时保存 CSV
                df = pd.DataFrame(all_results)
                # 确保 sample_id 排序
                df = df.sort_values(by="sample_id")
                df.to_csv(csv_path, index=False)
                
    except KeyboardInterrupt:
        print("\n[用户中断] 正在停止...")
    except Exception as e_main:
        print(f"\n[系统错误] {e_main}")
    finally:
        print("正在断开 COMSOL...")
        # client 析构会自动处理断开，显式调用也可
        # client.disconnect() 

    print(f"\n=== 处理结束 ===")
    print(f"成功: {len(all_results)} / {len(subdirs)}")

if __name__ == "__main__":
    main()
