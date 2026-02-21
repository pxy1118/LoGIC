# -*- coding: utf-8 -*-
"""
批量运行TPMS有限元RF-位移分析脚本
--------------------------------------------------
思路:
 1. 遍历 dataset 下所有包含 model.inp 的子目录 (支持任意层级/命名)
 2. 使用普通 Python 直接调用单次脚本: python abaqus_auto.py -- inp=<绝对路径>
    - 每个样本以其子目录为工作目录, 输出写入该目录下 output/
 3. 判断是否成功 (是否存在 output/rf_displacement_curve.csv)
 4. 读取该 CSV 的最大位移与最大反力 汇总到 batch_results_summary.csv
 5. 支持命令行参数: --root, --start, --end, --redo, --limit, --sleep, --python, --samples

新增功能:
 6. 运行开始前自动清理目录，只保留必需文件
 7. 处理结束后自动清理临时文件
 8. 使用 tqdm 显示实时分析进度条（通过读取 Job-1.sta 和 Job-1.dat）
 9. 保留机器学习训练所需的核心文件

使用示例 (在 test 目录下):
  python batch_run_abaqus.py --root dataset --start 1 --end 100
  python batch_run_abaqus.py --samples 1,5,20 --redo
  python batch_run_abaqus.py --python C:/Python39/python.exe --start 1 --end 10 --keep-odb

注意:
 - 当前模式不调用 Abaqus CAE GUI, 依赖 abaqus_auto.py 内部逻辑可在纯 Python 环境运行
 - 若需要恢复使用 Abaqus noGUI 模式, 可自行将 run_single_sample 中的命令换回
"""
from __future__ import annotations
import os
import sys
import csv
import time
import argparse
import subprocess
import shutil
import threading
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Set

# 尝试导入 tqdm
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("[WARN] 未安装 tqdm，将使用简单进度显示")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SINGLE_SCRIPT = os.path.join(SCRIPT_DIR, 'abaqus_auto.py')
DEFAULT_DATA_ROOT = os.path.join(SCRIPT_DIR, 'dataset_fea')
SUMMARY_CSV = os.path.join(SCRIPT_DIR, 'results/batch_results_summary.csv')
LOG_FILE = os.path.join(SCRIPT_DIR, 'logs/batch_run.log')

# 机器学习训练必需保留的文件/目录
REQUIRED_FILES = {
    'model.stl',
    'model.inp',
    'voxel.npy',
    'input_density_3x3x3.npy',
    'input_rotation_3x3x3.npy',
    'input_weight_3x3x3.npy',
    'weight.npy',
    'model_solid.stl',
    'output_voxel.npy',
    'ref_cube.png',
    'meta.json',
    'output',  # 目录
    'preview.png',
    'density.npy',
    'fields.npz',
    # 'Job-1.odb'
}


@dataclass(frozen=True)
class SampleInfo:
    """描述一个待分析样本."""
    name: str          # 相对 root 的显示名称 (使用正斜杠)
    path: str          # 样本绝对路径
    numeric_id: Optional[int]  # 若目录名为纯数字, 保存其数值, 否则为 None

# -------------------------------------------------
# 工具函数
# -------------------------------------------------

def log(msg: str, console: bool = True):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f: f.write(line + '\n')
    except Exception:
        pass
    if console:
        print(line)


def parse_int_list(spec: str) -> List[int]:
    out = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a,b = part.split('-',1)
            try:
                a_i = int(a); b_i = int(b)
            except ValueError:
                continue
            if a_i <= b_i:
                out.extend(range(a_i, b_i+1))
            else:
                out.extend(range(a_i, b_i-1, -1))
        else:
            try:
                out.append(int(part))
            except ValueError:
                continue
    # 去重保持顺序
    seen = set(); res = []
    for v in out:
        if v not in seen:
            seen.add(v); res.append(v)
    return res


def select_samples_by_spec(samples: List[SampleInfo], spec: str) -> List[SampleInfo]:
    tokens = [t.strip() for t in spec.split(',') if t.strip()]
    numeric_targets: Set[int] = set(parse_int_list(spec))
    selected: List[SampleInfo] = []
    seen: Set[str] = set()

    def add_sample(sample: SampleInfo):
        if sample.name not in seen:
            selected.append(sample)
            seen.add(sample.name)

    by_numeric: Dict[int, List[SampleInfo]] = {}
    by_name: Dict[str, SampleInfo] = {}
    by_leaf: Dict[str, List[SampleInfo]] = {}
    for sample in samples:
        if sample.numeric_id is not None:
            by_numeric.setdefault(sample.numeric_id, []).append(sample)
        by_name[sample.name] = sample
        leaf = os.path.basename(sample.path)
        by_leaf.setdefault(leaf, []).append(sample)

    # 先处理数字/区间
    for num in sorted(numeric_targets):
        matches = by_numeric.get(num)
        if matches:
            for m in matches:
                add_sample(m)
        else:
            log(f"[WARN] 未找到编号为 {num} 的样本")

    # 再处理名称匹配
    range_pattern = re.compile(r'^\d+\s*-\s*\d+$')
    for token in tokens:
        if token.isdigit() or range_pattern.match(token.replace(' ', '')):
            # 纯数字或区间已处理
            continue
        norm = token.replace('\\', '/').strip('/')
        direct = by_name.get(norm)
        if direct:
            add_sample(direct)
            continue
        # 尝试按叶子目录名称匹配
        leaf_matches = by_leaf.get(token) or by_leaf.get(norm.split('/')[-1])
        if leaf_matches:
            for m in leaf_matches:
                add_sample(m)
            continue
        log(f"[WARN] 指定的样本未找到: {token}")

    return selected


def collect_samples(root: str, start: Optional[int], end: Optional[int]) -> List[SampleInfo]:
    samples: List[SampleInfo] = []
    if not os.path.isdir(root):
        log(f"[ERROR] 数据根目录不存在: {root}")
        return samples

    abs_root = os.path.abspath(root)
    for current_root, _dirs, files in os.walk(abs_root):
        if 'model.inp' not in files:
            continue

        rel_path = os.path.relpath(current_root, abs_root)
        if rel_path == '.':
            rel_path = os.path.basename(os.path.normpath(current_root)) or current_root
        display_name = rel_path.replace(os.sep, '/').strip('/')
        leaf = os.path.basename(current_root)
        numeric_id: Optional[int] = int(leaf) if leaf.isdigit() else None

        if (start is not None or end is not None) and numeric_id is None:
            # 启用了编号过滤但该目录无法转换为编号, 直接跳过
            continue
        if start is not None and numeric_id is not None and numeric_id < start:
            continue
        if end is not None and numeric_id is not None and numeric_id > end:
            continue

        samples.append(SampleInfo(name=display_name or leaf, path=current_root, numeric_id=numeric_id))

    samples.sort(key=lambda s: (s.numeric_id if s.numeric_id is not None else float('inf'), s.name))
    return samples


def read_abaqus_progress_from_sta(sta_file: str, total_time: float = 1.0) -> tuple[float, str]:
    """
    鲁棒解析 Explicit .sta 进度:
    直接扫描全部行, 捕获形如:
        <inc>  <step_time(E)> <total_time(E)>  <wall_time> <stable_dt(E)> ...
    的数据行 (第一列整数 + 至少两个科学计数)
    取最后一条数据行的 TOTAL TIME(第3列) 计算百分比.
    若在文件中能匹配 "of <TOTAL>" 则更新 total_time.
    可通过环境变量 PROGRESS_DEBUG=1 输出调试.
    """
    debug = os.environ.get('PROGRESS_DEBUG') == '1'
    try:
        if not os.path.exists(sta_file):
            return 0.0, "准备中..."
        with open(sta_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        # 提取总时间 (倒序寻找最新的 of X.XXXE+XX)
        for line in reversed(lines):
            m_total = re.search(r'\bof\s+([0-9]+\.[0-9]+E[+\-]?[0-9]+)', line)
            if m_total:
                try:
                    tt = float(m_total.group(1))
                    if tt > 0:
                        total_time = tt
                        break
                except ValueError:
                    pass
        data_pattern = re.compile(r'^\s*(\d+)\s+([0-9]+\.[0-9]+E[+\-]?[0-9]+)\s+([0-9]+\.[0-9]+E[+\-]?[0-9]+)')
        last_match = None
        for line in lines:
            m = data_pattern.match(line)
            if m:
                last_match = m
        if not last_match:
            return 0.0, "初始化..."
        try:
            current_total_time = float(last_match.group(3))  # 第3列 TOTAL TIME
        except ValueError:
            try:
                current_total_time = float(last_match.group(2))  # 回退第2列
            except ValueError:
                return 0.0, "初始化..."
        if total_time <= 0:
            return 0.0, f"时间 {current_total_time:.3e}/?"
        percent = min(100.0, (current_total_time / total_time) * 100.0)
        return percent, f"时间 {current_total_time:.3e}/{total_time:.3e}"
    except Exception as e:
        if debug:
            print(f"[DEBUG] 解析 .sta 异常: {e}")
        return 0.0, "监控中..."


def read_abaqus_progress_from_dat(dat_file: str, total_time: float = 1.0) -> tuple[float, str]:
    """
    从Abaqus .dat文件读取进度信息（备用方法）
    返回: (进度百分比, 状态描述)
    """
    try:
        if not os.path.exists(dat_file):
            return 0.0, "准备中..."
            
        with open(dat_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        # 从后往前查找最新的时间信息
        for line in reversed(lines):
            line = line.strip()
            
            # 查找进度行
            pattern = r'^\s*\d+\s+(\d+\.\d+E[+-]?\d+)\s+(\d+\.\d+E[+-]?\d+)'
            match = re.search(pattern, line)
            if match:
                current_time = float(match.group(1))
                progress_percent = min(100.0, (current_time / total_time) * 100)
                return progress_percent, f"时间 {current_time:.3e}/{total_time}"
                
        return 0.0, "初始化..."
        
    except Exception as e:
        return 0.0, f"监控中..."


def update_progress_tqdm(folder: str, pbar: tqdm, stop_event: threading.Event, total_time: float = 1.0):
    """使用 tqdm 更新进度"""
    sta_file = os.path.join(folder, 'Job-1.sta')
    dat_file = os.path.join(folder, 'Job-1.dat')
    last_percent = 0
    
    while not stop_event.is_set():
        try:
            # 优先从 .sta 文件读取更准确的进度
            progress_percent, status = read_abaqus_progress_from_sta(sta_file, total_time)
            
            # 如果 .sta 文件没有进度信息，尝试从 .dat 文件读取
            if progress_percent <= 0:
                progress_percent, status = read_abaqus_progress_from_dat(dat_file, total_time)
            
            # 确保进度不超过100%
            progress_percent = min(100.0, progress_percent)
            
            # 更新进度条
            pbar.n = int(progress_percent)
            pbar.set_postfix_str(status)
            pbar.refresh()
                
        except Exception as e:
            pass
            
        time.sleep(2)  # 每2秒检查一次


def update_progress_simple(folder: str, stop_event: threading.Event, total_time: float = 1.0):
    """简单进度显示（不依赖 tqdm）"""
    sta_file = os.path.join(folder, 'Job-1.sta')
    dat_file = os.path.join(folder, 'Job-1.dat')
    
    while not stop_event.is_set():
        try:
            # 优先从 .sta 文件读取更准确的进度
            progress_percent, status = read_abaqus_progress_from_sta(sta_file, total_time)
            
            # 如果 .sta 文件没有进度信息，尝试从 .dat 文件读取
            if progress_percent <= 0:
                progress_percent, status = read_abaqus_progress_from_dat(dat_file, total_time)
            
            # 确保进度不超过100%
            progress_percent = min(100.0, progress_percent)
            
            # 显示进度
            sys.stdout.write(f'\r[进度] {progress_percent:3.0f}% {status}')
            sys.stdout.flush()
                
        except Exception as e:
            pass
            
        time.sleep(2)
    
    # 清除进度行
    sys.stdout.write('\r' + ' ' * 80 + '\r')
    sys.stdout.flush()


def cleanup_sample_files(folder: str, keep_odb: bool = False, pre_run: bool = False):
    """清理样本目录中的临时文件，保留机器学习训练所需的核心文件"""
    
    # 机器学习训练必须保留的文件/目录
    keep_files = REQUIRED_FILES.copy()
    
    # 如果保留ODB文件，则也保留
    if keep_odb:
        keep_files.add('Job-1.odb')
    
    log(f"[CLEAN] {'预运行清理' if pre_run else '后处理清理'} 目录: {folder}")
    
    try:
        for item in os.listdir(folder):
            if item in keep_files:
                continue
                
            item_path = os.path.join(folder, item)
            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)
                    log(f"  删除文件: {item}")
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    log(f"  删除目录: {item}")
            except Exception as e:
                log(f"  删除 {item} 失败: {e}")
                
        log(f"[CLEAN] 完成目录清理: {folder}")
        
    except Exception as e:
        log(f"[ERROR] 清理目录失败 {folder}: {e}")


def run_single_sample(sample: SampleInfo, python_cmd: str, redo: bool, sleep_s: float, keep_odb: bool) -> Dict[str,str]:
    folder = sample.path
    inp_path = os.path.join(folder, 'model.inp')
    result = {
        'sample': sample.name,
        'status': 'SKIP',
        'time_sec': '',
        'max_disp_mm': '',
        'max_rf_N': '',
        'note': ''
    }
    if not os.path.isdir(folder):
        log(f"[WARN] 样本目录缺失: {folder}")
        result['status'] = 'MISSING_DIR'
        return result
    if not os.path.isfile(inp_path):
        log(f"[WARN] 缺少 model.inp: {inp_path}")
        result['status'] = 'MISSING_INP'
        return result

    output_dir = os.path.join(folder, 'output')
    curve_csv = os.path.join(output_dir, 'rf_displacement_curve.csv')
    
    # 运行前先清理目录，只保留必需文件
    cleanup_sample_files(folder, keep_odb, pre_run=True)
    
    # 检查是否需要重新运行
    if os.path.isfile(curve_csv) and not redo:
        log(f"[SKIP] {sample.name} 已存在结果 (加 --redo 可强制重算)")
        result['status'] = 'EXIST'
        # 仍尝试读取指标
        read_curve_metrics(curve_csv, result)
        return result

    # 如果是重跑，清空output目录中的旧文件
    if os.path.isdir(output_dir) and redo:
        try:
            # 清空output目录中的所有文件
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
        except Exception as e:
            log(f"[WARN] 清空output目录失败: {e}")

    # 直接通过 Python 调用单次脚本
    cmd = [python_cmd, SINGLE_SCRIPT, '--', f'inp={inp_path}']
    log(f"[RUN] 样本 {sample.name} 开始: {' '.join(cmd)}")
    t0 = time.time()
    
    # 启动进度显示
    stop_progress = threading.Event()
    if TQDM_AVAILABLE:
        pbar = tqdm(total=100, desc=f"样本 {sample.name}", unit="%", ncols=100)
        progress_thread = threading.Thread(target=update_progress_tqdm, args=(folder, pbar, stop_progress, 1.0))
    else:
        print(f"[样本 {sample.name}] 开始分析...")
        progress_thread = threading.Thread(target=update_progress_simple, args=(folder, stop_progress, 1.0))
    
    progress_thread.daemon = True
    progress_thread.start()
    
    try:
        proc = subprocess.Popen(cmd, cwd=folder, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        # 持续打印实时输出 (可选, 若输出太多可注释)
        for line in proc.stdout:
            line_stripped = line.rstrip('\n')
            # 只打印关键信息行
            if any(k in line_stripped for k in ('[ERROR]','[OK]','[WARN]','RF-位移','提交作业','提取')):
                # 清除进度显示再打印信息
                if not TQDM_AVAILABLE:
                    sys.stdout.write('\r' + ' ' * 80 + '\r')
                    sys.stdout.flush()
                log(f"    {line_stripped}")
        ret = proc.wait()
        dt = time.time() - t0
        result['time_sec'] = f"{dt:.1f}"
        if ret != 0:
            log(f"[FAIL] 样本 {sample.name} Python 返回码 {ret}")
            result['status'] = 'RUN_ERR'
            return result
        # 检查曲线
        if os.path.isfile(curve_csv):
            result['status'] = 'OK'
            read_curve_metrics(curve_csv, result)
            log(f"[DONE] 样本 {sample.name} 完成, max_disp={result['max_disp_mm']} mm, max_RF={result['max_rf_N']} N")
        else:
            log(f"[FAIL] 样本 {sample.name} 未生成曲线文件: {curve_csv}")
            result['status'] = 'NO_CURVE'
    except KeyboardInterrupt:
        log('[STOP] 用户中断')
        raise
    except Exception as e:
        log(f"[ERROR] 样本 {sample.name} 运行异常: {e}")
        result['status'] = 'EXCEPTION'
        result['note'] = str(e)
    finally:
        # 停止进度显示
        stop_progress.set()
        progress_thread.join(timeout=2)
        
        # 关闭 tqdm 进度条
        if TQDM_AVAILABLE:
            pbar.close()
        else:
            sys.stdout.write('\r' + ' ' * 80 + '\r')
            sys.stdout.flush()
            print(f"[样本 {sample.name}] 分析完成")
        
        # 处理结束后进行清理
        if result['status'] in ('OK', 'EXIST') or result['status'] == 'NO_CURVE':
            cleanup_sample_files(folder, keep_odb, pre_run=False)
        
        if sleep_s > 0:
            time.sleep(sleep_s)
    return result


def read_curve_metrics(csv_path: str, result: Dict[str,str]):
    try:
        max_disp = 0.0
        max_rf = 0.0
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) < 3:
                    continue
                try:
                    disp = float(row[1])
                    rf = float(row[2])
                    if disp > max_disp: max_disp = disp
                    if rf > max_rf: max_rf = rf
                except ValueError:
                    continue
        result['max_disp_mm'] = f"{max_disp:.6f}" if max_disp>0 else ''
        result['max_rf_N'] = f"{max_rf:.2f}" if max_rf>0 else ''
    except Exception as e:
        result['note'] = (result.get('note','') + f"; METRIC_ERR:{e}").strip('; ')


def write_summary(rows: List[Dict[str,str]]):
    fieldnames = ['sample','status','time_sec','max_disp_mm','max_rf_N','note']
    try:
        with open(SUMMARY_CSV, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        log(f"[SUMMARY] 汇总已写入: {SUMMARY_CSV}")
    except Exception as e:
        log(f"[ERROR] 写入汇总失败: {e}")

# -------------------------------------------------
# 主逻辑
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='批量运行TPMS有限元RF-位移分析 (Python 直接调用模式)')
    parser.add_argument('--root', default=DEFAULT_DATA_ROOT, help='数据根目录 (含0001等子目录)')
    parser.add_argument('--start', type=int, help='起始编号 (含)')
    parser.add_argument('--end', type=int, help='结束编号 (含)')
    parser.add_argument('--samples', help='手动指定样本: 形式 1,5,10-12 (优先于 start/end)')
    parser.add_argument('--redo', action='store_true', help='即使已有结果也重新计算')
    parser.add_argument('--limit', type=int, help='最多处理多少个样本 (调试用)')
    parser.add_argument('--sleep', type=float, default=0.0, help='每个样本之间暂停秒数')
    parser.add_argument('--python', default=sys.executable, help='Python 解释器路径')
    parser.add_argument('--keep-odb', action='store_true', help='保留 Job-1.odb 文件 (用于后处理)')
    args = parser.parse_args()

    # 清空旧日志
    try:
        if os.path.isfile(LOG_FILE): os.remove(LOG_FILE)
    except Exception:
        pass
    
    log('===== 批量TPMS RF-位移分析开始 (Python模式) =====')
    log(f"单次脚本: {SINGLE_SCRIPT}")
    log(f"Python解释器: {args.python}")
    log(f"保留ODB文件: {args.keep_odb}")
    
    if TQDM_AVAILABLE:
        log("使用 tqdm 显示进度条")
    else:
        log("tqdm 未安装，使用简单进度显示")

    if not os.path.isfile(SINGLE_SCRIPT):
        log('[FATAL] 找不到单次分析脚本 abaqus_auto.py')
        sys.exit(1)

    all_samples = collect_samples(args.root, None if args.samples else args.start, None if args.samples else args.end)

    if args.samples:
        sample_list = select_samples_by_spec(all_samples, args.samples)
    else:
        sample_list = all_samples

    if args.limit and len(sample_list) > args.limit:
        sample_list = sample_list[:args.limit]

    if not sample_list:
        log('[ERROR] 没有找到需要处理的样本')
        sys.exit(1)

    preview = [s.name for s in sample_list[:10]]
    log(f"计划处理样本数量: {len(sample_list)} -> {preview}{' ...' if len(sample_list)>10 else ''}")

    rows = []
    ok_cnt = fail_cnt = 0
    for sample in sample_list:
        row = run_single_sample(sample, args.python, args.redo, args.sleep, args.keep_odb)
        rows.append(row)
        if row['status'] == 'OK':
            ok_cnt += 1
        elif row['status'] not in ('EXIST','SKIP'):
            fail_cnt += 1
    write_summary(rows)
    log(f"完成: 成功 {ok_cnt} 个, 失败 {fail_cnt} 个, 总计 {len(rows)}")
    log('===== 批量TPMS RF-位移分析结束 =====')


if __name__ == '__main__':
    main()