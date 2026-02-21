#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量应力-应变结果后处理脚本
================================
功能:
    - 批量遍历 dataset_fea 下所有包含结果的样本目录 (支持任意层级/命名)
  - 查找 output/rf_displacement_curve.csv
  - 复用 stress_strain_analysis.StressStrainAnalyzer 计算:
       弹性模量, 屈服强度(0.2%偏移/回退法), 极限强度, 能量吸收密度
  - 生成/更新每个样本的: stress_strain_curve.csv, mechanical_properties.csv, stress_strain_curve.png/svg, analysis_report.txt
  - 汇总所有样本指标到总表 batch_stress_strain_summary.csv

使用示例:
  python batch_postprocess_stress_strain.py --root dataset_fea --start 1 --end 100
    python batch_postprocess_stress_strain.py --samples 1,5,10-12 --area 400 --length 20
  python batch_postprocess_stress_strain.py --root dataset_fea --redo

参数说明:
    --root         数据集根目录 (可含任意命名的样本子目录)
  --start/end    起止编号 (含)
  --samples      逗号/连字符混合指定样本 如: 1,5,10-12
  --area         横截面积 mm^2
  --length       标距长度 mm
  --summary      汇总CSV输出路径 (默认脚本所在目录)
  --redo         即使目标 mechanical_properties.csv 已存在也重新计算
  --skip-fail    遇到异常仅记录并继续 (默认)
  --stop-on-fail 遇到异常立即退出
  --verbose      DEBUG日志
docs\paper.docx
依赖: pandas numpy matplotlib (与 stress_strain_analysis 相同)
"""
from __future__ import annotations
import os
import sys
import csv
import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import traceback
import pandas as pd
import json
import numpy as np

from tools.tpms_surface_metrics import TPMSSurfaceMetrics, calculate_tpm_ssa

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.join(SCRIPT_DIR, 'dataset_fea')
DEFAULT_SUMMARY = os.path.join(SCRIPT_DIR, 'dataset_fea/batch_stress_strain_summary.csv')
DEFAULT_E_CSV = os.path.join(SCRIPT_DIR, 'dataset_fea/E.csv')
DEFAULT_YIELD_CSV = os.path.join(SCRIPT_DIR, 'dataset_fea/yield.csv')

# 导入分析器
try:
    from tools.stress_strain_analysis import StressStrainAnalyzer, validate_data
except ImportError as e:
    print('[FATAL] 无法导入 stress_strain_analysis. 请确保文件在同一目录并无语法错误.')
    raise

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('BatchStressStrain')

# 禁用 matplotlib 弹窗显示（批处理场景不应弹窗）
try:
    import matplotlib.pyplot as plt  # 与 stress_strain_analysis 使用同一 pyplot 模块
    plt.show = lambda *args, **kwargs: None  # 覆盖为无操作
except Exception:
    pass


@dataclass(frozen=True)
class SampleInfo:
    """描述一个应力-应变后处理样本."""
    name: str
    path: str
    numeric_id: Optional[int]

# -------------------------------------------------
# 工具函数
# -------------------------------------------------

def parse_int_list(spec: str) -> List[int]:
    out: List[int] = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            try:
                ai = int(a); bi = int(b)
            except ValueError:
                continue
            if ai <= bi:
                out.extend(range(ai, bi + 1))
            else:
                out.extend(range(ai, bi - 1, -1))
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

    def add(sample: SampleInfo):
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

    for num in sorted(numeric_targets):
        matches = by_numeric.get(num)
        if matches:
            for m in matches:
                add(m)
        else:
            logger.warning(f'未找到编号为 {num:04d} 的样本')

    range_pattern = re.compile(r'^\d+\s*-\s*\d+$')
    for token in tokens:
        token_compact = token.replace(' ', '')
        if token_compact.isdigit() or range_pattern.match(token_compact):
            continue
        norm = token.replace('\\', '/').strip('/')
        direct = by_name.get(norm)
        if direct:
            add(direct)
            continue
        leaf_matches = by_leaf.get(token) or by_leaf.get(norm.split('/')[-1])
        if leaf_matches:
            for m in leaf_matches:
                add(m)
            continue
        logger.warning(f'指定的样本未找到: {token}')

    return selected


def collect_samples(root: str, start: Optional[int], end: Optional[int]) -> List[SampleInfo]:
    samples: List[SampleInfo] = []
    if not os.path.isdir(root):
        logger.error(f'数据根目录不存在: {root}')
        return samples

    abs_root = os.path.abspath(root)
    for current_root, dirs, files in os.walk(abs_root):
        leaf = os.path.basename(current_root)
        if leaf.lower() == 'output':
            continue

        has_output_dir = any(d.lower() == 'output' for d in dirs)
        dirs[:] = [d for d in dirs if d.lower() != 'output']
        has_curve = os.path.isfile(os.path.join(current_root, 'output', 'rf_displacement_curve.csv'))
        has_inp = 'model.inp' in files
        has_meta = 'meta.json' in files
        if not (has_curve or has_output_dir or has_inp or has_meta):
            continue

        rel_path = os.path.relpath(current_root, abs_root)
        if rel_path == '.':
            rel_path = os.path.basename(abs_root)
        display_name = rel_path.replace(os.sep, '/').strip('/') or leaf
        numeric_id = int(leaf) if leaf.isdigit() else None

        if (start is not None or end is not None) and numeric_id is None:
            continue
        if start is not None and numeric_id is not None and numeric_id < start:
            continue
        if end is not None and numeric_id is not None and numeric_id > end:
            continue

        samples.append(SampleInfo(name=display_name, path=current_root, numeric_id=numeric_id))

    samples.sort(key=lambda s: (s.numeric_id if s.numeric_id is not None else float('inf'), s.name))
    return samples

# -------------------------------------------------
# 新增: 计算顶面横截面积 (基于 voxel.npy + meta.json)
# -------------------------------------------------

def _load_meta(sample_dir: str) -> dict:
    meta_path = os.path.join(sample_dir, 'meta.json')
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def compute_top_area_from_voxel(sample_dir: str,
                                layers: int = 1,
                                axis: str = 'auto',
                                gauge_length: Optional[float] = None) -> Tuple[Optional[float], str]:
    """
    返回 (area_mm2, 状态字符串)
    状态: OK / NO_VOXEL / NO_SIZE / ERR / EMPTY
    逻辑:
      1. 读取 voxel.npy (0/1 或浮点) -> 形状 shape=(a,b,c)
      2. 确定竖直轴索引 z_idx:
         - 若 axis 指定为 x,y,z 则固定
         - auto: 默认取第0维为高度 (常见导出: (Nz, Ny, Nx))
      3. 取顶面 layers 层 (从最高层向下) 做占据并集 sum_occ
      4. 体素边长:
         - meta.json: length_mm 或 bbox_mm(list[3]) -> 用与 Nx 对应的长度 / Nx
         - 若没有 length 信息, 但提供 gauge_length 且 z 方向体素层数==gauge_length/voxel_size 假设 voxel_size = gauge_length / Nz
         - 简化: 优先 length_mm (整体立方边长) ; 若 bbox_mm 存在用 bbox_mm[0]; 最后用 gauge_length 估算
      5. 面积 = 占据数 * voxel_size^2 (若为密度浮点则用 clip 到 [0,1] 的和)
    """
    voxel_path = os.path.join(sample_dir, 'voxel.npy')
    if not os.path.isfile(voxel_path):
        return None, 'NO_VOXEL'
    try:
        voxel = np.load(voxel_path, allow_pickle=False)
        if voxel.ndim != 3:
            return None, 'BAD_DIM'
    except Exception as e:
        return None, 'ERR'

    shape = voxel.shape  # (d0,d1,d2)
    # 竖直轴选择
    axis_map = {'x':2, 'y':1, 'z':0}
    if axis.lower() in axis_map:
        z_idx = axis_map[axis.lower()]
    else:
        z_idx = 0  # auto 默认

    # 统一转置让顺序 (Nz, Ny, Nx)
    if z_idx != 0:
        # 将 z_idx 轴换到 0 位置
        order = [z_idx] + [i for i in range(3) if i != z_idx]
        voxel = np.transpose(voxel, axes=order)
    Nz, Ny, Nx = voxel.shape

    # 限制 layers
    layers = max(1, min(layers, Nz))
    top_block = voxel[Nz-layers:Nz]
    if np.issubdtype(voxel.dtype, np.floating):
        occ = np.clip(top_block, 0, 1).sum(axis=0)  # 按层叠加
        # 只要任一层有实体就计 1 (并集)
        occ_mask = (occ > 0).astype(float)
        occ_count = occ_mask.sum()
    else:
        occ_count = (top_block > 0).any(axis=0).sum()
    if occ_count == 0:
        return None, 'EMPTY'

    meta = _load_meta(sample_dir)
    voxel_size = None
    # 尝试 meta 中的长度
    try:
        if 'length_mm' in meta and meta['length_mm']:
            length_mm = float(meta['length_mm'])
            voxel_size = length_mm / Nx
        elif 'bbox_mm' in meta and isinstance(meta['bbox_mm'], (list,tuple)) and len(meta['bbox_mm'])==3:
            length_mm = float(meta['bbox_mm'][0])  # 假设等距立方
            voxel_size = length_mm / Nx
    except Exception:
        voxel_size = None

    if voxel_size is None and gauge_length and gauge_length > 0:
        # 用 gauge_length 反推: 假设 Nz * voxel_size ≈ gauge_length
        voxel_size = gauge_length / Nz

    if voxel_size is None or voxel_size <= 0:
        return None, 'NO_SIZE'

    area_mm2 = occ_count * (voxel_size ** 2)
    return float(area_mm2), 'OK'


def resolve_stl_path(sample_dir: str) -> Optional[Path]:
    """尽量定位样本的 STL 文件路径。"""
    root = Path(sample_dir)
    primary = root / 'model.stl'
    if primary.is_file():
        return primary

    # 回退：寻找第一个不在 output 目录下的 STL 文件
    for candidate in sorted(root.rglob('*.stl')):
        if any(part.lower() == 'output' for part in candidate.parts):
            continue
        return candidate
    return None


def save_surface_metrics(output_dir: str,
                         metrics: TPMSSurfaceMetrics,
                         bbox: Optional[np.ndarray],
                         *,
                         material: str,
                         stl_path: Path) -> None:
    """将表面积指标保存到 CSV 与文本报告。"""
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, 'tpms_surface_metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value', 'unit'])
        writer.writerow(['surface_area', f"{metrics.surface_area_mm2:.6f}", 'mm^2'])
        writer.writerow(['solid_volume', f"{metrics.solid_volume_mm3:.6f}", 'mm^3'])
        writer.writerow(['mass', f"{metrics.mass_g:.6f}", 'g'])
        writer.writerow(['porosity', f"{metrics.porosity:.6f}", ''])
        writer.writerow(['ssa_mass', f"{metrics.specific_surface_area_mm2_per_g:.6f}", 'mm^2/g'])
        writer.writerow(['ssa_volume', f"{metrics.specific_surface_area_mm2_per_mm3:.9f}", 'mm^-1'])
        if bbox is not None:
            writer.writerow(['bbox_x', f"{bbox[0]:.6f}", 'mm'])
            writer.writerow(['bbox_y', f"{bbox[1]:.6f}", 'mm'])
            writer.writerow(['bbox_z', f"{bbox[2]:.6f}", 'mm'])

    report_path = os.path.join(output_dir, 'tpms_surface_metrics.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(metrics.format_report(
            stl_path=stl_path,
            material=material,
            bounding_box_mm=bbox,
        ))

# -------------------------------------------------
# 核心处理 (修改: 支持 auto_area)
# -------------------------------------------------

def process_single(sample: SampleInfo,
                   area: float,
                   length: float,
                   redo: bool,
                   auto_area: bool = False,
                   voxel_layers: int = 1,
                   voxel_axis: str = 'auto',
                   fallback_area: Optional[float] = None,
                   *,
                   ssa_enabled: bool = True,
                   ssa_density: float = 4.43,
                   ssa_expected_size: Optional[float] = 20.0,
                   ssa_material: str = 'Ti6Al4V') -> Dict[str, str]:
    """处理单个样本并返回结果字典 (扩展: 自动顶面面积)"""
    sample_dir = sample.path
    out_dir = os.path.join(sample_dir, 'output')
    curve_csv = os.path.join(out_dir, 'rf_displacement_curve.csv')
    mech_csv = os.path.join(out_dir, 'mechanical_properties.csv')

    result: Dict[str, str] = {
        'sample': sample.name,
        'status': 'SKIP',
        'elastic_modulus': '',
        'yield_strength': '',
        'ultimate_strength': '',
        'energy_absorption': '',
        'max_strain': '',
        'yield_method': '',
        'used_area': '',
        'note': '',
        'ssa_surface_area_mm2': '',
        'ssa_volume_mm3': '',
        'ssa_mass_g': '',
        'ssa_porosity': '',
        'ssa_mm2_per_g': '',
        'ssa_mm2_per_mm3': '',
        'ssa_bbox_x_mm': '',
        'ssa_bbox_y_mm': '',
        'ssa_bbox_z_mm': '',
        'ssa_status': 'DISABLED' if not ssa_enabled else 'SKIP',
        'ssa_note': ''
    }

    if not os.path.isdir(sample_dir):
        result['status'] = 'NO_DIR'
        result['note'] = '目录缺失'
        logger.warning(f'样本 {sample.name} 目录缺失: {sample_dir}')
        return result

    stl_path: Optional[Path] = None
    if ssa_enabled:
        stl_path = resolve_stl_path(sample_dir)
        if stl_path is None:
            result['ssa_status'] = 'NO_STL'
            result['ssa_note'] = '未找到 STL 文件'
        else:
            try:
                metrics, bbox = calculate_tpm_ssa(
                    stl_path,
                    density_g_per_cm3=ssa_density,
                    expected_size_mm=ssa_expected_size,
                )
                result.update({
                    'ssa_surface_area_mm2': f"{metrics.surface_area_mm2:.6f}",
                    'ssa_volume_mm3': f"{metrics.solid_volume_mm3:.6f}",
                    'ssa_mass_g': f"{metrics.mass_g:.6f}",
                    'ssa_porosity': f"{metrics.porosity:.6f}",
                    'ssa_mm2_per_g': f"{metrics.specific_surface_area_mm2_per_g:.6f}",
                    'ssa_mm2_per_mm3': f"{metrics.specific_surface_area_mm2_per_mm3:.9f}",
                    'ssa_status': 'OK',
                })
                if bbox is not None:
                    result['ssa_bbox_x_mm'] = f"{bbox[0]:.6f}"
                    result['ssa_bbox_y_mm'] = f"{bbox[1]:.6f}"
                    result['ssa_bbox_z_mm'] = f"{bbox[2]:.6f}"
                save_surface_metrics(out_dir, metrics, bbox, material=ssa_material, stl_path=stl_path)
            except Exception as exc:
                result['ssa_status'] = 'ERROR'
                result['ssa_note'] = str(exc)

    if not os.path.isfile(curve_csv):
        result['status'] = 'NO_CURVE'
        result['note'] = '缺少rf_displacement_curve.csv'
        logger.warning(f'样本 {sample.name} 缺少曲线文件: {curve_csv}')
        return result

    # 决定使用的面积
    computed_area = None
    area_note = ''
    if auto_area:
        computed_area, flag = compute_top_area_from_voxel(sample_dir, layers=voxel_layers, axis=voxel_axis, gauge_length=length)
        if computed_area is not None and flag == 'OK':
            use_area = computed_area
            area_note = 'AUTO'
        else:
            # 回退: fallback_area > 指定 area
            if fallback_area and fallback_area > 0:
                use_area = fallback_area
                area_note = f'FALLBACK({flag})'
            else:
                use_area = area
                area_note = f'CLI({flag})'
    else:
        use_area = area
        area_note = 'CLI'

    if os.path.isfile(mech_csv) and not redo and not auto_area:
        # 读取已有 mechanical_properties.csv (可选) 仅在不需要重新计算且不做 auto_area
        try:
            df_mech = pd.read_csv(mech_csv)
            def pick(name: str):
                row = df_mech[df_mech['Property'] == name]
                return row['Value'].iloc[0] if not row.empty else ''
            result.update({
                'status': 'EXIST',
                'elastic_modulus': pick('弹性模量'),
                'yield_strength': pick('屈服强度'),
                'ultimate_strength': pick('极限强度'),
                'energy_absorption': pick('能量吸收密度'),
                'used_area': f"{use_area:.4f}" if use_area else ''
            })
        except Exception as e:
            result['note'] = f'读取已有结果失败:{e}'
        return result

    try:
        df = pd.read_csv(curve_csv)
        rename_map = {}
        for col in df.columns:
            lc = col.lower().strip()
            if lc in ('displacement(mm)', 'displacement', 'disp'):
                rename_map[col] = 'Displacement(mm)'
            elif lc in ('reaction_force(n)', 'reaction_force', 'force', 'rf'):
                rename_map[col] = 'Reaction_Force(N)'
        if rename_map:
            df = df.rename(columns=rename_map)
        validated = validate_data(df)
        if validated is False:
            result['status'] = 'INVALID_DATA'
            return result
        df = validated
        displacement = df['Displacement(mm)'].values
        force = df['Reaction_Force(N)'].values

        analyzer = StressStrainAnalyzer(displacement, force, use_area, length)
        analyzer.analyze_all()
        analyzer.plot_stress_strain(output_dir=out_dir, save=True)
        analyzer.save_results(output_dir=out_dir)

        res = analyzer.results
        result['status'] = 'OK'
        result['elastic_modulus'] = f"{res['elastic_modulus']:.2f}" if res['elastic_modulus'] else ''
        result['yield_strength'] = f"{res['yield_strength']:.2f}" if res['yield_strength'] else ''
        result['ultimate_strength'] = f"{res['ultimate_strength']:.2f}" if res['ultimate_strength'] else ''
        result['energy_absorption'] = f"{res['energy_absorption']:.4f}" if res.get('energy_absorption') else ''
        result['yield_method'] = res['method'].get('yield', '')
        result['max_strain'] = f"{max(analyzer.strain):.5f}"
        result['used_area'] = f"{use_area:.4f}"
        if area_note != 'AUTO':
            result['note'] = area_note
    except Exception as e:
        result['status'] = 'ERROR'
        result['note'] = str(e)
        logger.error(f"样本 {sample.name} 处理失败: {e}")
        logger.debug(traceback.format_exc())
    return result

# -------------------------------------------------
# 汇总 (新增 used_area 字段)
# -------------------------------------------------

def write_summary(rows: List[Dict[str, str]], path: str):
    fieldnames = [
        'sample','status','elastic_modulus','yield_strength','ultimate_strength','energy_absorption','max_strain',
        'yield_method','used_area','note','ssa_status','ssa_surface_area_mm2','ssa_volume_mm3','ssa_mass_g',
        'ssa_porosity','ssa_mm2_per_g','ssa_mm2_per_mm3','ssa_bbox_x_mm','ssa_bbox_y_mm','ssa_bbox_z_mm','ssa_note'
    ]
    try:
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        logger.info(f'汇总已写入: {path} (共 {len(rows)} 条)')
    except Exception as e:
        logger.error(f'写入汇总失败: {e}')

# 新增: 输出单项指标 CSV（如 E.csv / yield.csv）

def write_metric_csv(rows: List[Dict[str, str]], path: str, field: str, header: str):
    try:
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['sample', header])
            for r in rows:
                val = r.get(field, '')
                if val not in (None, ''):
                    w.writerow([r.get('sample',''), val])
        logger.info(f'指标({header}) 已写入: {path}')
    except Exception as e:
        logger.error(f'写入 {path} 失败: {e}')

# -------------------------------------------------
# 主函数 (新增参数)
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='TPMS 批量应力-应变后处理工具',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--root', default=DEFAULT_ROOT, help='数据根目录 (含 0001 等子目录)')
    parser.add_argument('--start', type=int, default=None, help='起始编号 (含)')
    parser.add_argument('--end', type=int, default=None, help='结束编号 (含)')
    parser.add_argument('--samples', help='指定样本列表, 如 1,5,10-12 (优先于 start/end)')
    parser.add_argument('--area', type=float, default=576, help='横截面积 mm^2 (非自动模式，若未启用自动面积则使用该值)')
    parser.add_argument('--length', type=float, default=24.0, help='标距长度 mm (用于应变与自动体素尺寸推断)')
    parser.add_argument('--auto-area', action='store_true', default=False, help='启用: 基于 voxel 顶面自动计算横截面积 (默认关闭)')
    parser.add_argument('--voxel-layers', type=int, default=1, help='自动面积: 取顶面多少层合并 (>=1)')
    parser.add_argument('--voxel-axis', default='auto', choices=['auto','x','y','z'], help='自动面积: 垂直(加载)方向轴 (auto=假定第0维)')
    parser.add_argument('--fallback-area', type=float, help='自动面积失败时回退常数面积 (缺省则用 --area)')
    parser.add_argument('--summary', default=None, help='汇总CSV输出路径 (默认写至 <root>/batch_stress_strain_summary.csv)')
    parser.add_argument('--ssa-density', type=float, default=4.43, help='TPMS 表面积分析: 材料密度 (g/cm^3)')
    parser.add_argument('--ssa-expected-size', type=float, default=24.0, help='TPMS 表面积分析: 参考立方体边长 (mm)，配合 --ssa-auto-size 控制')
    parser.add_argument('--ssa-auto-size', action='store_true', help='TPMS 表面积分析: 忽略参考尺寸，使用 STL 包络盒估算孔隙率')
    parser.add_argument('--ssa-material', default='Ti6Al4V', help='TPMS 表面积分析: 报告中显示的材料名称 (仅文本用途)')
    parser.add_argument('--disable-ssa', action='store_true', help='关闭 TPMS 表面积计算 (默认开启)')
    parser.add_argument('--redo', action='store_true', help='即使存在 mechanical_properties.csv 也重算')
    parser.add_argument('--stop-on-fail', action='store_true', help='遇到失败立即停止 (默认继续)')
    parser.add_argument('--verbose', action='store_true', help='DEBUG 日志')

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # 若未指定 --summary，则默认写到 --root/batch_stress_strain_summary.csv
    if not args.summary:
        args.summary = os.path.join(args.root, 'batch_stress_strain_summary.csv')

    sample_pool = collect_samples(args.root, None if args.samples else args.start, None if args.samples else args.end)

    if args.samples:
        sample_list = select_samples_by_spec(sample_pool, args.samples)
    else:
        sample_list = sample_pool

    if not sample_list:
        logger.error('没有发现需要处理的样本')
        sys.exit(1)

    mode_desc = 'AUTO' if args.auto_area else 'CONST'
    preview = [s.name for s in sample_list[:12]]
    logger.info(f'计划处理样本数: {len(sample_list)} -> {preview}{" ..." if len(sample_list)>12 else ""}')
    logger.info(f'面积模式={mode_desc}, area={args.area} mm^2, length={args.length} mm, layers={args.voxel_layers}, axis={args.voxel_axis}, redo={args.redo}')
    if args.disable_ssa:
        logger.info('TPMS 表面积分析: 已禁用 (--disable-ssa)')
    else:
        bbox_desc = 'auto-shape' if args.ssa_auto_size else f'{args.ssa_expected_size}mm cube'
        logger.info(f'TPMS 表面积分析: density={args.ssa_density} g/cm^3, bbox={bbox_desc}, material={args.ssa_material}')

    rows: List[Dict[str, str]] = []
    ok_cnt = fail_cnt = 0
    for s in sample_list:
        row = process_single(
            s,
            args.area,
            args.length,
            args.redo,
            auto_area=args.auto_area,
            voxel_layers=args.voxel_layers,
            voxel_axis=args.voxel_axis,
            fallback_area=args.fallback_area,
            ssa_enabled=not args.disable_ssa,
            ssa_density=args.ssa_density,
            ssa_expected_size=None if args.ssa_auto_size else args.ssa_expected_size,
            ssa_material=args.ssa_material
        )
        rows.append(row)
        if row['status'] in ('OK', 'EXIST'):
            ok_cnt += 1
        elif row['status'] in ('NO_CURVE','NO_DIR'):
            pass
        else:
            fail_cnt += 1
            if args.stop_on_fail:
                logger.error('配置为失败即停止, 退出...')
                break
    write_summary(rows, args.summary)
    # 生成 E.csv 与 yield.csv -> 输出到当前 root 目录
    e_csv_path = os.path.join(args.root, 'E.csv')
    yield_csv_path = os.path.join(args.root, 'yield.csv')
    write_metric_csv(rows, e_csv_path, 'elastic_modulus', 'E')
    write_metric_csv(rows, yield_csv_path, 'yield_strength', 'yield')
    surface_csv_path = os.path.join(args.root, 'tpms_surface_area_mm2.csv')
    mass_ssa_csv_path = os.path.join(args.root, 'tpms_specific_surface_area_mm2_per_g.csv')
    volume_ssa_csv_path = os.path.join(args.root, 'tpms_specific_surface_area_mm2_per_mm3.csv')
    porosity_csv_path = os.path.join(args.root, 'tpms_porosity.csv')
    write_metric_csv(rows, surface_csv_path, 'ssa_surface_area_mm2', 'surface_area_mm2')
    write_metric_csv(rows, mass_ssa_csv_path, 'ssa_mm2_per_g', 'ssa_mm2_per_g')
    write_metric_csv(rows, volume_ssa_csv_path, 'ssa_mm2_per_mm3', 'ssa_mm2_per_mm3')
    write_metric_csv(rows, porosity_csv_path, 'ssa_porosity', 'porosity')
    logger.info('额外指标已导出: %s, %s, %s, %s, %s, %s',
                e_csv_path, yield_csv_path, surface_csv_path, mass_ssa_csv_path, volume_ssa_csv_path, porosity_csv_path)
    logger.info(f'完成: 成功/已有 {ok_cnt}, 失败 {fail_cnt}, 总计 {len(rows)}')

if __name__ == '__main__':
    main()
