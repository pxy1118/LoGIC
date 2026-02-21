#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
应力-应变分析后处理脚本 - 严格匹配参考实现
功能：从Abaqus RF-位移数据计算应力-应变关系和力学性能参数
关键修复：完全匹配参考代码的0.2%偏移法实现
使用方法：
  1. 确保已安装依赖: pip install pandas numpy matplotlib scipy
    2. 运行: python stress_strain_analysis.py --input output/rf_displacement_curve.csv
    3. 或指定参数: python stress_strain_analysis.py --input data.csv --area 400 --length 20

作者: TPMS分析工具链
版本: 1.2 (严格匹配参考实现)
日期: 2025-09-02
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("StressStrainAnalysis")

class StressStrainAnalyzer:
    """应力-应变分析器核心类 - 严格匹配参考实现"""
    
    def __init__(self, displacement, force, cross_sectional_area, gauge_length):
        """
        初始化分析器
        
        参数:
        displacement: 位移数据 (mm)
        force: 反力数据 (N)
        cross_sectional_area: 横截面积 (mm²)
        gauge_length: 标距长度 (mm)
        """
        self.displacement = np.array(displacement)
        self.force = np.array(force)
        self.cross_sectional_area = cross_sectional_area
        self.gauge_length = gauge_length
        
        # 计算基本参数
        self.stress = self.force / self.cross_sectional_area  # MPa
        self.strain = self.displacement / self.gauge_length  # 无量纲
        
        # 结果存储
        self.results = {
            'elastic_modulus': None,
            'yield_strength': None,
            'ultimate_strength': None,
            'yield_point': (None, None),
            'ultimate_point': (None, None),
            'energy_absorption': None,
            'method': {
                'elastic': '线性段斜率法',
                'yield': '0.2%偏移法'
            }
        }
        self.linear_segment = None
    
    def calculate_elastic_modulus(self, strain_threshold=0.015, min_points=2):
        """
        计算弹性模量 (使用所有应变不超过阈值的数据进行线性拟合)

        参数:
        strain_threshold: 弹性拟合所用的最大应变阈值 (默认 0.01)
        min_points: 拟合所需最少数据点 (默认 2)

        返回:
        弹性模量 (MPa)
        """
        threshold = float(strain_threshold)
        mask = self.strain <= threshold
        strain_linear = self.strain[mask]
        stress_linear = self.stress[mask]
        used_count = strain_linear.size

        if used_count < min_points:
            logger.error(
                "应变<=%.4f 的数据点不足以拟合弹性模量 (点数=%d < %d)",
                threshold,
                used_count,
                min_points,
            )
            return None

        if np.isclose(np.ptp(strain_linear), 0.0):
            logger.error(
                "应变<=%.4f 的数据点缺乏足够的应变变化，无法拟合弹性模量",
                threshold,
            )
            return None

        # 线性拟合
        coeffs = np.polyfit(strain_linear, stress_linear, 1)
        elastic_modulus = coeffs[0]  # 斜率即为弹性模量
        
        self.results['elastic_modulus'] = elastic_modulus
        
        # 存储用于绘图的线性段
        strain_fit = np.linspace(strain_linear[0], strain_linear[-1], 100)
        stress_fit = np.polyval(coeffs, strain_fit)
        self.linear_segment = (strain_fit, stress_fit)
        
        logger.info(
            "弹性模量计算完成: %.2f MPa (使用 %d 个数据点, strain<=%.4f)",
            elastic_modulus,
            used_count,
            threshold,
        )
        return elastic_modulus
    
    def calculate_yield_strength_02(self, elastic_modulus, offset=0.002):
        """
        修正版：使用0.2%偏移法计算屈服强度 (严格匹配参考实现)
        
        参数:
        elastic_modulus: 弹性模量 (MPa)
        offset: 偏移量 (默认0.002 = 0.2%)
        
        返回:
        屈服强度 (MPa)
        """
        if elastic_modulus is None or elastic_modulus <= 0:
            logger.error("弹性模量无效，无法使用0.2%偏移法")
            return None
        
        # 0.2%应变偏移线
        offset_strain = offset
        offset_stress = elastic_modulus * (self.strain - offset_strain)

        # 计算差值曲线
        diff = self.stress - offset_stress

        intersections = []

        # 在线性插值层面寻找交点（优先使用）
        for i in range(1, len(diff)):
            d0, d1 = diff[i-1], diff[i]
            s0, s1 = self.strain[i-1], self.strain[i]
            sig0, sig1 = self.stress[i-1], self.stress[i]

            if np.isnan(d0) or np.isnan(d1):
                continue

            # 若恰好某点在偏移线上
            if d0 == 0 and s0 >= offset_strain:
                intersections.append((float(s0), float(sig0)))
                # 不立即跳出，防止后续还有更大的交点
                continue

            if d0 * d1 > 0:
                # 尚未跨越偏移线
                continue

            denom = (d1 - d0)
            if abs(denom) < 1e-12:
                continue

            alpha = -d0 / denom  # 0~1 之间
            if not (0.0 <= alpha <= 1.0):
                continue

            s_interp = s0 + alpha * (s1 - s0)
            if s_interp < offset_strain - 1e-6:
                # 交点尚未超过偏移应变，继续寻找
                continue

            sig_interp = sig0 + alpha * (sig1 - sig0)
            intersections.append((float(s_interp), float(sig_interp)))

        yield_strain = None
        yield_stress = None

        if intersections:
            yield_strain, yield_stress = max(intersections, key=lambda x: x[0])
            logger.debug(
                "0.2%%偏移法共找到 %d 个交点，选取最大应变交点: strain=%.6f, stress=%.2f",
                len(intersections),
                yield_strain,
                yield_stress,
            )
        else:
            # 回退：使用最接近的离散点
            diff_abs = np.abs(diff)
            yield_idx = int(np.argmin(diff_abs))
            yield_strain = float(self.strain[yield_idx])
            yield_stress = float(self.stress[yield_idx])
            if yield_strain < offset_strain:
                logger.warning(
                    "插值未找到有效交点，回退离散点且屈服应变小于偏移，结果可能偏低"
                )
        
        self.results['yield_strength'] = yield_stress
        self.results['yield_point'] = (yield_strain, yield_stress)
        self.results['method']['yield'] = f"0.2%偏移法 (线性插值, 偏移={offset_strain})"
        
        logger.info(f"0.2%偏移法屈服强度: {yield_stress:.2f} MPa (应变={yield_strain:.4f})")
        return yield_stress
    
    def calculate_ultimate_strength(self):
        """计算极限强度"""
        max_idx = np.argmax(self.stress)
        ultimate_strength = self.stress[max_idx]
        ultimate_strain = self.strain[max_idx]
        
        self.results['ultimate_strength'] = ultimate_strength
        self.results['ultimate_point'] = (ultimate_strain, ultimate_strength)
        
        logger.info(f"极限强度: {ultimate_strength:.2f} MPa (应变={ultimate_strain:.4f})")
        return ultimate_strength
    
    def calculate_energy_absorption(self):
        """计算能量吸收 (力-位移曲线下的面积)"""
        # 转换为应变-应力曲线下的面积 (单位: MJ/m³)
        energy = 0.0
        for i in range(1, len(self.strain)):
            d_strain = self.strain[i] - self.strain[i-1]
            avg_stress = (self.stress[i] + self.stress[i-1]) / 2.0
            energy += avg_stress * d_strain
        
        # 转换为 MJ/m³ (1 MPa * 无量纲 = 1 MJ/m³)
        energy_density = energy / 1000.0  # MJ/m³
        
        self.results['energy_absorption'] = energy_density
        logger.info(f"能量吸收密度: {energy_density:.4f} MJ/m³")
        return energy_density
    
    def analyze_all(self):
        """执行全部分析"""
        # 检查应变范围是否足够
        max_strain = np.max(self.strain)
        if max_strain < 0.01:
            logger.warning(f"最大应变仅为{max_strain:.4f}，可能不足以观察屈服现象")
        
        # 计算弹性模量
        E = self.calculate_elastic_modulus()
        if E and E > 0:
            # 计算屈服强度
            ys = self.calculate_yield_strength_02(E)
            if ys is None:
                logger.warning("0.2%偏移法失败，改用70%最大应力法")
                ys = self.calculate_yield_strength_70()
        else:
            logger.error("弹性模量计算失败，无法进行屈服强度计算")
            ys = self.calculate_yield_strength_70()
        
        self.calculate_ultimate_strength()
        self.calculate_energy_absorption()
        return self.results
    
    def calculate_yield_strength_70(self):
        """使用70%最大应力法计算屈服强度 (备选方法)"""
        max_stress = np.max(self.stress)
        yield_stress_target = 0.7 * max_stress
        
        # 找到第一个达到70%最大应力的点
        yield_idx = np.where(self.stress >= yield_stress_target)[0]
        
        if len(yield_idx) == 0:
            logger.error("无法确定屈服点")
            return None
        
        yield_idx = yield_idx[0]
        yield_strength = self.stress[yield_idx]
        yield_strain = self.strain[yield_idx]
        
        self.results['yield_strength'] = yield_strength
        self.results['yield_point'] = (yield_strain, yield_strength)
        self.results['method']['yield'] = "70%最大应力法"
        
        logger.info(f"70%最大应力法屈服强度: {yield_strength:.2f} MPa (应变={yield_strain:.4f})")
        return yield_strength
    
    def plot_stress_strain(
        self,
        output_dir="output",
        save=True,
        show=False,
                close=True,
                *,
                auto_crop=True,
                crop_margin=0.1,
                save_full_range=False,
                auto_crop_x=False,
    ):
        """绘制应力-应变曲线

        参数:
          - save: 是否保存到文件
          - show: 是否显示窗口 (批处理建议 False)
          - close: 是否在保存/显示后关闭图像以释放内存 (批处理建议 True)
          - auto_crop: 是否自动裁剪坐标范围（默认仅作用于应力轴）
          - crop_margin: 自动裁剪时的额外放大比例 (例如 0.1 表示保留 10% 余量)
          - save_full_range: 在自动裁剪的同时是否额外输出全范围图像
          - auto_crop_x: 是否允许对应变轴也进行裁剪 (默认 False 保留完整范围)
        """

        def _create_figure(apply_crop: bool, crop_x: bool):
            fig, ax = plt.subplots(figsize=(10, 6))

            # 主曲线
            ax.plot(self.strain, self.stress, 'b-', linewidth=1.5, label='Stress-Strain Curve')
            ax.set_xlabel('Strain (mm/mm)')
            ax.set_ylabel('Stress (MPa)')
            ax.set_title('Stress-Strain Curve')
            ax.grid(True, linestyle='--', alpha=0.7)

            # 弹性模量线
            if self.linear_segment is not None and self.results['elastic_modulus'] is not None:
                strain_fit, stress_fit = self.linear_segment
                ax.plot(
                    strain_fit,
                    stress_fit,
                    'r--',
                    linewidth=2,
                    label=f"E = {self.results['elastic_modulus']:.2f} MPa",
                )

            # 0.2%偏移线及屈服点
            if self.results['elastic_modulus'] is not None and self.results['yield_strength'] is not None:
                offset = 0.002
                offset_stress = self.results['elastic_modulus'] * (self.strain - offset)
                ax.plot(self.strain, offset_stress, 'g--', alpha=0.7, label='0.2% Offset Line')
                yield_strain, yield_strength = self.results['yield_point']
                ax.plot(
                    yield_strain,
                    yield_strength,
                    'ro',
                    markersize=8,
                    label=f"Yield Strength = {yield_strength:.2f} MPa",
                )

            # 极限强度点
            if self.results['ultimate_strength'] is not None:
                ultimate_strain, ultimate_strength = self.results['ultimate_point']
                ax.plot(
                    ultimate_strain,
                    ultimate_strength,
                    'ks',
                    markersize=8,
                    label=f"Ultimate Strength = {ultimate_strength:.2f} MPa",
                )

            if apply_crop:
                # X 轴裁剪：仅在允许时执行
                if crop_x:
                    x_upper = float(np.max(self.strain)) if len(self.strain) else 0.0
                    ultimate_strain_val = None
                    if self.results.get('ultimate_point'):
                        ultimate_strain_val = self.results['ultimate_point'][0]
                    if ultimate_strain_val and np.isfinite(ultimate_strain_val) and ultimate_strain_val > 0:
                        candidate = ultimate_strain_val * (1.0 + crop_margin)
                        if candidate < x_upper:
                            x_upper = candidate
                    if x_upper > 0:
                        ax.set_xlim(0, x_upper)

                # Y 轴裁剪：依据实际应力范围
                y_upper = float(np.max(self.stress)) if len(self.stress) else 0.0
                if y_upper > 0:
                    ax.set_ylim(0, y_upper * (1.0 + crop_margin))

            ax.legend()
            fig.tight_layout()
            return fig, ax

        fig, _ = _create_figure(apply_crop=auto_crop, crop_x=auto_crop_x)

        if save:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logger.info(f"创建输出目录: {output_dir}")

            primary_png = os.path.join(output_dir, 'stress_strain_curve.png')
            fig.savefig(primary_png, dpi=300, bbox_inches='tight')
            fig.savefig(os.path.join(output_dir, 'stress_strain_curve.svg'), bbox_inches='tight')
            logger.info("应力-应变曲线图已保存: %s", primary_png)

            if save_full_range and auto_crop:
                fig_full, _ = _create_figure(apply_crop=False, crop_x=False)
                full_png = os.path.join(output_dir, 'stress_strain_curve_full.png')
                fig_full.savefig(full_png, dpi=300, bbox_inches='tight')
                fig_full.savefig(
                    os.path.join(output_dir, 'stress_strain_curve_full.svg'),
                    bbox_inches='tight',
                )
                logger.info("应力-应变全范围图已保存: %s", full_png)
                if close:
                    plt.close(fig_full)

        if show:
            plt.show()

        if close:
            plt.close(fig)
        return fig
    
    def save_results(self, output_dir="output"):
        """保存分析结果到CSV文件"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 1. 保存应力-应变数据
        stress_strain_df = pd.DataFrame({
            'Strain': self.strain,
            'Stress(MPa)': self.stress
        })
        stress_strain_path = os.path.join(output_dir, 'stress_strain_curve.csv')
        stress_strain_df.to_csv(stress_strain_path, index=False)
        logger.info(f"应力-应变数据已保存: {stress_strain_path}")
        
        # 2. 保存力学性能结果
        results_data = [
            ['Property', 'Value', 'Unit', 'Method/Notes'],
            ['弹性模量', f"{self.results['elastic_modulus']:.2f}" if self.results['elastic_modulus'] else 'N/A', 'MPa', self.results['method'].get('elastic', '')],
            ['屈服强度', f"{self.results['yield_strength']:.2f}" if self.results['yield_strength'] else 'N/A', 'MPa', self.results['method'].get('yield', '')],
            ['极限强度', f"{self.results['ultimate_strength']:.2f}" if self.results['ultimate_strength'] else 'N/A', 'MPa', '最大应力点'],
            ['能量吸收密度', f"{self.results['energy_absorption']:.4f}" if self.results.get('energy_absorption') else 'N/A', 'MJ/m³', '应力-应变曲线下面积'],
            ['横截面积', f"{self.cross_sectional_area:.2f}", 'mm²', '输入参数'],
            ['标距长度', f"{self.gauge_length:.2f}", 'mm', '输入参数']
        ]
        
        results_df = pd.DataFrame(results_data[1:], columns=results_data[0])
        results_path = os.path.join(output_dir, 'mechanical_properties.csv')
        results_df.to_csv(results_path, index=False)
        logger.info(f"力学性能结果已保存: {results_path}")
        
        # 3. 生成综合报告
        report_path = os.path.join(output_dir, 'analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("TPMS结构应力-应变分析报告\n")
            f.write("="*60 + "\n")
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"分析脚本: {os.path.basename(__file__)} (版本 1.2)\n\n")
            
            f.write("输入参数:\n")
            f.write(f"  横截面积: {self.cross_sectional_area:.2f} mm²\n")
            f.write(f"  标距长度: {self.gauge_length:.2f} mm\n")
            f.write(f"  数据点数量: {len(self.strain)}\n")
            f.write(f"  最大应变: {np.max(self.strain):.4f}\n\n")
            
            f.write("计算结果:\n")
            f.write(f"  弹性模量: {self.results['elastic_modulus']:.2f} MPa ({self.results['method'].get('elastic', '')})\n")
            
            if self.results['yield_strength']:
                f.write(f"  屈服强度: {self.results['yield_strength']:.2f} MPa ({self.results['method'].get('yield', '')})\n")
            else:
                f.write("  屈服强度: N/A (可能应变范围不足)\n")
                
            f.write(f"  极限强度: {self.results['ultimate_strength']:.2f} MPa\n")
            f.write(f"  能量吸收密度: {self.results['energy_absorption']:.4f} MJ/m³\n\n")
        
        logger.info(f"分析报告已保存: {report_path}")
        return results_path

def validate_data(df):
    """验证输入数据的有效性"""
    required_cols = ['Displacement(mm)', 'Reaction_Force(N)']
    
    # 检查列是否存在
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"缺失必要列: {col}")
            return False
    
    # 检查数据是否为空
    if len(df) == 0:
        logger.error("输入数据为空")
        return False
    
    # 检查数据是否有效
    valid_mask = (~df['Displacement(mm)'].isna() & 
                 ~df['Reaction_Force(N)'].isna() &
                 (df['Displacement(mm)'] >= 0) &
                 (df['Reaction_Force(N)'] >= 0))
    
    valid_count = valid_mask.sum()
    if valid_count < 10:
        logger.error(f"有效数据点不足 ({valid_count} < 10)")
        return False
    
    # 清理无效数据
    df = df[valid_mask].copy()
    
    # 确保数据按位移排序（参考代码假设数据已排序）
    df = df.sort_values(by='Displacement(mm)')
    
    logger.info(f"数据验证通过，有效数据点: {len(df)}")
    
    return df

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='TPMS结构应力-应变分析后处理工具 (严格匹配参考实现)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input', type=str, required=True,
                        help='输入的RF-位移CSV文件路径')
    parser.add_argument('--area', type=float, default=576.0,
                        help='横截面积 (mm²), 默认=20×20=400')
    parser.add_argument('--length', type=float, default=24.0,
                        help='标距长度 (mm), 默认=20')
    parser.add_argument('--output', type=str, default='output',
                        help='输出目录')
    parser.add_argument('--verbose', action='store_true',
                        help='显示详细日志')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # 检查输入文件
    if not os.path.exists(args.input):
        logger.error(f"输入文件不存在: {args.input}")
        sys.exit(1)
    
    logger.info(f"开始应力-应变分析 (输入: {args.input})")
    logger.info(f"参数配置: 横截面积={args.area} mm², 标距长度={args.length} mm")
    
    try:
        # 读取数据
        df = pd.read_csv(args.input)
        logger.info(f"成功读取数据文件: {args.input} ({len(df)}行)")
        
        # 验证数据
        df = validate_data(df)
        if df is False:
            sys.exit(1)
        
        # 提取位移和反力
        displacement = df['Displacement(mm)'].values
        force = df['Reaction_Force(N)'].values
        
        # 创建分析器
        analyzer = StressStrainAnalyzer(
            displacement, force, 
            args.area, args.length
        )
        
        # 执行分析
        logger.info("开始计算力学性能参数...")
        analyzer.analyze_all()
        
        # 生成图表
        logger.info("生成应力-应变曲线图...")
        analyzer.plot_stress_strain(output_dir=args.output)
        
        # 保存结果
        analyzer.save_results(output_dir=args.output)
        
        logger.info("="*50)
        logger.info("分析完成! 结果已保存至: " + os.path.abspath(args.output))
        logger.info(f"  - stress_strain_curve.csv: 应力-应变数据")
        logger.info(f"  - mechanical_properties.csv: 力学性能参数")
        logger.info(f"  - stress_strain_curve.png/svg: 分析图表")
        logger.info(f"  - analysis_report.txt: 详细分析报告")
        logger.info("="*50)
        
    except Exception as e:
        logger.exception("分析过程中发生错误")
        sys.exit(1)

if __name__ == "__main__":
    main()