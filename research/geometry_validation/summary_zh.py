"""Generate the Chinese experiment summary from saved, immutable measurements."""
from pathlib import Path
import sys

from experiment import digest, read, save, verify


def write_summary(root):
    root = Path(root); protocol = verify(root)
    summary = read(root / "report/results.json")
    case = lambda name: read(root / "cases" / name / "result.json")
    q40 = case("orientation_gradient_z_40")["orientation"]
    q60 = case("orientation_gradient_z_60")["orientation"]
    twist = case("orientation_twist_z_30")["orientation"]
    multi = case("orientation_multiaxis")["orientation"]
    topo = case("topology_gradient")["density"]
    coupled = case("coupled_three_controls")["density"]
    content = f"""# 3.1 几何控制实验结果

本轮固定执行 38 个 240³ 主案例、3 个 120³ 分辨率复核案例，并从已完成的 FEA 发布队列
预先分层抽取 50 个已有初始几何。共 {summary['generated_completed'] + summary['reused_completed']} 项测量完成，
计算失败 {summary['failures']} 项。计算完成不等于所有质量指标都达到理想值。

## 主要发现

| 问题 | 实际结果 | 论文中可支持的结论 |
|---|---|---|
| 均匀密度能否控制准确？ | 20 个案例导出 STL 的 SVF 平均绝对误差 {summary['uniform_surface_absolute_error_pp']['mean']:.4f} 个百分点，最大 {summary['uniform_surface_absolute_error_pp']['maximum']:.4f} 个百分点。 | 在所测拓扑及均匀目标下，全局材料用量控制准确。 |
| 渐变方向能否实现？ | 0→40° 案例平均局部 z 轴误差 {q40['local_z_error_mean_deg']:.3f}°，最大 {q40['local_z_error_max_deg']:.3f}°；0→60° 强梯度最大误差 {q60['local_z_error_max_deg']:.3f}°。 | 可以量化方向实现精度及梯度边界，不能写任意方向场都无畸变。 |
| 单独密度梯度是否准确？ | 三个方向单胞尺度区域 MAE 平均 {summary['gradient_regional_mae_pp']['mean']:.3f} 个百分点。 | 所测固定拓扑混合、无旋转条件下，单胞平均密度梯度较准确。 |
| 拓扑改变是否不影响局部密度？ | 拓扑梯度的区域 MAE {topo['regional_mae_pp']:.3f} 个百分点，最大局部偏差 {topo['regional_max_absolute_error_pp']:.3f} 个百分点。 | 存在明显局部耦合，不能由全局 SVF 准确推出局部解耦。 |
| 三控制场同时变化如何？ | 区域 MAE {coupled['regional_mae_pp']:.3f} 个百分点，最大 {coupled['regional_max_absolute_error_pp']:.3f} 个百分点；导出表面的全局偏差 {coupled['mesh_error_pp']:.3f} 个百分点。 | 应将控制表述为可独立指定、几何响应存在耦合。 |

误差均以百分点表示，例如 SVF 0.30 与 0.31 的差为 1 个百分点，不是相对误差 1%。
均匀目标的采样固体比例由分位数阈值控制，接近目标本身是构造性质；导出表面体积和
匹配窗口内的局部 SVF 是另外两项检查，三者没有混用。

全部方向配置都进入方向图。扭转场的最大局部 z 轴误差为 {twist['local_z_error_max_deg']:.3f}°，
多轴场为 {multi['local_z_error_max_deg']:.3f}°；y 向渐变场也略超过 10°。这些案例没有按误差剔除。
相邻角差限制不能代替输出方向诊断；数据集本身的导出误差筛查与本轮不筛除案例的验证范围应区分。

## 局部测量尺度

主要统计采用固定 3×3×3 个单胞尺度窗口，比较同一窗口内实际占据率与三线性目标的平均值。
半单胞尺度敏感性中，三个轴向密度梯度的区域 MAE 平均为
{summary['gradient_half_cell_regional_mae_pp']['mean']:.3f} 个百分点；对应的均匀 G+D、SVF 0.35 案例在该尺度下也有
{summary['uniform_mix_half_cell_regional_mae_pp']:.3f} 个百分点的波动。因此必须区分周期微结构本身的单胞内部变化，
与单胞平均密度控制误差。窗口已在测量前固定，没有根据结果更换定义。

50 个已有异质结构的单胞尺度区域 MAE 平均为 {summary['reused_density_regional_mae_pp']['mean']:.3f} 个百分点，
最大逐结构 MAE 为 {summary['reused_density_regional_mae_pp']['maximum']:.3f} 个百分点。这是设计控制与最终初始几何的
一致性描述，不重新评判其力学标签是否有效，也不用于修改已冻结预测实验。

## 几何完整性与网格阶段

- 新主案例闭合表面：{summary['new_closed_surfaces']}/{summary['new_main_completed']}。未进行平滑或碎片删除。
- 新主案例中 {summary['new_main_multiple_solid_components_6']}/{summary['new_main_completed']} 的原始掩膜存在额外 6 邻接实体分量；
  最大实体分量占比最低 {100 * summary['new_main_largest_solid_fraction_6']['minimum']:.5f}%。应报告这些小分量，不能宣称绝无碎片。
- 50 个已有初始 FE 体素的最大实体分量占比最低
  {100 * summary['reused_solid_largest_fraction_6']['minimum']:.5f}%，z 向贯通孔隙体积分数最低
  {100 * summary['reused_pore_spanning_z_6']['minimum']:.5f}%。多个独立贯通孔道可以同时贡献该比例。
- 新生成原始三角面的 AR<5 比例平均为 {100 * summary['new_main_AR_below_5_fraction']['mean']:.2f}%。
  这是原始等值面三角剖分，不是重网格后的表面或体网格，不能与原稿的 99.95% 直接对照。
- 已有结构的原 source.stl 不在保留的旧算例路径中；本轮没有用体素重建面片冒充其原始网格统计。
- 闭合性、体素连通性不能单独证明无自交、可制造性或最终体网格质量。本轮不启动商业网格器或求解器。

## 对 3.1 组织与结论的影响

建议仍保留四个小节：代表性单场/耦合结构；方向实现精度与映射畸变；密度精度及交叉影响；
几何完整性与仿真前处理。应替换“严格正交解耦”“任意梯度下无畸变”“局部误差始终低于 1%”
“无需修复即可普遍进入仿真”等强表述。新的实验支持低维变量可编辑、若干条件下控制准确、
生成几何具有较好的主体连通和孔隙贯通，同时明确展示当前全局分位数映射的局部密度耦合。

本轮仅完成实验和结果文件，未修改 Word、生成器、FEA/CFD 数据或预测模型。
如果后续需要保留严格局部密度解耦的主张，应单独设计并验证阈值算法改进，不能通过挑选案例或
把全局准确性替代局部指标来维持该结论。

## 文件与复现

- `orientation.csv`：方向案例及全部节点诊断、固定稀疏节点的三轴细节。
- `density.csv`、`coupling.csv`、`regional_density.csv`：新案例及区域数据。
- `quality.csv`、`reused_density.csv`、`reused_groups.csv`：完整性、已有结构密度及分组统计。
- `resolution.csv`：3 个 120³/240³ 对照，属于有限几何分辨率复核，不是 FEA 网格收敛。
- `resources.csv`：各项计算耗时与峰值 RAM。
- `figures/`：4 组真实几何与量化图，PDF/SVG/600 dpi PNG/TIFF；`preview.png` 用于屏幕预览。
- `CAPTIONS_EN.md`：对应英文图注。
- 上级 `protocol.json`、`source/`、`cases/` 保存冻结配置、代码、原始 STL/体素及逐案例哈希。

从项目根目录执行 `research/geometry_validation/main.py status` 查看状态，
`research/geometry_validation/main.py report` 用保存结果重建报告；Python 使用项目约定的 tpms 环境。
"""
    path = root / "report/RESULTS_ZH.md"; path.write_text(content, encoding="utf-8")
    manifest = read(root / "report_manifest.json")
    manifest["summary_source_sha256"] = digest(Path(__file__))
    manifest["artifacts"]["report/RESULTS_ZH.md"] = digest(path)
    save(root / "report_manifest.json", manifest)
    return path


if __name__ == "__main__":
    print(write_summary(Path(sys.argv[1]).resolve()))
