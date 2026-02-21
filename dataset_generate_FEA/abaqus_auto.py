# -*- coding: utf-8 -*-
#
# TPMS结构RF-位移分析脚本 - 专业版 (适配24x24x24新模型尺寸)
# 说明: 本脚本仅负责模型创建、作业提交和RF-位移数据提取
#       详细的应力-应变分析请使用配套的后处理脚本: stress_strain_analysis.py
# 用户: pxy1118
# 日期: 2025-09-02
#

from abaqus import *
from abaqusConstants import *
import job
import os
import time
import csv
from datetime import datetime
import odbAccess
from odbAccess import *
import sys
import traceback

# ============ 新增: 输入文件可配置 =================
# 获取脚本所在目录
DEFAULT_INP = r'D:\Workplace\Papers\TPMS\dataset_generate_FEA\dataset_fea\0000\model.inp'

def _get_inp_path():
    for arg in sys.argv[1:]:
        if arg.lower().startswith('inp='):
            return arg.split('=', 1)[1].strip().strip('"')
    return DEFAULT_INP

INP_FILE = _get_inp_path()
print('\n使用输入文件: %s' % INP_FILE)
if not os.path.isfile(INP_FILE):
    raise IOError('指定的 inp 文件不存在: %s' % INP_FILE)
# ====================================================

# ============ 日志记录功能 ============
LOG_FILE = 'logs/abaqus_analysis.log'

def log_message(message, print_to_console=True):
    """日志记录函数"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
    except Exception as e:
        print(f"[LOG ERROR] 无法写入日志: {str(e)}")
    
    if print_to_console:
        print(log_entry)

# 清空旧日志
if os.path.exists(LOG_FILE):
    try:
        os.remove(LOG_FILE)
    except:
        pass
log_message("===== TPMS RF-位移分析开始（专业版 - 24x24x24新尺寸）=====")
log_message("注意: 本脚本仅提取RF-位移数据，详细的应力-应变分析")
log_message("      请使用配套的后处理脚本: stress_strain_analysis.py")
# ========================================

def safe_divide(a, b, default=0.0):
    """安全除法，避免除零错误"""
    try:
        if abs(b) < 1e-12:
            return default
        return float(a) / float(b)
    except:
        return default

def extract_rf_displacement_data(job_name, output_dir):
    """
    作业完成后提取RF-位移曲线 - 专业版
    仅负责提取和清理原始数据，不进行复杂力学计算
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        log_message(f"[OK] 创建输出目录: {os.path.abspath(output_dir)}")
    else:
        log_message(f"[INFO] 使用现有输出目录: {os.path.abspath(output_dir)}")
    
    log_message("\n开始提取RF-位移分析结果...")
    
    try:
        # 等待作业完成
        log_message(f"等待作业 {job_name} 完成...")
        mdb.jobs[job_name].waitForCompletion()
        log_message("作业已完成!")
        
        # 检查作业状态
        status = mdb.jobs[job_name].status
        status_names = {
            COMPLETED: "已完成",
            ABORTED: "已中止", 
            TERMINATED: "已终止",
            ERROR: "出错"
        }
        log_message(f"作业状态: {status_names.get(status, status)}")
        
        # 打开ODB文件
        odb_path = f"{job_name}.odb"
        if not os.path.exists(odb_path):
            log_message("[ERROR] ODB文件不存在")
            return False
        
        odb = odbAccess.openOdb(path=odb_path)
        log_message(f"[OK] 成功打开ODB文件: {odb_path}")
        
        # 初始化结果
        results = {
            'time': [],
            'displacement': [],
            'reaction_force': []
        }
        
        # 查找顶部平板参考点（承受反力的点）
        top_ref_region = None
        found_region_name = ""
        
        # 优先查找PART-2-2的参考点区域
        for region_name, region in odb.steps['Step-1'].historyRegions.items():
            if 'PART-2-2' in region_name.upper():
                outputs = region.historyOutputs.keys()
                if 'U3' in outputs and 'RF3' in outputs:
                    top_ref_region = region
                    found_region_name = region_name
                    log_message(f"[OK] 找到顶部参考点: {region_name}")
                    break
        
        # 如果没找到，查找任何有RF3和U3的区域
        if not top_ref_region:
            for region_name, region in odb.steps['Step-1'].historyRegions.items():
                outputs = region.historyOutputs.keys()
                if 'U3' in outputs and 'RF3' in outputs:
                    top_ref_region = region
                    found_region_name = region_name
                    log_message(f"[OK] 找到参考点区域: {region_name}")
                    break
        
        if top_ref_region:
            # 提取时间、位移和反力数据
            u3_data = top_ref_region.historyOutputs['U3'].data
            rf3_data = top_ref_region.historyOutputs['RF3'].data
            
            # 确保数据长度一致
            min_len = min(len(u3_data), len(rf3_data))
            
            # 提取并清理数据
            log_message("[INFO] 开始数据提取和清理...")
            
            raw_times = []
            raw_displacements = []
            raw_forces = []
            
            for i in range(min_len):
                time_val = u3_data[i][0]
                disp_val = abs(u3_data[i][1])
                force_val = abs(rf3_data[i][1])
                if (isinstance(time_val, (int, float)) and 
                    isinstance(disp_val, (int, float)) and 
                    isinstance(force_val, (int, float)) and
                    force_val < 1e10):
                    raw_times.append(time_val)
                    raw_displacements.append(disp_val)
                    raw_forces.append(force_val)
            
            log_message(f"[DEBUG] 原始提取点数: {len(raw_times)}")
            
            # 如果点数过少，尝试使用 session.XYDataFromHistory 再提取一次
            if len(raw_times) <= 1:
                log_message('[WARN] HistoryOutput 仅得到 %d 个点, 尝试二次提取 (可能原因: 步时间过短/numIntervals 未生效/分析提前结束)' % len(raw_times))
                try:
                    from abaqus import session
                    import xyPlot
                    if found_region_name.startswith('Node '):
                        _tmp = found_region_name.split()
                        inst_node = _tmp[1]
                        inst_name, node_label = inst_node.rsplit('.',1)
                        node_label_int = int(node_label)
                        rf_name = f"Reaction force: RF3 PI: {inst_name} Node {node_label_int} in NSET SET-1"
                        u_name  = f"Spatial displacement: U3 PI: {inst_name} Node {node_label_int} in NSET SET-1"
                        log_message(f"[INFO] 方式A: session.XYDataFromHistory 读取 {rf_name}")
                        try:
                            xy_rf = session.XYDataFromHistory(name='TMP_RF3', odb=odb, outputVariableName=rf_name, steps=('Step-1',), suppressQuery=True)
                            xy_u  = session.XYDataFromHistory(name='TMP_U3',  odb=odb, outputVariableName=u_name,  steps=('Step-1',), suppressQuery=True)
                        except Exception as e_session:
                            log_message(f"[WARN] 方式A失败: {e_session}; 尝试方式B xyPlot.XYDataFromHistory")
                            xy_rf = xyPlot.XYDataFromHistory(odb=odb, outputVariableName=rf_name, steps=('Step-1',), suppressQuery=True)
                            xy_u  = xyPlot.XYDataFromHistory(odb=odb, outputVariableName=u_name, steps=('Step-1',), suppressQuery=True)
                        if len(xy_rf) == len(xy_u) and len(xy_rf) > 1:
                            raw_times = [pt[0] for pt in xy_u]
                            raw_displacements = [abs(pt[1]) for pt in xy_u]
                            raw_forces = [abs(pt[1]) for pt in xy_rf]
                            log_message(f"[OK] 二次提取得到 {len(raw_times)} 个点")
                        else:
                            log_message(f"[WARN] 二次提取仍仅 {len(xy_rf)} 个点; 请检查: 步时间(timePeriod), 载荷时间, HistoryOutputRequest 设置")
                except Exception as _e2:
                    log_message(f"[WARN] 二次提取总失败: {_e2}")
            
            # 按位移排序（确保单调性）
            sorted_data = sorted(zip(raw_displacements, raw_forces, raw_times))
            
            results['displacement'] = [item[0] for item in sorted_data]
            results['reaction_force'] = [item[1] for item in sorted_data]
            results['time'] = [item[2] for item in sorted_data]
            
            log_message(f"[OK] 提取到RF-位移数据: {len(results['displacement'])}个点")
            if results['displacement']:
                log_message(f"[数据] 位移范围: {min(results['displacement']):.4f} ~ {max(results['displacement']):.4f} mm")
            if results['reaction_force']:
                log_message(f"[数据] 反力范围: {min(results['reaction_force']):.2f} ~ {max(results['reaction_force']):.2f} N")
            
            # 保存RF-位移曲线到CSV (符合后处理脚本要求的格式)
            csv_path = os.path.join(output_dir, 'rf_displacement_curve.csv')
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Time(s)', 'Displacement(mm)', 'Reaction_Force(N)', 'Comments'])
                
                for i in range(len(results['time'])):
                    writer.writerow([
                        f"{results['time'][i]:.6f}",
                        f"{results['displacement'][i]:.6f}",
                        f"{results['reaction_force'][i]:.2f}",
                        'Cleaned_Data'
                    ])
            
            log_message(f"[OK] RF-位移曲线已保存: {csv_path}")
            log_message(f"[提示] 请使用后处理脚本分析数据: python stress_strain_analysis.py --input {csv_path}")
            
            # 生成简要报告
            report_path = os.path.join(output_dir, 'data_summary.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("TPMS结构RF-位移数据摘要\n")
                f.write("="*50 + "\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"输入文件: {INP_FILE}\n")
                f.write(f"作业名称: {job_name}\n\n")
                
                f.write("数据统计:\n")
                f.write(f"  数据点数量: {len(results['time'])}\n")
                if results['displacement']:
                    f.write(f"  最大位移: {max(results['displacement']):.4f} mm\n")
                if results['reaction_force']:
                    f.write(f"  最大反力: {max(results['reaction_force']):.2f} N\n")
                
                f.write("\n后续分析说明:\n")
                f.write("1. 本文件仅包含原始RF-位移数据\n")
                f.write("2. 详细的应力-应变分析请使用配套后处理脚本:\n")
                f.write("   python stress_strain_analysis.py --input rf_displacement_curve.csv\n")
                f.write("3. 后处理脚本将计算:\n")
                f.write("   - 弹性模量 (使用弹性阶段线性段)\n")
                f.write("   - 屈服强度 (0.2%偏移法)\n")
                f.write("   - 极限强度\n")
                f.write("   - 能量吸收密度\n")
                f.write("   - 生成专业分析报告和图表\n")
            
            log_message(f"[OK] 数据摘要已保存: {report_path}")
        
        else:
            log_message("[ERROR] 未找到参考点的RF和位移数据")
            odb.close()
            return False
        
        odb.close()
        return True
        
    except Exception as e:
        log_message(f"[ERROR] 分析过程出错: {str(e)}")
        log_message(traceback.format_exc())
        return False

# ================== Abaqus模型设置（按照完整分析流程）==================

session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=210.915008544922, 
    height=147.490005493164)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from caeModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)

# 导入几何模型
# 记录导入前的部件列表，以支持动态部件名称
existing_parts = set(mdb.models['Model-1'].parts.keys())

# 为了兼容性，如果存在PART-1尝试清理
if 'PART-1' in mdb.models['Model-1'].parts:
    del mdb.models['Model-1'].parts['PART-1']
    
mdb.models['Model-1'].PartFromInputFile(inputFileName=INP_FILE)

# 动态识别导入的部件名称
current_parts = set(mdb.models['Model-1'].parts.keys())
new_parts = list(current_parts - existing_parts)

if len(new_parts) > 0:
    imported_part_name = new_parts[0]
elif 'PART-1' in current_parts:
    imported_part_name = 'PART-1'
else:
    # 尝试查找除Part-2(刚性板)之外的任意部件
    candidates = [n for n in current_parts if n != 'Part-2']
    if candidates:
        imported_part_name = candidates[0]
    else:
        raise Exception("无法识别导入的部件名称，请检查INP文件")

log_message(f"[INFO] 成功识别导入部件: {imported_part_name}")
p = mdb.models['Model-1'].parts[imported_part_name]
session.viewports['Viewport: 1'].setValues(displayedObject=p)

# ==================== 新增：修复Set-1定义以避免索引越界 ====================
# 检查并修复部件中的Set-1定义
log_message(f"[INFO] 检查并修复部件 {imported_part_name} 中的Set-1定义...")

# 获取实际最大单元编号
if len(p.elements) > 0:
    max_elem_id = max([e.label for e in p.elements])
    log_message(f"[INFO] {imported_part_name} 实际最大单元编号: {max_elem_id}")
    
    # 删除可能存在的错误Set-1
    if 'Set-1' in p.sets:
        del p.sets['Set-1']
        log_message("[INFO] 删除旧的Set-1定义")
    
    # 创建新的、正确的Set-1，包含所有有效单元
    region = p.Set(elements=p.elements, name='Set-1')
    log_message("[OK] 已创建正确的Set-1，包含所有有效单元")
else:
    log_message(f"[ERROR] 部件 {imported_part_name} 中没有单元！")
    raise Exception(f"部件 {imported_part_name} 中没有单元！")

# 检查节点信息
if len(p.nodes) > 0:
    max_node_id = max([n.label for n in p.nodes])
    log_message(f"[INFO] {imported_part_name} 实际最大节点编号: {max_node_id}")
else:
    log_message(f"[WARN] 部件 {imported_part_name} 中没有节点")
# ========================================================================

# 创建刚性平面部件 Part-2 (适配24x24x24模型)
s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=200.0)
g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
s.setPrimaryObject(option=STANDALONE)
# 修改为适配24x24x24模型的尺寸 (-2.0, -2.0) 到 (26.0, 26.0)
s.rectangle(point1=(-2.0, -2.0), point2=(26.0, 26.0))
p = mdb.models['Model-1'].Part(name='Part-2', dimensionality=THREE_D, type=DISCRETE_RIGID_SURFACE)
p = mdb.models['Model-1'].parts['Part-2']
p.BaseShell(sketch=s)
s.unsetPrimaryObject()
p = mdb.models['Model-1'].parts['Part-2']
session.viewports['Viewport: 1'].setValues(displayedObject=p)

# 删除草图
del mdb.models['Model-1'].sketches['__profile__']

# 设置视图
session.viewports['Viewport: 1'].view.setValues(nearPlane=178.665, 
    farPlane=217.315, width=153.781, height=79.8545, viewOffsetX=11.4117, 
    viewOffsetY=-2.03815)

# 添加参考点和质量属性
p = mdb.models['Model-1'].parts['Part-2']
v1, e, d1, n = p.vertices, p.edges, p.datums, p.nodes
# 使用更明确的中心点参考点
p.ReferencePoint(point=(12.0, 12.0, 0.0))  # 24x24平面的中心点

# 切换到装配体
a = mdb.models['Model-1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)

# 创建装配实例
a.DatumCsysByDefault(CARTESIAN)
p = mdb.models['Model-1'].parts[imported_part_name]
a.Instance(name='PART-1-1', part=p, dependent=ON)
p = mdb.models['Model-1'].parts['Part-2']
a.Instance(name='Part-2-1', part=p, dependent=ON)
a = mdb.models['Model-1'].rootAssembly
p = mdb.models['Model-1'].parts['Part-2']
a.Instance(name='Part-2-2', part=p, dependent=ON)
a = mdb.models['Model-1'].rootAssembly
# 修改平移距离为24.0（适配24x24x24模型）
a.translate(instanceList=('Part-2-2', ), vector=(0.0, 0.0, 24.0))

# 设置显示选项
session.viewports['Viewport: 1'].assemblyDisplay.setValues(adaptiveMeshConstraints=ON)
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON, 
    engineeringFeatures=ON)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=OFF)

# 设置参考点和质量属性
p = mdb.models['Model-1'].parts['Part-2']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
p = mdb.models['Model-1'].parts['Part-2']
r = p.referencePoints
# 修改为正确的参考点索引
refPoints=(r[2], )
region=p.Set(referencePoints=refPoints, name='Set-1')
mdb.models['Model-1'].parts['Part-2'].engineeringFeatures.PointMassInertia(
    name='Inertia-1', region=region, mass=1.0, i11=1.0, i22=1.0, i33=1.0, 
    alpha=0.0, composite=0.0)

# 定义材料属性
mdb.models['Model-1'].Material(name='Ti6Al4V')
mdb.models['Model-1'].materials['Ti6Al4V'].Density(table=((4.43e-09, ), ))
mdb.models['Model-1'].materials['Ti6Al4V'].Elastic(table=((112000.0, 0.33), ))
mdb.models['Model-1'].materials['Ti6Al4V'].Plastic(scaleStress=None, table=((
    946.0, 0.0), (982.0, 0.0034), (998.0, 0.0076), (1010.0, 0.0123), (1020.0, 
    0.0178), (1027.0, 0.024), (1033.0, 0.0303), (1037.0, 0.0364), (1039.0, 
    0.0426)))
mdb.models['Model-1'].materials['Ti6Al4V'].JohnsonCookDamageInitiation(table=(
    (-0.68, 0.73, -0.25, 0.0, 0.0, 1670.0, 990.0, 1.0), 
))
# 添加损伤演化（位移型）
mdb.models['Model-1'].materials['Ti6Al4V'].johnsonCookDamageInitiation.DamageEvolution(
    type=DISPLACEMENT, table=((0.1, ), ))
# 重新生成装配体
a = mdb.models['Model-1'].rootAssembly
a.regenerate()
a = mdb.models['Model-1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)

# 创建分析步 (保持速度，通过延长时间实现12mm位移)
mdb.models['Model-1'].ExplicitDynamicsStep(name='Step-1', previous='Initial', 
    timePeriod=2.0, massScaling=((SEMI_AUTOMATIC, MODEL, THROUGHOUT_STEP, 0.0, 
    1.5e-05, BELOW_MIN, 100, 0, 0.0, 0.0, 0, None), ), improvedDtMethod=ON)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Step-1')

# 设置输出请求
mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(variables=(
    'S', 'SVAVG', 'PEEQ', 'LE', 'U', 'V', 'A', 'RF', 'CSTRESS', 'STATUS'), 
    timeInterval=0.05)
mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].suppress()
mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].resume()
mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].suppress()
mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].resume()

# 设置历史输出请求
# 检查Set-1是否存在
if 'Set-1' in mdb.models['Model-1'].rootAssembly.allInstances['Part-2-2'].sets:
    regionDef=mdb.models['Model-1'].rootAssembly.allInstances['Part-2-2'].sets['Set-1']
    mdb.models['Model-1'].HistoryOutputRequest(name='RF', createStepName='Step-1', 
        variables=('U1', 'U2', 'U3', 'UR1', 'UR2', 'UR3', 'RF1', 'RF2', 'RF3', 
        'RM1', 'RM2', 'RM3'), timeInterval=0.05, region=regionDef, 
        sectionPoints=DEFAULT, rebar=EXCLUDE)
else:
    log_message("[WARN] Part-2-2中的Set-1未定义，跳过HistoryOutputRequest")

# 设置接触
session.viewports['Viewport: 1'].assemblyDisplay.setValues(interactions=ON, 
    constraints=ON, connectors=ON, engineeringFeatures=ON, 
    adaptiveMeshConstraints=OFF)
mdb.models['Model-1'].ContactProperty('IntProp-1')
mdb.models['Model-1'].interactionProperties['IntProp-1'].TangentialBehavior(
    formulation=PENALTY, directionality=ISOTROPIC, slipRateDependency=OFF, 
    pressureDependency=OFF, temperatureDependency=OFF, dependencies=0, table=((
    0.2, ), ), shearStressLimit=None, maximumElasticSlip=FRACTION, 
    fraction=0.005, elasticSlipStiffness=None)
mdb.models['Model-1'].ContactExp(name='Int-1', createStepName='Step-1')
mdb.models['Model-1'].interactions['Int-1'].includedPairs.setValuesInStep(
    stepName='Step-1', useAllstar=ON)
mdb.models['Model-1'].interactions['Int-1'].contactPropertyAssignments.appendInStep(
    stepName='Step-1', assignments=((GLOBAL, SELF, 'IntProp-1'), ))
mdb.models['Model-1'].interactions['Int-1'].wearSurfacePropertyAssignments.appendInStep(
    stepName='Step-1', assignments=((GLOBAL, ''), ))
mdb.models['Model-1'].interactions['Int-1'].move('Step-1', 'Initial')

# 设置边界条件
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
    predefinedFields=ON, interactions=OFF, constraints=OFF, 
    engineeringFeatures=OFF)
session.viewports['Viewport: 1'].view.setValues(session.views['Bottom'])

a = mdb.models['Model-1'].rootAssembly
r1 = a.instances['Part-2-1'].referencePoints
refPoints1=(r1[2], )
region = a.Set(referencePoints=refPoints1, name='Set-1')
mdb.models['Model-1'].EncastreBC(name='BC-1', createStepName='Step-1', 
    region=region, localCsys=None)
mdb.models['Model-1'].boundaryConditions['BC-1'].move('Step-1', 'Initial')

a = mdb.models['Model-1'].rootAssembly
r1 = a.instances['Part-2-2'].referencePoints
refPoints1=(r1[2], )
region = a.Set(referencePoints=refPoints1, name='Set-2')
mdb.models['Model-1'].DisplacementBC(name='BC-2', createStepName='Step-1', 
    region=region, u1=0.0, u2=0.0, u3=UNSET, ur1=0.0, ur2=0.0, ur3=0.0, 
    amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='', 
    localCsys=None)
mdb.models['Model-1'].boundaryConditions['BC-2'].move('Step-1', 'Initial')

a = mdb.models['Model-1'].rootAssembly
r1 = a.instances['Part-2-2'].referencePoints
refPoints1=(r1[2], )
region = a.Set(referencePoints=refPoints1, name='Set-3')
# 保持速度为-1.2，通过延长分析步实现2.4mm位移
mdb.models['Model-1'].VelocityBC(name='BC-3', createStepName='Step-1', 
    region=region, v1=UNSET, v2=UNSET, v3=-1.2, vr1=UNSET, vr2=UNSET, 
    vr3=UNSET, amplitude=UNSET, localCsys=None, distributionType=UNIFORM, 
    fieldName='')

# 网格划分 (修改网格尺寸为2.4)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=ON, loads=OFF, 
    bcs=OFF, predefinedFields=OFF, connectors=OFF)
session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
    meshTechnique=ON)
p = mdb.models['Model-1'].parts['Part-2']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=OFF, 
    engineeringFeatures=OFF, mesh=ON)
session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
    meshTechnique=ON)
p = mdb.models['Model-1'].parts['Part-2']
# 修改网格尺寸为2.4（适配新模型尺寸）
p.seedPart(size=2.4, deviationFactor=0.1, minSizeFactor=0.1)
p = mdb.models['Model-1'].parts['Part-2']
p.generateMesh()
a1 = mdb.models['Model-1'].rootAssembly
a1.regenerate()
a = mdb.models['Model-1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=OFF)
session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
    meshTechnique=OFF)

# 截面分配
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON, 
    engineeringFeatures=ON, mesh=OFF)
session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
    meshTechnique=OFF)
p = mdb.models['Model-1'].parts['Part-2']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
p = mdb.models['Model-1'].parts[imported_part_name]
session.viewports['Viewport: 1'].setValues(displayedObject=p)

mdb.models['Model-1'].HomogeneousSolidSection(name='Section-1', 
    material='Ti6Al4V', thickness=None)

# ==================== 修改：使用安全的Set创建方式 ====================
p = mdb.models['Model-1'].parts[imported_part_name]
# 删除不安全的getSequenceFromMask方式，改用直接使用所有元素
region = p.Set(elements=p.elements, name='Set-1_Part')  # 使用不同名称避免冲突
p = mdb.models['Model-1'].parts[imported_part_name]
p.SectionAssignment(region=region, sectionName='Section-1', offset=0.0, 
    offsetType=MIDDLE_SURFACE, offsetField='', 
    thicknessAssignment=FROM_SECTION)
# ===================================================================

a1 = mdb.models['Model-1'].rootAssembly
a1.regenerate()
a = mdb.models['Model-1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)

# ================== 作业提交和分析 ==================
job_name = 'Job-1'
# 修改：使输出目录相对于 INP 文件所在的文件夹，而不是固定在脚本运行目录
# 这样可以确保每个样本的结果保存在各自的文件夹中
inp_folder = os.path.dirname(os.path.abspath(INP_FILE))
output_dir = os.path.join(inp_folder, 'output')

# 清理旧作业
if job_name in mdb.jobs.keys():
    log_message(f"清理已存在的作业: {job_name}")
    del mdb.jobs[job_name]

# 创建作业
log_message(f"创建RF分析作业: {job_name}")
mdb.Job(name=job_name, model='Model-1', description='TPMS RF-Displacement Analysis - Professional Version (24x24x24)', 
    type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None, 
    memory=90, memoryUnits=PERCENTAGE, explicitPrecision=SINGLE, 
    nodalOutputPrecision=SINGLE, echoPrint=OFF, modelPrint=OFF, 
    contactPrint=OFF, historyPrint=OFF, userSubroutine='', scratch='', 
    resultsFormat=ODB, numDomains=12, activateLoadBalancing=False, 
    numThreadsPerMpiProcess=0, numCpus=12)

# 提交作业
log_message(f"提交作业: {job_name}")
mdb.jobs[job_name].submit(consistencyChecking=OFF)

# 等待完成并提取数据
try:
    log_message("等待作业完成并提取RF-位移数据...")
    success = extract_rf_displacement_data(job_name, output_dir)
    
    if success:
        log_message("\nRF-位移数据提取完成！")
        log_message("结果已保存至: " + os.path.abspath(output_dir))
        log_message("  - rf_displacement_curve.csv: RF-位移曲线数据（已清理）")
        log_message("  - data_summary.txt: 数据摘要")
        log_message("\n下一步:")
        log_message("  请使用后处理脚本分析数据:")
        log_message(f"  python stress_strain_analysis.py --input {os.path.join(output_dir, 'rf_displacement_curve.csv')}")
    else:
        log_message("[ERROR] RF-位移数据提取失败")
        sys.exit(1)
        
except Exception as e:
    log_message(f"[ERROR] 作业执行或数据提取出错: {str(e)}")
    log_message(traceback.format_exc())
    sys.exit(1)

log_message(f"\n===== TPMS RF-位移分析结束（专业版 - 24x24x24新尺寸）=====")
log_message("注意: 本脚本仅提取RF-位移数据，详细的应力-应变分析")
log_message("      请使用配套的后处理脚本: stress_strain_analysis.py")