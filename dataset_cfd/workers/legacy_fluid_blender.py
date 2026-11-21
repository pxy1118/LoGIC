"""The archived EXACT Boolean recipe, operating only on a new working copy."""
import os
import json
import time
from pathlib import Path
import bpy

case=Path(os.environ['CFD_V2_CASE']).resolve()
root=Path(__file__).resolve().parents[1]
if not case.is_relative_to(root/'runs'):raise RuntimeError('Case outside CFD runs')
cfg=json.loads((case/'protocol.json').read_text(encoding='utf8'))['geometry']
working_scale=cfg.get('processing_scale',1.0)
started=time.time()
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
bpy.ops.wm.stl_import(filepath=str(case/('solid_processing.stl' if working_scale!=1 else 'solid_initial.stl')))
parts=[o for o in bpy.context.selected_objects if o.type=='MESH']
if len(parts)!=1:raise RuntimeError('Expected one initial solid')
solid=parts[0]
origin=[x*working_scale for x in cfg['box_origin_mm']]
dims=[x*working_scale for x in cfg['box_dimensions_mm']]
bpy.ops.mesh.primitive_cube_add(size=1,location=[origin[i]+dims[i]/2 for i in range(3)])
fluid=bpy.context.object;fluid.name='fluid_boolean_raw';fluid.dimensions=dims
bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
modifier=fluid.modifiers.new(name='Boolean_Diff',type='BOOLEAN')
modifier.operation='DIFFERENCE';modifier.solver='EXACT';modifier.object=solid
bpy.ops.object.modifier_apply(modifier=modifier.name)
bpy.data.objects.remove(solid,do_unlink=True)
bpy.ops.object.select_all(action='DESELECT');fluid.select_set(True)
bpy.context.view_layer.objects.active=fluid
bpy.ops.wm.stl_export(filepath=str(case/('fluid_boolean_working.stl' if working_scale!=1 else 'fluid_boolean_raw.stl')),export_selected_objects=True)
r={'status':'BOOLEAN_EXPORTED','elapsed_s':time.time()-started,
   'backend':'Blender EXACT','blender_version':bpy.app.version_string,
   'working_scale':working_scale,'box_origin_working':origin,'box_dimensions_working':dims,
   'polygons':len(fluid.data.polygons)}
(case/'blender_report.json').write_text(json.dumps(r,indent=2),encoding='utf8')
print(r,flush=True)
