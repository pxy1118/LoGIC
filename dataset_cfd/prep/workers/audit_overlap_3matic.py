"""Read-only comparison of the overlap counter and explicitly toleranced marking."""
import os,json,traceback
from pathlib import Path
import trimatic

case=Path(os.environ['FEA_V2_CASE'])
rows=[]
try:
    for path in json.loads((case/'requests.json').read_text()):
        part=trimatic.import_part_stl(path,fix_normals=True)
        if isinstance(part,(list,tuple)):part=part[0]
        row=dict(path=path,default_overlap=trimatic.get_number_of_overlapping_triangles(part),
                 intersections=trimatic.get_number_of_intersecting_triangles(part))
        for tolerance in (.1,.001,.00001):
            trimatic.mark_entities(part,mark=False)
            marks=trimatic.mark_overlapping_triangles(part,tolerance=tolerance)
            if isinstance(marks,(list,tuple)):marks=marks[0] if marks else None
            try:row[str(tolerance)]=0 if marks is None else marks.number_of_triangles
            except AttributeError as exc:
                if str(exc)!='Invalid object.':raise
                row[str(tolerance)]=0
        rows.append(row);trimatic.delete(part)
    (case/'result.json').write_text(json.dumps(rows,indent=2))
except Exception:
    (case/'error.txt').write_text(traceback.format_exc())
finally:trimatic.exit()
