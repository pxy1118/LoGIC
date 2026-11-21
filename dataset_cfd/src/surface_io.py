"""Preserve native double coordinates through ASCII STL and geometry checks.

The writer uses only the standard library so the embedded 3-matic Python
can use it. Imports for the main-runtime reader are deliberately deferred.
"""
import math
from pathlib import Path


def write_ascii_stl64(path, vertices, triangles, scale=1.):
    if not math.isfinite(scale) or scale<=0:
        raise ValueError('Coordinate scale must be finite and positive')
    points=[tuple(float(v)/scale for v in p) for p in vertices]
    if any(len(p)!=3 or not all(math.isfinite(v) for v in p) for p in points):
        raise ValueError('Invalid native coordinates')
    with Path(path).open('x',encoding='ascii',newline='\n') as f:
        f.write('solid fluid_native_float64\n')
        count=0
        for indices in triangles:
            ids=tuple(int(i) for i in indices)
            if len(ids)!=3 or min(ids)<0 or max(ids)>=len(points):
                raise ValueError('Native triangle indexing must be zero-based')
            a,b,c=[points[i] for i in ids]
            u=[b[i]-a[i] for i in range(3)];v=[c[i]-a[i] for i in range(3)]
            cross=(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0])
            norm=math.sqrt(sum(x*x for x in cross))
            if norm==0:raise ValueError('Native surface contains a zero-area triangle')
            normal=tuple(x/norm for x in cross)
            f.write('facet normal '+' '.join(format(x,'.17g') for x in normal)+'\nouter loop\n')
            for p in (a,b,c):f.write('vertex '+' '.join(format(x,'.17g') for x in p)+'\n')
            f.write('endloop\nendfacet\n');count+=1
        f.write('endsolid fluid_native_float64\n')
    return {'triangles':count,'native_points':len(points),'format':'ASCII STL, 17 significant digits',
            'physical_unit':'mm','working_coordinate_scale':scale}


def read_ascii_stl64(path):
    """Avoid vtkSTLReader's float32 output when auditing our float64 STL."""
    import numpy as np
    import pyvista as pv
    def coordinates():
        with Path(path).open(encoding='ascii') as f:
            for line in f:
                fields=line.split()
                if fields and fields[0]=='vertex':
                    if len(fields)!=4:raise ValueError('Malformed STL vertex')
                    for value in fields[1:]:yield float(value)
    values=np.fromiter(coordinates(),dtype=np.float64)
    if len(values)==0 or len(values)%9 or not np.isfinite(values).all():
        raise ValueError('Invalid ASCII STL triangle coordinates')
    xyz=values.reshape(-1,3)
    faces=np.column_stack([np.full(len(xyz)//3,3),np.arange(len(xyz)).reshape(-1,3)])
    return pv.PolyData(xyz,faces).clean(tolerance=0.)
