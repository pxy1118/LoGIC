"""Remove verified intermediate files while preserving resumable final artifacts."""
import re
from .common import ROOT,inside,read,save,digest,verify_files,lock,now


INTERMEDIATES={'source.stl','removed_components.stl','wrapped.stl','fea_surface.stl',
               'volume_3matic.inp','volume.inp','remeshed.stl'}


def removable(name):
    return name in INTERMEDIATES or re.fullmatch(r'surface_stage[0-9]+\.stl',name) is not None


def verify_case_files(case,hashes):
    """Missing files require an explicit deletion journal and intact retained files."""
    case=inside(case)
    journal=case/'intermediate_cleanup.json'
    record=read(journal) if journal.exists() else {}
    removed=record.get('deleted_files',{})
    if record:
        anchors=record.get('retained_hashes',{})
        if not anchors:raise RuntimeError('Intermediate cleanup lacks retained artifact hashes')
        verify_files(case,anchors)
    for name,expected in hashes.items():
        path=inside(case/name,case)
        if path.is_file():verify_files(case,{name:expected})
        elif not (removable(name) and removed.get(name,{}).get('sha256')==expected):
            raise RuntimeError('Missing or changed artifact: '+str(path))


def compact_case(case):
    from . import runtime
    case=inside(case)
    # A preserved or suspended external job must keep every input it may still use.
    with lock(ROOT/'runs/.external.lock'):
        if runtime.active_external() or list(case.glob('*.lck')):
            return dict(skipped='external job active',removed_bytes=0)
        geometry=read(case/'geometry_quality.json') if (case/'geometry_quality.json').exists() else {}
        mesh=read(case/'mesh_quality.json') if (case/'mesh_quality.json').exists() else {}
        mesher=read(case/'mesher_result.json') if (case/'mesher_result.json').exists() else {}
        complete=mesh.get('passed') and 'voxel_fea.npz' in mesh.get('hashes',{})
        rejected=geometry.get('passed') is False or mesh.get('passed') is False or mesher.get('status')=='FAILED'
        if not complete and not rejected:return dict(skipped='stage incomplete',removed_bytes=0)
        for receipt in (geometry,mesh):
            if receipt:verify_case_files(case,receipt.get('hashes',{}))
        has_input=(case/'analysis.inp').exists()
        if has_input:
            analysis=read(case/'analysis.json')
            verify_files(case,{'analysis.inp':analysis['input_sha256']})
        if complete:
            required=['design.json','inputs.npz','voxel_fea.npz','geometry_features.json','remeshed.stl']
            required+=['analysis.inp','analysis.json'] if has_input else ['volume.inp']
            if not geometry.get('passed'):raise RuntimeError('Mesh cleanup requires accepted initial geometry')
            if (case/'midpoint_reconstruction.json').exists():
                verify_case_files(case,read(case/'midpoint_reconstruction.json')['hashes'])
        else:
            required=['design.json']
            if has_input:required+=['analysis.inp','analysis.json']
            if (case/'inputs.npz').exists():required.append('inputs.npz')
        for name in required:
            if not (case/name).is_file():raise RuntimeError('Cleanup would leave no required artifact: '+name)
        retained=set(required)
        retained.update(name for name in ['geometry_quality.json','mesh_quality.json','component_cleanup.json',
                        'midpoint_reconstruction.json','mesher_result.json'] if (case/name).is_file())
        journal=case/'intermediate_cleanup.json'
        old=read(journal) if journal.exists() else {}
        removed=dict(old.get('deleted_files',{}));targets=[]
        for path in case.iterdir():
            if path.is_file() and removable(path.name) and path.name not in retained:
                resolved=inside(path,case)
                entry=dict(sha256=digest(resolved),bytes=resolved.stat().st_size)
                if path.name in removed and removed[path.name]!=entry:
                    raise RuntimeError('Previously cleaned intermediate was recreated with different contents')
                removed[path.name]=entry;targets.append(resolved)
        anchors={name:digest(case/name) for name in sorted(retained)}
        record=dict(policy='compact_intermediates_v1',utc=now(),status='PRUNING',
                    stage='input_ready' if has_input and complete else ('mesh_ready' if complete else 'rejected'),
                    deleted_files=removed,retained_hashes=anchors)
        # Write deletion intent first: interruption between unlink and receipt update remains resumable.
        save(journal,record)
        freed=0
        for path in targets:
            if digest(path)!=removed[path.name]['sha256']:raise RuntimeError('Intermediate changed during cleanup')
            path.unlink();freed+=removed[path.name]['bytes']
        record.update(status='COMPLETED',total_removed_bytes=sum(value['bytes'] for name,value in removed.items() if not (case/name).exists()))
        save(journal,record);verify_files(case,anchors)
        return dict(removed_bytes=freed,total_removed_bytes=record['total_removed_bytes'],stage=record['stage'])
