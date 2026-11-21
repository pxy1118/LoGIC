"""Finish a fixed-size mesh pool, then solve the explicitly selected default protocol."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import read, save, digest, object_hash, verify_files, safe_id, inside, lock, now
from src import runtime
from src.design import extend_run
from src.parallel_prepare import prepare_parallel, assert_generators_idle
from src.storage import verify_case_files
from src.package import canonical_group
from src.material import audit_material
from scripts.ten_fea_trial import solve_case
from scripts.fea_progress import SampleProgress


RUNNER_FILES = ('scripts/dataset_batch.py', 'scripts/ten_fea_trial.py',
                'scripts/clean_dataset.py', 'scripts/renumber_training.py', 'scripts/fea_progress.py')


def ensure_clean_dataset(run):
    from scripts.clean_dataset import build_dataset
    with lock(ROOT/'datasets/.training_index.lock'):
        build_dataset(run)


def sync_clean_result(run, source_id):
    from scripts.clean_dataset import sync_case_result
    with lock(ROOT/'datasets/.training_index.lock'):
        return sync_case_result(run, source_id)


def reject_failed_case(run,case,cfg,error):
    runtime.stop_case(case,cfg)
    audit=ROOT/'audits/failed_samples'/run.name/case.name/uuid.uuid4().hex
    audit.mkdir(parents=True)
    results=('extraction.json','metrics.json','quality.json','curve.npz','curve.png','stress_strain.csv',
             'raw_history.csv.gz','energy_history.json.gz','field_diagnostics.json','sparse_displacements.json.gz',
             'export_manifest.json','failed_export_manifest.json','cleanup_receipt.json','failure_diagnostics.json')
    save(audit/'failure.json',dict(utc=now(),sample_id=case.name,error=str(error),training_eligible=False,
        input_sha256=digest(case/'analysis.inp') if (case/'analysis.inp').exists() else None,status='REJECTED'))
    with zipfile.ZipFile(audit/'logs.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in case.iterdir():
            if path.is_file() and (path.suffix in ('.sta','.dat','.msg','.log') or '.msg.' in path.name):
                archive.write(path,path.name)
        for name in results:
            path=case/name
            if path.exists() and path.suffix=='.json':archive.write(path,name)
        for name in ('solver_execution.json','solver_retry.json'):
            if (case/name).exists():archive.write(case/name,name)
    deleted=[]
    for path in list(case.iterdir()):
        scratch=path.name.startswith(('analysis.','preflight.')) and path.name not in ('analysis.inp','analysis.json')
        if path.is_file() and (scratch or path.name in results):
            deleted.append(dict(name=path.name,bytes=path.stat().st_size));path.unlink()
    if (case/'datacheck.json').exists():
        try:passed=read(case/'datacheck.json')['passed']
        except (ValueError,KeyError):passed=False
        if not passed:
            shutil.copy2(case/'datacheck.json',audit/'datacheck.json');(case/'datacheck.json').unlink()
    save(audit/'cleanup.json',dict(utc=now(),deleted=deleted,analysis_input_retained=True))
    save(case/'fea_failure.json',dict(utc=now(),error=str(error),audit=str(audit),training_eligible=False))
    runtime.state(case,'FEA_FAILED',error=str(error))
    return str(audit)


def solve_with_retry(run,case,cfg):
    try:
        solve_case(case,cfg)
    except runtime.InitializationStalledError as exc:
        if runtime.effective_solver_config(case,cfg)['solver']['cpus']!=16:raise
        audit=reject_failed_case(run,case,cfg,exc)
        save(case/'solver_execution.json',dict(input_sha256=read(case/'analysis.json')['input_sha256'],
            cpus=8,domains=8,threads_per_mpi_process=1,utc=now(),reason='Retry stalled 16-CPU Explicit initialization',previous_attempt_audit=audit))
        journal=dict(status='RETRY_RUNNING',utc=now(),primary_error=str(exc),primary_attempt_audit=audit,
                     retry_cpus=8,retry_domains=8,automatic_retry_limit=1)
        save(case/'solver_retry.json',journal)
        print(case.name+' initialization stalled; retrying at 8 CPUs / 8 domains',flush=True)
        try:solve_case(case,cfg)
        except BaseException as retry_error:
            journal.update(status='FAILED' if isinstance(retry_error,Exception) else 'INTERRUPTED',
                           utc=now(),retry_error=str(retry_error));save(case/'solver_retry.json',journal)
            raise
        else:
            journal.update(status='COMPLETED',utc=now());save(case/'solver_retry.json',journal)
            (case/'fea_failure.json').unlink(missing_ok=True)
    else:
        if (case/'solver_retry.json').exists():
            journal=read(case/'solver_retry.json');journal.update(status='COMPLETED',utc=now())
            save(case/'solver_retry.json',journal);(case/'fea_failure.json').unlink(missing_ok=True)


def check_run(run):
    cfg = read(run/'config.json'); manifest = read(run/'manifest.json')
    if manifest['purpose'] != 'dataset' or digest(run/'config.json') != manifest['config_sha256']:
        raise RuntimeError('Dataset configuration changed')
    selection = cfg.get('numerical_calibration', {})
    if selection.get('status') != 'USER_SELECTED_DEFAULT':
        raise RuntimeError('This entry point requires an explicitly selected default')
    approval_path = inside(ROOT/selection['record'])
    approval = read(approval_path)
    if approval['status'] != 'USER_SELECTED_DEFAULT' or approval['new_default_sha256'] != manifest['config_sha256']:
        raise RuntimeError('Selected default does not match this dataset')
    verify_files(approval_path.parent, approval['evidence_files'])
    verify_files(ROOT, manifest['pipeline_hashes'])
    verify_files(run/'protocol/source', manifest['generator_hashes'])
    revision = run/'protocol/revisions'/object_hash(manifest['pipeline_hashes'])
    verify_files(revision, manifest['pipeline_hashes'])
    runtime.WORKER_SOURCE = revision/'workers'
    return cfg, manifest, approval_path


def rejected(case):
    for name in ('geometry_quality.json', 'mesh_quality.json'):
        if (case/name).exists() and read(case/name).get('passed') is False:
            return True
    if (case/'mesher_result.json').exists() and read(case/'mesher_result.json').get('status') == 'FAILED':
        return True
    # An interrupted mesh without a result cannot be reused by runtime.mesh_case.
    # Keep its artifacts and replace the candidate after the global idle check.
    if (case/'mesher_config.json').exists() and not (case/'mesher_result.json').exists():
        return True
    return read(case/'state.json')['stage'] in ('GEOMETRY_REJECTED', 'MESH_REJECTED')


def inventory(run, cfg, manifest, cache=False):
    """Count accepted initial meshes, removing exact voxel/axial-rotation duplicates."""
    import numpy as np
    ready = []; pending = []; exclusions = []; groups = set()
    previous = read(run/'batch_prepare.json') if (run/'batch_prepare.json').exists() else {}
    prior_failures = {r['sample_id'] for r in previous.get('rejections', [])}
    for row in manifest['samples']:
        case = run/'samples'/row['sample_id']
        design = read(case/'design.json'); declared = design.pop('design_hash')
        if declared != row['design_hash'] or object_hash(design) != declared:
            raise RuntimeError('Design controls changed: '+case.name)
        if rejected(case) or case.name in prior_failures:
            exclusions.append(dict(sample_id=case.name, reason='rejected_candidate')); continue
        required = ('analysis.inp', 'analysis.json', 'geometry_quality.json', 'mesh_quality.json', 'voxel_fea.npz')
        if not all((case/name).exists() for name in required):
            pending.append(row); continue
        analysis = read(case/'analysis.json')
        if analysis['config'] != cfg:
            raise RuntimeError('Prepared input configuration changed: '+case.name)
        if not read(case/'geometry_quality.json')['passed'] or not read(case/'mesh_quality.json')['passed']:
            raise RuntimeError('Prepared mesh is not accepted: '+case.name)
        voxel_hash = digest(case/'voxel_fea.npz')
        if read(case/'mesh_quality.json')['hashes'].get('voxel_fea.npz') != voxel_hash:
            raise RuntimeError('Initial FE voxel changed: '+case.name)
        path = case/'batch_geometry_group.json'
        group_record = read(path) if path.exists() else {}
        if group_record.get('voxel_sha256') != voxel_hash:
            with np.load(case/'voxel_fea.npz') as arrays:
                group_record = dict(voxel_sha256=voxel_hash, group=canonical_group(arrays['voxel']))
            if cache: save(path, group_record)
        if group_record['group'] in groups:
            exclusions.append(dict(sample_id=case.name, reason='duplicate_initial_geometry')); continue
        groups.add(group_record['group'])
        ready.append(dict(row, geometry_group=group_record['group'], input_sha256=analysis['input_sha256']))
    return ready, pending, exclusions


def verify_input(case, row, cfg):
    analysis = read(case/'analysis.json')
    if analysis['config'] != cfg or analysis['input_sha256'] != row['input_sha256']:
        raise RuntimeError('Selected input changed: '+case.name)
    verify_files(case, {'analysis.inp': row['input_sha256']})
    for name in ('geometry_quality.json', 'mesh_quality.json'):
        record = read(case/name)
        if not record['passed']: raise RuntimeError('Rejected initial geometry/mesh: '+case.name)
        verify_case_files(case, record['hashes'])


def select_inputs(run, cfg, manifest, approval_path, ready):
    chosen = ready[:cfg['samples']]
    if len(chosen) != cfg['samples']:
        raise RuntimeError('Need %d prepared independent meshes, have %d' % (cfg['samples'], len(chosen)))
    for row in chosen: verify_input(run/'samples'/row['sample_id'], row, cfg)
    path = run/'batch_selection.json'
    scripts = {name:digest(ROOT/name) for name in RUNNER_FILES}
    receipt = dict(status='USER_SELECTED_DEFAULT', utc=now(), config_sha256=manifest['config_sha256'],
        manifest_sha256=digest(run/'manifest.json'), selected_default=str(approval_path),
        selected_default_sha256=digest(approval_path), count=len(chosen), samples=chosen,
        pipeline_hashes=manifest['pipeline_hashes'], runner_hashes=scripts,
        material_audit=audit_material(cfg), basis='User selected current simulation settings after the recorded trial',
        scope='Production solve authorization for the selected settings; retains actual numerical quality flags')
    if path.exists():
        old = read(path)
        for key in ('config_sha256', 'manifest_sha256', 'samples', 'runner_hashes', 'selected_default_sha256'):
            if old[key] != receipt[key]: raise RuntimeError('Selected batch changed: '+key)
        return old
    for name, sha in scripts.items():
        dest = run/'protocol/batch_runners'/sha/Path(name).name
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/name, dest)
    save(path, receipt)
    return receipt


def finish_pool(run, args):
    cfg, manifest, approval_path = check_run(run)
    ready, pending, exclusions = inventory(run, cfg, manifest, cache=not args.dry_run)
    target = cfg['samples']
    print(json.dumps(dict(phase='prepare', target=target, prepared_unique=len(ready),
        pending=len(pending), excluded=len(exclusions), dry_run=args.dry_run,
        time_limit_minutes=None if args.budget_minutes==float('inf') else args.budget_minutes), ensure_ascii=False), flush=True)
    if args.dry_run: return
    start = time.monotonic()
    def remaining():
        minutes = args.budget_minutes-(time.monotonic()-start)/60
        if minutes <= 0: raise TimeoutError('Preparation budget reached; rerun this command to continue')
        return minutes
    summary = read(run/'batch_prepare.json') if (run/'batch_prepare.json').exists() else dict(rejections=[])
    summary.update(status='PREPARING', utc=now(), target=target, prepared_unique=len(ready),
                   candidate_count=manifest['count'], runner_sha256=digest(__file__))
    save(run/'batch_prepare.json', summary)
    options = argparse.Namespace(workers=args.workers, prefetch=args.prefetch,
        geometry_threads=args.geometry_threads, with_datacheck=False, keep_intermediates=False,
        speed_mm_s=None, continue_on_error=True)
    with runtime.mesh_session(run, cfg):
        try:
            while len(ready) < target:
                remaining()
                if not pending:
                    count = min(args.replacement_batch, target-len(ready), args.max_candidates-manifest['count'])
                    if count < 1: raise RuntimeError('Candidate limit reached; inspect batch_prepare.json')
                    extend_run(run, count, 'automatic replacement to reach %d accepted independent meshes' % target)
                    manifest = read(run/'manifest.json')
                    ready, pending, exclusions = inventory(run, cfg, manifest, cache=True)
                rows = pending[:target-len(ready)]
                errors = prepare_parallel(run, cfg, manifest, rows, options, manifest['pipeline_hashes'], remaining)
                for error in errors:
                    case = run/'samples'/error['sample_id']
                    if not rejected(case):
                        raise RuntimeError('Preparation infrastructure/input error: '+str(error))
                    summary['rejections'].append(error)
                summary.update(utc=now(), prepared_unique=len(ready), candidate_count=manifest['count'])
                save(run/'batch_prepare.json', summary)
                ready, pending, exclusions = inventory(run, cfg, manifest, cache=True)
                summary.update(prepared_unique=len(ready), utc=now())
                save(run/'batch_prepare.json', summary)
                print('Prepared independent meshes: %d/%d' % (len(ready), target), flush=True)
            summary.update(status='READY', prepared_unique=len(ready), candidate_count=manifest['count'],
                           exclusions=exclusions, utc=now())
            select_inputs(run, cfg, manifest, approval_path, ready)
        except BaseException as exc:
            summary.update(status='INTERRUPTED', error=str(exc), utc=now()); raise
        finally: save(run/'batch_prepare.json', summary)
    print('READY: %d independent structures; FEA has not started.' % target, flush=True)


def verify_batch(run):
    cfg, manifest, approval_path = check_run(run)
    if not (run/'batch_selection.json').exists():
        raise RuntimeError('Finish the dataset_batch.py prepare command before starting FEA')
    receipt = read(run/'batch_selection.json')
    if (receipt['status'] != 'USER_SELECTED_DEFAULT' or receipt['config_sha256'] != manifest['config_sha256']
            or receipt['manifest_sha256'] != digest(run/'manifest.json')
            or receipt['selected_default_sha256'] != digest(approval_path)
            or receipt['count'] != cfg['samples'] or len(receipt['samples']) != cfg['samples']):
        raise RuntimeError('Frozen selected batch changed')
    verify_files(ROOT, receipt['runner_hashes'])
    rows = {r['sample_id']:r for r in manifest['samples']}
    if len({r['geometry_group'] for r in receipt['samples']}) != receipt['count']:
        raise RuntimeError('Selected batch contains duplicate geometry')
    for row in receipt['samples']:
        if row['design_hash'] != rows[row['sample_id']]['design_hash']:
            raise RuntimeError('Selected design changed')
    return cfg, receipt


def run_fea(run, args):
    if args.dry_run and not (run/'batch_selection.json').exists():
        cfg, manifest, _ = check_run(run)
        ready, _, _ = inventory(run, cfg, manifest)
        print(json.dumps(dict(phase='fea', prepared_unique=len(ready), required=cfg['samples'],
            selection_ready=False, solver_command=runtime.solver_command(cfg), dry_run=True))); return
    cfg, receipt = verify_batch(run)
    summary = read(run/'batch_fea.json') if (run/'batch_fea.json').exists() else dict(completed=[], failures=[])
    summary.setdefault('publication_failures',[])
    attempted = set(summary['completed']) | (set() if getattr(args,'retry_failed',False) else {r['sample_id'] for r in summary['failures']})
    todo = [r for r in receipt['samples'] if r['sample_id'] not in attempted][:args.limit]
    training_ids = {r['sample_id']:i for i,r in enumerate(receipt['samples'])}
    print(json.dumps(dict(phase='fea', selected=receipt['count'], completed=len(summary['completed']),
        failed=len(summary['failures']), pending=len(todo), cpus=cfg['solver']['cpus'], dry_run=args.dry_run,
        stage_time_limit_minutes=None)), flush=True)
    if args.dry_run: return
    runtime.recover_interrupted_run(run)
    # Reconcile prior exports on restart before scheduling another FEA job.
    # Per-case publication below copies only that case's result files.
    ensure_clean_dataset(run)
    summary.update(status='RUNNING', utc=now(), config_sha256=receipt['config_sha256'])
    try:
        for row in todo:
            case = run/'samples'/row['sample_id']; summary['current'] = case.name
            save(run/'batch_fea.json', summary)
            phase='solve'
            summary['failures']=[r for r in summary['failures'] if r['sample_id']!=case.name]
            summary['publication_failures']=[r for r in summary['publication_failures'] if r['sample_id']!=case.name]
            try:
                verify_input(case,row,cfg)
                effective=runtime.effective_solver_config(case,cfg)
                print(case.name+' FEA: %d CPUs, complete 5%% compression' % effective['solver']['cpus'],flush=True)
                if (case/'cleanup_receipt.json').exists():
                    if read(case/'cleanup_receipt.json')['diagnostic_only']:raise RuntimeError('Incomplete prior solver result: '+case.name)
                    verify_files(case,read(case/'export_manifest.json')['hashes'])
                with runtime.batch_execution(),SampleProgress(case, training_ids[case.name], receipt['count']):
                    solve_with_retry(run,case,cfg)
                extraction = read(case/'extraction.json')
                if not extraction['solver_completed']: raise RuntimeError('Incomplete solver export')
                verify_files(case, read(case/'export_manifest.json')['hashes'])
                phase='publication'
                sync_clean_result(run,case.name)
            except Exception as exc:
                if phase=='publication':
                    summary['publication_failures'].append(dict(sample_id=case.name,error=str(exc),utc=now()))
                    print('PUBLICATION FAILED '+case.name+': '+str(exc)+'; complete source result retained',flush=True)
                else:
                    try:audit=reject_failed_case(run,case,cfg,exc)
                    except Exception as recovery_error:raise RuntimeError('Cannot safely release failed case: '+str(exc)+'; '+str(recovery_error)) from recovery_error
                    summary['failures'].append(dict(sample_id=case.name,error=str(exc),utc=now(),audit=audit))
                    print('FAILED '+case.name+': '+str(exc)+'; continuing to next sample',flush=True)
            else:
                summary['completed'].append(case.name)
            summary.update(utc=now(), numerical_passed=sum(
                read(run/'samples'/sid/'quality.json')['passed'] for sid in summary['completed']))
            save(run/'batch_fea.json', summary)
        n = len(summary['completed'])+len(summary['failures'])+len(summary['publication_failures'])
        summary.update(status=('COMPLETED_WITH_FAILURES' if summary['failures'] or summary['publication_failures'] else 'COMPLETED')
                       if n == receipt['count'] else 'PAUSED_AFTER_LIMIT', current=None)
    except BaseException as exc:
        summary.update(status='INTERRUPTED', error=str(exc)); raise
    finally:
        summary['utc'] = now(); save(run/'batch_fea.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'all', 'status'))
    parser.add_argument('--run-id', default='design500_adaptive025_20261001')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--prefetch', type=int, default=2)
    parser.add_argument('--geometry-threads', type=int, default=4)
    parser.add_argument('--budget-minutes', type=float, default=float('inf'),
                        help='Optional preparation time budget; default is unlimited. FEA stages run to completion.')
    parser.add_argument('--replacement-batch', type=int, default=50)
    parser.add_argument('--max-candidates', type=int, default=2000)
    parser.add_argument('--limit', type=int, default=500)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--retry-failed', action='store_true', help='Retry rejected FEA cases on this invocation')
    args = parser.parse_args()
    if not (1 <= args.workers <= 4 and args.workers <= args.prefetch <= 8
            and args.geometry_threads > 0 and args.budget_minutes > 0
            and args.replacement_batch > 0 and args.max_candidates > 0 and args.limit > 0):
        parser.error('Invalid worker, prefetch, thread, budget or count setting')
    run = ROOT/'runs'/safe_id(args.run_id)
    if args.action == 'status':
        found = False
        for name in ('batch_prepare.json', 'batch_fea.json'):
            if (run/name).exists():
                found = True; record = read(run/name)
                print(json.dumps({name:{key:value for key,value in record.items()
                    if key not in ('rejections', 'exclusions', 'completed', 'failures')}}, ensure_ascii=False, indent=2))
                if name == 'batch_fea.json':
                    print(json.dumps(dict(completed=len(record['completed']), failed=len(record['failures']))))
                    if record.get('current'):
                        sta = run/'samples'/record['current']/'analysis.sta'
                        if sta.exists(): print('\n'.join(sta.read_text(errors='replace').splitlines()[-5:]))
                elif record['status'] == 'PREPARING' and (run/'parallel_prepare.json').exists():
                    progress = read(run/'parallel_prepare.json')
                    print(json.dumps(dict(current_prepare=progress.get('current'),
                        prepared_this_round=len(progress['prepared']), rejected_this_round=len(progress['failures']))))
        if not found:
            cfg, manifest, _ = check_run(run)
            stages = {}
            for row in manifest['samples']:
                stage = read(run/'samples'/row['sample_id']/'state.json')['stage']
                stages[stage] = stages.get(stage, 0)+1
            print(json.dumps(dict(target=cfg['samples'], candidates=manifest['count'], stages=stages)))
        return
    with lock(run/'.lock'):
        if args.action in ('prepare','all'):
            if not args.dry_run:
                if (run/'batch_selection.json').exists():
                    verify_batch(run)
                    runtime.recover_interrupted_run(run)
                runtime.assert_external_idle(); assert_generators_idle(run)
            finish_pool(run, args)
            if args.action=='all':run_fea(run,args)
        else: run_fea(run, args)


if __name__ == '__main__': main()
