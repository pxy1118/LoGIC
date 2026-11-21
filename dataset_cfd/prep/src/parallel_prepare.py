"""Bounded geometry producers feeding one mesh/input consumer."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import time
import psutil
from .common import ROOT,read,save,digest,verify_files,object_hash,lock,now
from . import runtime


_THREAD_LIMITER=None


def initialize_generator(threads):
    global _THREAD_LIMITER
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VTK_SMP_MAX_THREADS'):
        os.environ[name]=str(threads)
    from threadpoolctl import threadpool_limits
    _THREAD_LIMITER=threadpool_limits(limits=threads)


def assert_generators_idle(run):
    for path in (run/'samples').glob('*/geometry_process.json'):
        record=read(path)
        if record.get('status')!='RUNNING':continue
        try:
            process=psutil.Process(record['pid'])
            if abs(process.create_time()-record['created'])<.01 and process.is_running():
                raise RuntimeError('Geometry worker is still active in '+str(path.parent))
        except psutil.NoSuchProcess:pass


def generate_one(case_path,cfg,source_path,generator_hashes,declared_hash):
    from .design import generate
    case=Path(case_path);source=Path(source_path)
    design=read(case/'design.json');declared=design.pop('design_hash')
    if declared!=declared_hash or object_hash(design)!=declared:
        raise RuntimeError('Design controls changed')
    verify_files(source,generator_hashes)
    with lock(case/'.geometry.lock'):
        started=time.monotonic();process=psutil.Process()
        record=dict(pid=process.pid,created=process.create_time(),started_utc=now(),status='RUNNING')
        save(case/'geometry_process.json',record)
        try:
            result=generate(case,cfg,source)
            record['status']='COMPLETED'
            return dict(sample=case.name,elapsed_s=time.monotonic()-started,passed=result['passed'])
        except BaseException as exc:
            record.update(status='FAILED',error=str(exc))
            save(case/'last_error.json',dict(action='generate',error=str(exc),utc=now()))
            raise
        finally:
            record.update(finished_utc=now(),elapsed_s=time.monotonic()-started)
            save(case/'geometry_process.json',record)


def prepare_parallel(run,cfg,manifest,rows,args,code_hashes,remaining):
    from threadpoolctl import threadpool_limits
    from .mesh import validate_mesh
    from .deck import write_deck
    from .storage import compact_case
    runtime.assert_external_idle();assert_generators_idle(run)
    source=run/'protocol/source';failures=[];prepared=[];pending=deque();iterator=iter(rows)
    summary_path=run/'parallel_prepare.json'
    summary=dict(started_utc=now(),status='RUNNING',workers=args.workers,prefetch=args.prefetch,
                 geometry_threads=args.geometry_threads,meshing_instances=1,datacheck_requested=args.with_datacheck,
                 pending=[],prepared=prepared,failures=failures)
    save(summary_path,summary)
    executor=ProcessPoolExecutor(max_workers=args.workers,initializer=initialize_generator,initargs=(args.geometry_threads,))
    def submit_next():
        row=next(iterator,None)
        if row is None:return False
        remaining()
        case=run/'samples'/row['sample_id']
        save(case/'stage_code/prepare_parallel.json',dict(utc=now(),hashes=code_hashes,
             workers=args.workers,geometry_threads=args.geometry_threads))
        future=executor.submit(generate_one,str(case),cfg,str(source),manifest['generator_hashes'],row['design_hash'])
        pending.append((row,future))
        summary['pending']=[r['sample_id'] for r,_ in pending];save(summary_path,summary)
        return True
    try:
        with threadpool_limits(limits=args.geometry_threads):
            for _ in range(args.prefetch):
                if not submit_next():break
            while pending:
                row,future=pending.popleft();case=run/'samples'/row['sample_id']
                summary.update(current=row['sample_id'],pending=[r['sample_id'] for r,_ in pending])
                save(summary_path,summary)
                print(row['sample_id']+' prepare parallel',flush=True)
                try:
                    wait_minutes=remaining()
                    # Windows cannot convert an infinite timeout to a lock wait.
                    future.result(timeout=None if wait_minutes==float('inf') else wait_minutes*60)
                    # Refill before the consumer starts: new geometry overlaps this mesh job.
                    submit_next()
                    runtime.mesh_case(case,cfg,remaining());validate_mesh(case,cfg)
                    if manifest['purpose']=='dataset' and not args.keep_intermediates:compact_case(case)
                    deck=write_deck(case,cfg,args.speed_mm_s if args.speed_mm_s is not None else cfg['loading']['mean_speed_mm_s'])
                    if args.with_datacheck:runtime.datacheck(case,cfg,remaining())
                    save(case/'preparation.json',dict(utc=now(),input_sha256=deck['input_sha256'],
                         datacheck_requested=args.with_datacheck,mode='parallel',
                         scope='Geometry, mesh quality and input preparation; datacheck is optional'))
                    if manifest['purpose']=='dataset' and not args.keep_intermediates:compact_case(case)
                    prepared.append(row['sample_id'])
                except Exception as exc:
                    save(case/'last_error.json',dict(action='prepare_parallel',error=str(exc),utc=now()))
                    failures.append(dict(sample_id=row['sample_id'],error=str(exc)))
                    if not isinstance(exc,TimeoutError) and manifest['purpose']=='dataset' and not args.keep_intermediates:
                        try:compact_case(case)
                        except Exception as cleanup_error:save(case/'storage_error.json',dict(error=str(cleanup_error),utc=now()))
                    print('REJECTED '+row['sample_id']+': '+str(exc),flush=True)
                    if not args.continue_on_error or isinstance(exc,TimeoutError) or runtime.active_external():raise
                    # Generation failure occurs before refill; keep the queue supplied.
                    while len(pending)<args.prefetch:
                        if not submit_next():break
                summary['pending']=[r['sample_id'] for r,_ in pending];save(summary_path,summary)
        summary['status']='COMPLETED_WITH_REJECTIONS' if failures else 'COMPLETED'
    except BaseException as exc:
        summary.update(status='INTERRUPTED',error=str(exc));raise
    finally:
        for _,future in pending:future.cancel()
        print('Finishing active geometry workers; completed geometry is retained for resume',flush=True)
        executor.shutdown(wait=True,cancel_futures=True)
        summary.update(finished_utc=now(),pending=[r['sample_id'] for r,_ in pending])
        save(summary_path,summary)
    return failures
