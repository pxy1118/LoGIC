"""FEA prediction CLI. Never modifies or invokes the live FEA pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from surrogate.common import DEFAULT_DATASET, named_path, read, save
from surrogate.data import DEFAULT_VOXEL_SIZE


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    study=sub.add_parser('study')
    study_sub=study.add_subparsers(dest='study_command',required=True)
    for command in ('run','status','report'):
        sp=study_sub.add_parser(command);sp.add_argument('--study-id',required=True)
        if command=='run':
            sp.add_argument('--config',type=Path);sp.add_argument('--dataset',type=Path)
            sp.add_argument('--budget-minutes',type=float)
    a=sub.add_parser('audit');a.add_argument('--dataset',type=Path,default=DEFAULT_DATASET);a.add_argument('--output',type=Path)
    s=sub.add_parser('snapshot');s.add_argument('--dataset',type=Path)
    s.add_argument('--snapshot-id',required=True);s.add_argument('--quality',choices=['strict','simulation'])
    s.add_argument('--voxel-size',type=int,default=DEFAULT_VOXEL_SIZE)
    s.add_argument('--match-snapshot-id',help='Freeze the same cohort/labels as this snapshot; rebuild voxels from original inputs')
    t=sub.add_parser('train');t.add_argument('--snapshot-id');t.add_argument('--run-id',required=True)
    t.add_argument('--config',type=Path);t.add_argument('--resume',action='store_true');t.add_argument('--smoke',action='store_true')
    t.add_argument('--budget-minutes',type=float);t.add_argument('--device',choices=['cpu','cuda'])
    t.add_argument('--epochs',type=int);t.add_argument('--methods',nargs='+')
    st=sub.add_parser('status');st.add_argument('--run-id',required=True)
    e=sub.add_parser('evaluate');e.add_argument('--run-id',required=True);e.add_argument('--split',choices=['validation','test'],default='validation')
    e.add_argument('--method');e.add_argument('--device',choices=['cpu','cuda'],default='cpu');e.add_argument('--output',type=Path)
    pr=sub.add_parser('predict');pr.add_argument('--run-id',required=True);pr.add_argument('--sample-dir',type=Path,required=True)
    pr.add_argument('--output',type=Path,required=True);pr.add_argument('--method');pr.add_argument('--input-config',type=Path)
    pr.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    comp=sub.add_parser('compare-resolutions')
    comp.add_argument('--run-ids',nargs='+',required=True);comp.add_argument('--output',type=Path,required=True)
    comp.add_argument('--method',choices=['fusion_pod','fusion_direct'],default='fusion_pod');comp.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=p.parse_args()
    if args.command=='study':
        from surrogate.study import run_study, status
        root=named_path('studies',args.study_id)
        if args.study_command=='run':
            result=run_study(root,args.config,args.dataset,args.budget_minutes)
        elif args.study_command=='status':
            result=status(root)
        else:
            from surrogate.paper_report import report_existing
            from surrogate.training import run_lock
            with run_lock(root):
                result=report_existing(root)
    elif args.command=='audit':
        from surrogate.data import audit
        result=audit(args.dataset)
        if args.output:
            from surrogate.inference import writable_output
            save(writable_output(args.output),result)
    elif args.command=='snapshot':
        from surrogate.data import snapshot, revoxelize
        if args.match_snapshot_id:
            if args.dataset or args.quality:
                p.error('--match-snapshot-id inherits the dataset and quality policy; omit --dataset and --quality')
            result=revoxelize(named_path('snapshots',args.match_snapshot_id),named_path('snapshots',args.snapshot_id),args.voxel_size)
        else:
            result=snapshot(args.dataset or DEFAULT_DATASET,named_path('snapshots',args.snapshot_id),args.quality or 'strict',args.voxel_size)
        result={k:v for k,v in result.items() if k not in ('samples','excluded')}
    elif args.command=='train':
        from surrogate.training import train
        if args.resume and (args.snapshot_id or args.config or args.device or args.epochs or args.methods or args.smoke):
            p.error('--resume uses the frozen run settings; only budget may change')
        if not args.resume and not args.snapshot_id:
            p.error('New training requires --snapshot-id')
        config=read(args.config) if args.config else {}
        for key in ('device','epochs','methods'):
            if getattr(args,key) is not None:
                config[key]=getattr(args,key)
        result=train(named_path('snapshots',args.snapshot_id) if args.snapshot_id else None,
                     named_path('runs',args.run_id),config or None,args.resume,args.budget_minutes,args.smoke)
    elif args.command=='status':
        result=read(named_path('runs',args.run_id)/'status.json')
    elif args.command=='predict':
        from surrogate.inference import predict
        result=predict(named_path('runs',args.run_id),args.sample_dir,args.output,args.method,args.device,args.input_config)
    elif args.command=='compare-resolutions':
        from surrogate.resolution import compare
        result=compare([named_path('runs',r) for r in args.run_ids],args.output,args.method,args.device)
    else:
        from surrogate.inference import evaluate
        result=evaluate(named_path('runs',args.run_id),args.split,args.method,args.device,args.output)
    print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':
    main()
