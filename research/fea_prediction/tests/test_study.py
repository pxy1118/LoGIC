import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from test_pipeline import create_dataset
from surrogate.common import ROOT, read, save
from surrogate.data import snapshot, load_selected, load_split
from surrogate.models import BRANCHES, CurveNet, fit_preprocessing, control_statistics, tabular
from surrogate.paper_report import paired_interval, summarize
from surrogate.study import build_tasks, nested_order, run_study, verify_test_freeze
from surrogate import training


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.dataset=create_dataset(self.root)
        self.snap=self.root/'snapshot'
        snapshot(self.dataset,self.snap,'simulation',4)
        torch.set_num_threads(1)

    def test_branch_removal_and_old_state_compatibility(self):
        data=load_split(self.snap,'train')
        prep=fit_preprocessing(data,4)
        inputs=[torch.from_numpy(data[k][:2]) for k in ('controls','geometry','voxel')]
        for method,branches in BRANCHES.items():
            model=CurveNet(prep,method).eval()
            self.assertEqual(hasattr(model,'control_encoder'),'C' in branches)
            self.assertEqual(hasattr(model,'geometry_encoder'),'G' in branches)
            self.assertEqual(hasattr(model,'voxel_encoder'),'V' in branches)
            output=model(*inputs)
            changed=[x.clone() for x in inputs]
            for i,b in enumerate('CGV'):
                if b not in branches and not (b=='C' and 'S' in branches):
                    changed[i]=torch.full_like(changed[i],float('nan'))
            torch.testing.assert_close(output,model(*changed))
            self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(torch.count_nonzero(output[:,0]),0)
        # Historic checkpoints do not contain stats buffers or an explicit branch config.
        old_prep={k:v for k,v in prep.items() if not k.startswith('stats_')}
        for method in ('control_pod','fusion_pod','fusion_direct'):
            model=CurveNet(old_prep,method)
            restored=CurveNet(old_prep,method)
            restored.load_state_dict(model.state_dict(),strict=True)
        self.assertEqual(sum(p.numel() for p in CurveNet(dict(prep,rank=8,basis=np.zeros((8,1001),dtype='float32')),'fusion_pod').parameters()),52448)

    def test_stats_remove_spatial_order_but_keep_channels(self):
        x=np.arange(702,dtype='float32').reshape(2,351)
        permuted=x.reshape(2,27,13)[:,::-1].reshape(2,351)
        np.testing.assert_allclose(control_statistics(x),control_statistics(permuted))
        data=load_split(self.snap,'train');prep=fit_preprocessing(data,4)
        a=tabular(data,prep,'density_pod_krr')
        data['geometry'][:,1:]+=1000;data['controls'][:]+=1000
        np.testing.assert_array_equal(a,tabular(data,prep,'density_pod_krr'))

    def test_subset_statistics_and_test_guard(self):
        subset=load_selected(self.snap,[0,1,2])
        prep=fit_preprocessing(subset,2)
        np.testing.assert_allclose(prep['geometry_mean'],subset['geometry'].mean(0))
        with self.assertRaisesRegex(ValueError,'forbidden'):
            load_selected(self.snap,[10])
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            load_selected(self.snap,[0,0])
        with self.assertRaisesRegex(ValueError,'locked'):
            verify_test_freeze(self.root/'not_started')
        with self.assertRaisesRegex(ValueError,'original test'):
            training.train(self.snap,self.root/'leak',dict(selection=dict(train=[0,1,10],validation=[8,9]),device='cpu'))
        with self.assertRaisesRegex(ValueError,'Overlapping'):
            training.train(self.snap,self.root/'overlap',dict(selection=dict(train=[0,1],validation=[1,8]),device='cpu'))

    def test_full_matrix_and_nested_strata(self):
        cfg=read(ROOT/'configs/paper_fea.json')
        rows=[]
        for split,n,triple,boundary in [('train',304,34,26),('validation',67,9,6),('test',62,4,9)]:
            for i in range(n):
                rows.append(dict(sample_id=len(rows),split=split,geometry_group=str(len(rows)),
                                 topology_family=['Gyroid','Diamond','Primitive'] if i<triple else ['Gyroid'],
                                 kind='boundary' if i>=n-boundary else 'coverage'))
        tasks,order=build_tasks(dict(samples=rows),cfg)
        self.assertEqual(len(tasks),9)
        self.assertEqual(len(set(order)),304)
        self.assertEqual(order,nested_order(rows[:304],cfg['subset_seed']))
        learn=[t for t in tasks if t['kind']=='learning']
        self.assertTrue(set(learn[0]['selection']['train'])<set(learn[1]['selection']['train'])<set(learn[2]['selection']['train']))
        for name,counts in [('holdout_triple',(270,58,43)),('holdout_boundary',(278,61,32))]:
            task=next(t for t in tasks if t['name']==name)
            self.assertEqual(tuple(len(task['selection'][k]) for k in ('train','validation','holdout')),counts)
            for selected in task['selection'].values():
                self.assertFalse(set(selected)&set(range(371,433)))

    def test_bootstrap_units_and_missing_yield(self):
        result=paired_interval([2,4,6],[1,3,5],['a','b','c'],10000,17)
        self.assertEqual(result['ci95_MPa'],[1.,1.])
        data=load_split(self.snap,'validation')
        linear=np.tile(np.linspace(0,50,1001),(2,1))
        metrics,records=summarize(linear,linear,data['rows'],data['geometry'],read(self.snap/'config.json'))
        self.assertEqual(metrics['mean_curve_rmse_MPa'],0.)
        self.assertEqual(metrics['global_r2'],1.)
        self.assertIsNone(metrics['scalar_metrics']['yield_strength_MPa']['MAE'])
        self.assertEqual(metrics['scalar_metrics']['yield_strength_MPa']['both_valid'],0)
        self.assertEqual(records[0]['predicted_yield_status'],'not_observed_by_5pct')

    def test_tiny_study_end_to_end_resume_and_frozen_test(self):
        study=self.root/'study'
        cfg=read(ROOT/'configs/paper_fea.json')
        cfg.update(expected_split_counts=dict(train=8,validation=2,test=2),voxel_size=4,resolutions=[4],
                   learning_sizes=[8],methods=['mean','density_pod_krr','controls_only_pod','fusion_pod'],
                   seeds=[17],supplementary_seeds=[17],pod_diagnostic_ranks=[2])
        cfg['training'].update(epochs=1,patience=1,rank=2,krr_ranks=[2],device='cpu',threads=1)
        cfg['benchmark'].update(warmup=1,repetitions=2,devices=['cpu'])
        config=self.root/'study_config.json';save(config,cfg)
        def tiny_tasks(manifest,settings):
            return [dict(name='main',kind='main',voxel_size=4,methods=settings['methods'],seeds=[17])],list(range(8))
        # Exercise the complete execution and report pipeline on synthetic labels.
        with patch('surrogate.study.build_tasks',tiny_tasks):
            result=run_study(study,config,self.dataset,budget_minutes=.00001)
            self.assertEqual(result['status'],'PARTIAL_NOT_FINAL')
            self.assertFalse((study/'test_access.json').exists())
            with self.assertRaisesRegex(ValueError,'locked'):
                verify_test_freeze(study)
            with patch('surrogate.paper_report.figures') as render:
                self.assertEqual(run_study(study)['status'],'COMPLETE')
                self.assertTrue(render.called)
                self.assertEqual(run_study(study)['status'],'COMPLETE')
        self.assertTrue((study/'paper/manuscript_replacement.md').exists())
        self.assertTrue((study/'test_access.json').exists())
        verify_test_freeze(study)
        protocol=read(study/'protocol.json')
        protocol['config']['rank']=99;save(study/'protocol.json',protocol)
        with self.assertRaisesRegex(ValueError,'freeze'):
            verify_test_freeze(study)


if __name__=='__main__':
    unittest.main()
