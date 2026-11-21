import copy
from io import BytesIO
from pathlib import Path
import sys

import numpy as np
import unittest
import tempfile
from unittest.mock import patch
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from surrogate.common import digest, read, save
from surrogate.data import audit, encode_inputs, load_split, read_initial_sample, snapshot, validate_curve, revoxelize, cohort_signature
from surrogate.inference import derived, evaluate, predict
from surrogate.models import CurveNet, fit_preprocessing
from surrogate import training


def create_dataset(tmp_path):
    root=tmp_path/'live'
    config=dict(geometry=dict(size_mm=7.5),loading=dict(max_strain=.05),
                material=dict(density_tonne_mm3=4.5e-9),mesh={},solver={})
    save(root/'config.json',config)
    entries=[];rng=np.random.default_rng(7)
    for sid in range(13):
        p=root/'samples'/str(sid);p.mkdir(parents=True)
        split='train' if sid<8 else 'validation' if sid<10 else 'test'
        meta=dict(sample_id=sid,split=split,geometry_group=f'g{sid}',topology_family=['Gyroid'],kind='coverage')
        save(p/'sample.json',meta)
        rho=.25+.015*sid
        q=np.zeros((3,3,3,4));q[...,3]=1
        w=np.zeros((3,3,3,3));w[...,0]=1
        np.savez(p/'controls.npz',weights=w,density=np.full((3,3,3),rho),quaternion_xyzw=q,
                 topology_names=['Gyroid','Diamond','Primitive'])
        vox=rng.random((8,8,8))<rho
        np.savez(p/'voxel.npz',voxel=vox,spacing_mm=np.full(3,7.5/8),origin_mm=np.full(3,7.5/16))
        save(p/'features.json',dict(relative_density=rho,voxel_fraction=float(vox.mean()),surface_area_mm2=300+sid,
                                    z_area_mm2=[rho*7.5**2]*8,minimum_voxel_section_mm2=rho*7.5**2))
        strain=np.linspace(0,.05,1001)
        stress=np.minimum(strain*(1000+sid*80),12+sid)
        np.savez(p/'curve.npz',strain=strain,stress_MPa=stress)
        save(p/'metrics.json',derived(stress,rho,config))
        save(p/'quality.json',dict(passed=False,failures=['maximum_mass_change_percent']))
        entries.append(dict(sample_id=sid,path='samples/'+str(sid),split=split,labels_ready=sid!=12,
                            hashes={f.name:digest(f) for f in p.iterdir()}))
    save(root/'manifest.json',dict(sample_count=13,samples=entries))
    return root


def create_snapshot(dataset,tmp_path):
    output=tmp_path/'snapshot'
    snapshot(dataset,output,'simulation',4)
    return output


class PipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp_path = Path(temporary.name)
        self.dataset = create_dataset(self.tmp_path)
        self.frozen = create_snapshot(self.dataset,self.tmp_path)
        torch.set_num_threads(1)

    def patch_object(self, obj, name, value):
        patcher=patch.object(obj,name,value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_published_only_and_strict_quality(self):
        dataset, tmp_path, frozen = self.dataset, self.tmp_path, self.frozen
        initial={str(p):digest(p) for p in dataset.rglob('*') if p.is_file()}
        assert audit(dataset)['complete_curves']==12
        assert audit(dataset)['numerical_passed']==0
        with self.assertRaisesRegex(ValueError,'No eligible'):
            snapshot(dataset,tmp_path/'strict','strict',4)
        manifest=snapshot(dataset,tmp_path/'simulation','simulation',4)
        assert manifest['split_counts']==dict(train=8,validation=2,test=2)
        assert manifest['scope']=='SIMULATION_PROTOCOL_SURROGATE'
        assert 12 not in [s['sample_id'] for s in manifest['samples']]
        assert initial=={str(p):digest(p) for p in dataset.rglob('*') if p.is_file()}


    def test_changed_published_file_is_rejected(self):
        dataset, tmp_path, frozen = self.dataset, self.tmp_path, self.frozen
        (dataset/'samples/0/quality.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            snapshot(dataset,tmp_path/'bad','simulation',4)


    def test_cross_split_group_rejected(self):
        dataset, tmp_path, frozen = self.dataset, self.tmp_path, self.frozen
        p=dataset/'samples/8/sample.json';meta=read(p);meta['geometry_group']='g0';save(p,meta)
        m=read(dataset/'manifest.json');m['samples'][8]['hashes']['sample.json']=digest(p);save(dataset/'manifest.json',m)
        with self.assertRaisesRegex(ValueError,'Duplicate geometry'):
            audit(dataset)


    def test_no_truncated_curve_padding(self):
        dataset, tmp_path, frozen = self.dataset, self.tmp_path, self.frozen
        with self.assertRaisesRegex(ValueError,'1001-point'):
            validate_curve(dict(strain=np.linspace(0,.04,801),stress_MPa=np.ones(801)),{'valid':{'curve':True}})


    def test_snapshot_tampering_rejected(self):
        dataset, tmp_path, frozen = self.dataset, self.tmp_path, self.frozen
        p=frozen/'samples/0/data.npz';p.write_bytes(p.read_bytes()+b'x')
        with self.assertRaisesRegex(ValueError,'Modified frozen sample'):
            load_split(frozen,'train')


    def test_quaternion_sign_and_fractional_voxels(self):
        dataset, tmp_path, frozen = self.dataset, self.tmp_path, self.frozen
        path=dataset/'samples/0';cfg=read(dataset/'config.json')
        original=read_initial_sample(path,cfg,4)
        blobs={n:(path/n).read_bytes() for n in ('controls.npz','features.json','voxel.npz')}
        with np.load(BytesIO(blobs['controls.npz'])) as a:
            c={k:a[k].copy() for k in a.files}
        c['quaternion_xyzw']*=-1
        b=BytesIO();np.savez(b,**c);blobs['controls.npz']=b.getvalue()
        encoded=encode_inputs(blobs,cfg,4)
        np.testing.assert_array_equal(original[0],encoded[0])
        with np.load(path/'voxel.npz') as a:
            assert np.isclose(original[2].mean(),a['voxel'].mean())
        assert np.any((original[2]>0)&(original[2]<1))


    def test_train_statistics_and_hard_origin(self):
        dataset, tmp_path, frozen = self.dataset, self.tmp_path, self.frozen
        tr=load_split(frozen,'train');va=load_split(frozen,'validation')
        before=fit_preprocessing(tr,4)
        va['stress'][:]=1e20;va['controls'][:]=1e20
        after=fit_preprocessing(tr,4)
        for k in before:
            np.testing.assert_array_equal(before[k],after[k])
        for method in ('fusion_pod','control_pod','fusion_direct'):
            model=CurveNet(before,method).eval()
            pred=model(*(torch.from_numpy(tr[k][:2]) for k in ('controls','geometry','voxel')))
            assert pred.shape==(2,1001)
            assert torch.count_nonzero(pred[:,0])==0


    def test_metrics_linear_unobserved_yield(self):
        dataset, tmp_path, frozen = self.dataset, self.tmp_path, self.frozen
        strain=np.linspace(0,.05,1001)
        cfg=dict(geometry=dict(size_mm=7.5),loading=dict(max_strain=.05),material=dict(density_tonne_mm3=4.5e-9))
        m=derived(1000*strain,.4,cfg)
        assert np.isclose(m['elastic_modulus_MPa'],1000)
        assert m['yield_strength_MPa'] is None
        assert m['yield_status']=='not_observed_by_5pct'
        assert np.isclose(m['W5_MJ_m3'],1.25)
        assert np.isclose(m['SEA_J_g'],1.25/(.4*4.5))


    def test_resume_and_inference_without_labels(self):
        dataset, tmp_path, frozen = self.dataset, self.tmp_path, self.frozen
        cfg=dict(methods=['control_pod'],seeds=[17],epochs=3,patience=5,device='cpu',threads=1,batch_size=4,rank=4)
        original_load=training.load_split
        def guard(root,split):
            assert split!='test'
            return original_load(root,split)
        self.patch_object(training,'load_split',guard)
        reference=tmp_path/'full';training.train(frozen,reference,cfg)
        interrupted=tmp_path/'interrupted';original_save=training.atomic_checkpoint
        def interrupt_after_first(path,data):
            original_save(path,data)
            if path.name=='last.pt' and data['epoch']==1:
                raise KeyboardInterrupt
        self.patch_object(training,'atomic_checkpoint',interrupt_after_first)
        assert training.train(frozen,interrupted,cfg)['status']=='PARTIAL_NOT_FINAL'
        self.patch_object(training,'atomic_checkpoint',original_save)
        assert training.train(None,interrupted,resume=True)['status']=='TRAINED'
        a=torch.load(reference/'control_pod_seed17/best.pt',weights_only=False)
        b=torch.load(interrupted/'control_pod_seed17/best.pt',weights_only=False)
        for k in a['state_dict']:
            torch.testing.assert_close(a['state_dict'][k],b['state_dict'][k],rtol=0,atol=0)
        import shutil
        unlabeled=tmp_path/'new_structure';unlabeled.mkdir()
        for name in ('controls.npz','features.json','voxel.npz'):
            shutil.copy2(dataset/'samples/0'/name,unlabeled/name)
        result=predict(reference,unlabeled,tmp_path/'prediction',input_config=dataset/'config.json')
        assert result['scope']=='SIMULATION_PROTOCOL_SURROGATE'
        assert (tmp_path/'prediction/curve.csv').exists()
        # This is an artificial fixture; actual held-out test data are not accessed by acceptance tests.
        report=evaluate(reference,'test',output=tmp_path/'fixture_evaluation')
        assert report['n']==2


    def test_smoke_test_set_guard(self):
        dataset, tmp_path, frozen = self.dataset, self.tmp_path, self.frozen
        run=tmp_path/'smoke'
        training.train(frozen,run,dict(methods=['mean'],device='cpu'),smoke=True)
        with self.assertRaisesRegex(ValueError,'Smoke runs cannot'):
            evaluate(run,'test')

    def test_matched_snapshot_ignores_new_completions_and_preserves_labels(self):
        original = read(self.frozen/'manifest.json')
        before = {str(p):digest(p) for p in self.frozen.rglob('*') if p.is_file()}
        live = read(self.dataset/'manifest.json')
        live['samples'][12]['labels_ready'] = True
        save(self.dataset/'manifest.json', live)
        newer = self.tmp_path/'resolution8'
        changed = revoxelize(self.frozen, newer, 8)
        self.assertEqual(cohort_signature(original), cohort_signature(changed))
        self.assertEqual(len(changed['samples']), 12)
        self.assertEqual(changed['split_counts'], original['split_counts'])
        for split in ('train','validation','test'):
            old, new = load_split(self.frozen, split), load_split(newer, split)
            for k in ('controls','geometry','stress'):
                np.testing.assert_array_equal(old[k],new[k])
            n=len(new['voxel'])
            recoarsened=new['voxel'].reshape(n,1,4,2,4,2,4,2).mean((3,5,7))
            np.testing.assert_allclose(recoarsened, old['voxel'], rtol=0, atol=1e-7)
        self.assertEqual(before, {str(p):digest(p) for p in self.frozen.rglob('*') if p.is_file()})

    def test_revoxelize_rejects_changed_original_geometry(self):
        raw=self.dataset/'samples/0/voxel.npz'
        raw.write_bytes(raw.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            revoxelize(self.frozen,self.tmp_path/'bad_resolution',8)

    def test_snapshot_cli_defaults_to_80(self):
        import main
        from contextlib import redirect_stdout
        from io import StringIO
        with patch('sys.argv',['main.py','snapshot','--snapshot-id','test_default']), \
             patch('surrogate.data.snapshot',return_value={}) as call, redirect_stdout(StringIO()):
            main.main()
        self.assertEqual(call.call_args.args[-1],80)
        self.assertEqual(call.call_args.args[-2],'strict')
        with patch('sys.argv',['main.py','snapshot','--snapshot-id','test_default','--match-snapshot-id','old']), \
             patch('surrogate.data.revoxelize',return_value={}) as call, redirect_stdout(StringIO()):
            main.main()
        self.assertEqual(call.call_args.args[-1],80)

    def test_comparison_rejects_unmatched_protocol(self):
        from surrogate.resolution import validate_runs
        high=self.tmp_path/'resolution8';revoxelize(self.frozen,high,8)
        runs=[]
        for n,snap in [(4,self.frozen),(8,high)]:
            run=self.tmp_path/f'run{n}';runs.append(run)
            plan=dict(snapshot=str(snap),snapshot_sha256=digest(snap/'manifest.json'),voxel_size=n,
                      config=dict(methods=['fusion_pod'],batch_size=12),versions={},code_hashes={},smoke=True)
            save(run/'plan.json',plan);save(run/'status.json',dict(status='SMOKE_PASSED'))
        validate_runs(runs,'fusion_pod')
        plan=read(runs[1]/'plan.json');plan['config']['batch_size']=6;save(runs[1]/'plan.json',plan)
        with self.assertRaisesRegex(ValueError,'Unmatched training'):
            validate_runs(runs,'fusion_pod')
        plan['config']['batch_size']=12
        m=read(high/'manifest.json');m['samples'][0]['split']='validation';save(high/'manifest.json',m)
        plan['snapshot_sha256']=digest(high/'manifest.json');save(runs[1]/'plan.json',plan)
        with self.assertRaisesRegex(ValueError,'Different cohorts'):
            validate_runs(runs,'fusion_pod')

    def test_high_resolution_forward_preserves_output_shape(self):
        tr=load_split(self.frozen,'train');prep=fit_preprocessing(tr,4)
        for method in ('fusion_pod','fusion_direct'):
            model=CurveNet(prep,method).eval()
            for n in (40,80,120):
                with torch.inference_mode():
                    y=model(torch.from_numpy(tr['controls'][:1]),torch.from_numpy(tr['geometry'][:1]),
                            torch.zeros(1,1,n,n,n))
                self.assertEqual(tuple(y.shape),(1,1001))
                self.assertEqual(float(y[0,0]),0.)

if __name__=="__main__":
    unittest.main()
