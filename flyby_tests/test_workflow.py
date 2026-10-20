import contextlib
import io
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from src.research.flyby import load_request,run_encounters

ROOT=Path(__file__).resolve().parents[1]


class Workflow(unittest.TestCase):
    def test_inertial_transform_and_saved_batch(self):
        data=json.loads((ROOT/'examples/flyby-earth.json').read_text())
        body=np.array([30000.,-1200.,500.])
        data['velocity_origin']='inertial';data['body_velocity_m_s']=body.tolist()
        for row in data['encounters']:
            row['vin_m_s']=(np.array(row['vin_m_s'])+body).tolist()
            if 'vout_m_s' in row:row['vout_m_s']=(np.array(row['vout_m_s'])+body).tolist()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'request.json';path.write_text(json.dumps(data))
            _,rows,_,_=load_request(path)
            np.testing.assert_array_equal(rows['vin'],[[10000.,0.,0.]]*3)
            with contextlib.redirect_stdout(io.StringIO()): run,meta=run_encounters(path,d,'cpu',2)
            with np.load(run/'encounters.npz',allow_pickle=False) as f:
                self.assertTrue((f['results']['status']==0).all())
                np.testing.assert_allclose(f['vout_inertial_m_s'],f['results']['vout']+body)
            self.assertEqual(meta['request_sha256'],hashlib.sha256((run/'request.json').read_bytes()).hexdigest())
            self.assertEqual(meta['data_sha256'],hashlib.sha256((run/'encounters.npz').read_bytes()).hexdigest())
            self.assertEqual(meta['status_counts'],{'SUCCESS':3})
            self.assertTrue(meta['inertial_output_available'])
            self.assertTrue((run/'request.json').is_file())
            self.assertTrue((run/'metadata.json').is_file())

    def test_invalid_provenance_and_schema(self):
        original=json.loads((ROOT/'examples/flyby-earth.json').read_text())
        changes=[{'schema_version':2},{'schema_version':True},{'frame':'ITRF93'},
                 {'velocity_origin':'inertial'},{'mu_m3_s2':-1}, {'min_altitude_m':-1},
                 {'epoch_tdb':'2035-01-01T00:00:00+00:00'}, {'typo':1},
                 {'encounters':[]},{'body_velocity_m_s':[1,2]},{'radius_m':True}]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'request.json'
            for fields in changes:
                with self.subTest(fields=fields),self.assertRaises((ValueError,TypeError)):
                    path.write_text(json.dumps(dict(original,**fields)));load_request(path)

    @unittest.skipUnless(os.environ.get('LAMBERT_TEST_CUDA')=='1','explicit GPU opt-in')
    def test_cuda_saved_batch(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            path,meta=run_encounters(ROOT/'examples/flyby-earth.json',d,'cuda',2)
            self.assertEqual(meta['status_counts'],{'SUCCESS':3})
            self.assertGreater(meta['kernel_ms'],0.)
            with np.load(path/'encounters.npz') as f:self.assertTrue((f['results']['status']==0).all())
