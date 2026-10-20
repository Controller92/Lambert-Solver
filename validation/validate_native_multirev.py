"""Apply the independent M>0 validation corpus directly to native results.

No Python roots or velocities feed the native solve. Adapters only translate
native statuses/diagnostics and provide the geometric transfer angle used by
independent propagation checks. --gpu explicitly executes the CUDA kernel.
"""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.lambert_geometry import prepare_geometry, LambertInput
from src.lambert_multirev import MinimumResult, MinimumStatus, MinimumOptions
from src.lambert_multirev_root import MultiRootResult, MultiRootStatus, RootBranch
from src.lambert_solver import LambertStatus
from src.lambert_root import RootOptions
from src.lambert_native import NativeEngine, pack_multi_rev_pairs
import validate_multirev_solver as checks
import validate_juno
import validate_multirev as preparation
from src.lambert_tof import TofResult, TofStatus
from src.lambert_root_preparation import DerivativeResult


ROOT_NAMES = dict(time_nondim='time', householder_steps='householder', bisection_steps='bisection',
                  verification_steps='verification', rejected_householder_steps='rejected_householder',
                  bracket_expansions='expansions', tof_evaluations='evaluations')
MIN_NAMES = dict(x_min='x', time_nondim='time', halley_steps='halley', bisection_steps='bisection',
                 zero_derivative_steps='zero_derivative', bracket_expansions='expansions', tof_evaluations='evaluations')


def root_result(row):
    r, minimum = row['root'], row['minimum']
    mr = MinimumResult(status=MinimumStatus(int(minimum['status'])), revolutions=int(row['revolutions']))
    for name in ('x_min','time_nondim','estimated_time_error','first','lower','upper','first_lower','first_upper',
                 'iterations','halley_steps','bisection_steps','zero_derivative_steps','bracket_expansions','tof_evaluations'):
        value = minimum[MIN_NAMES.get(name,name)].item()
        setattr(mr, name, None if isinstance(value,float) and math.isnan(value) else value)
    result = MultiRootResult(status=MultiRootStatus(int(r['status'])), minimum=mr,
                             branch=RootBranch.LEFT if row['branch']==0 else RootBranch.RIGHT,
                             revolutions=int(row['revolutions']))
    for name in ('x','y','last_x','time_nondim','residual','relative_residual','last_step','lower','upper',
                 'estimated_x_error','iterations','householder_steps','bisection_steps','verification_steps',
                 'rejected_householder_steps','derivative_failures','bracket_expansions','tof_evaluations'):
        value=r[ROOT_NAMES.get(name,name)].item()
        setattr(result,name,None if isinstance(value,float) and math.isnan(value) else value)
    result.x_precision_limited = None if r['precision_limited']==-1 else bool(r['precision_limited'])
    return result


class Adapter:
    def __init__(self, engine):
        self.engine = engine
        self.batch = engine.cuda_batch(2) if engine.backend == 'cuda' else None

    def roots(self,target,lam,m,options=RootOptions(),*,q=None,minimum_options=MinimumOptions()):
        rows=self.engine.probe('multi_roots',[[target,lam,math.nan if q is None else q,m,b] for b in (0,1)],
                               options,minimum_options=minimum_options)
        return tuple(root_result(row) for row in rows)

    def pair(self,problem,options=RootOptions(),*,minimum_options=MinimumOptions()):
        rows=pack_multi_rev_pairs([problem])
        if self.batch:
            self.batch.upload(rows);self.batch.run(options,minimum_options=minimum_options)
            output=self.batch.download()
        else:
            output=self.engine.cpu(rows,options,minimum_options=minimum_options)
        geometry=prepare_geometry(replace(problem,revolutions=0))
        return tuple(SimpleNamespace(status=LambertStatus(int(row['status'])),geometry=geometry,
                    root=root_result(row),branch=RootBranch.LEFT if row['branch']==0 else RootBranch.RIGHT,
                    v1_m_s=tuple(row['v1']) if row['status']==0 else None,
                    v2_m_s=tuple(row['v2']) if row['status']==0 else None) for row in output)

    def close(self):
        if self.batch: self.batch.close()


def roundoff_problems():
    for c in json.loads((ROOT/'validation/data/multirev-roundoff.json').read_text()):
        yield LambertInput(tuple(c['r1']),tuple(c['r2']),c['tof'],c['mu'],c['long_way'],c['revolutions'])


def validate_preparation(adapter):
    def forward(x,lam,m,*,q=None):
        r=adapter.engine.probe('multi_tof',[[x,lam,math.nan if q is None else q,m,0]])[0]
        return TofResult(TofStatus(int(r['status'])),float(r['time']),float(r['y']))
    def derivative(x,lam,m,*,q=None):
        r=adapter.engine.probe('multi_derivatives',[[x,lam,math.nan if q is None else q,m,0]])[0]
        return DerivativeResult(TofStatus(int(r['status'])),*(float(r[n]) for n in ('time','y','first','second','third')))
    def minimum(lam,m,*,q=None,options=MinimumOptions()):
        return adapter.roots(1.,lam,m,q=q,minimum_options=options)[0].minimum
    preparation.evaluate_tof_multi_rev=forward
    preparation.evaluate_derivatives_multi_rev=derivative
    preparation.find_minimum_multi_rev=minimum
    return preparation.validate()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',action='store_true')
    parser.add_argument('--section',choices=('all','preparation','scalar','conics','actual','conditioning','juno'),default='all')
    parser.add_argument('--json',type=Path)
    args=parser.parse_args()
    original_hard=checks.hard_problems
    def hard_with_regressions():
        yield from original_hard()
        yield from roundoff_problems()
    checks.hard_problems=hard_with_regressions
    result={}
    for backend in (('cpu','cuda') if args.gpu else ('cpu',)):
        adapter=Adapter(NativeEngine(backend))
        checks.solve_x_multi_rev_pair=adapter.roots
        checks.solve_multi_rev_pair=adapter.pair
        validate_juno.solve_multi_rev_pair=adapter.pair
        functions=dict(preparation=lambda:validate_preparation(adapter),scalar=checks.validate_scalar,conics=checks.validate_conics,
                       actual=checks.validate_actual_inputs,conditioning=checks.validate_conditioning)
        try:
            result[backend]={}
            if backend=='cuda': result[backend]['device']=adapter.engine.device_info(multi=True)
            for name,check in functions.items():
                # Scalar validation probes are host functions, even in CUDA .so.
                if backend=='cuda' and name in ('scalar','preparation'):continue
                if args.section in ('all',name):
                    print(backend,name,flush=True)
                    result[backend][name]=check()
            if args.section in ('all','juno'):
                result[backend]['juno']=validate_juno.validate()
        finally: adapter.close()
    if args.json: args.json.write_text(json.dumps(result,indent=2)+'\n')
    print('All requested native multi-revolution checks passed. CUDA '+('was executed.' if args.gpu else 'not requested.'),flush=True)


if __name__=='__main__':main()
