"""Re-evaluate and independently validate fixed complete-chain search examples.

These are finite-search feasibility examples, not claimed optimal missions.
They include all parking, landing/launch and optional flyby maneuvers. No
network is used unless --horizons-refresh is explicitly supplied.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.research.mission_config import load_mission
from src.research.mission_dynamics import MissionEvaluator
from src.research.mission_search import fingerprint,write_json
from src.research.mission_validation import validate_candidate
from src.research.mission_plot import plot_candidate
from src.research.mission_horizons import crosscheck_candidate


def reproduce(outdir,backend='cpu',horizons=False):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=False)
    config=load_mission(ROOT/'examples/missions/emc-ten-year.json');config.backend=backend
    summaries=[]
    cases=json.loads((ROOT/'validation/mission-cases.json').read_text())['cases']
    for case in cases:
        c=deepcopy(config);c.model=case['model']
        folder=outdir/('validated-'+c.model);folder.mkdir()
        write_json(folder/'config.json',c.to_dict())
        with MissionEvaluator(c,folder/'state-cache') as evaluator:
            candidate=evaluator.candidate(case['decisions_days'])
            if candidate is None:raise AssertionError('Saved search point is no longer feasible')
            fp=fingerprint(c,evaluator);candidate['fingerprint']=fp
            candidate['derivation']=case
            for relative in fp['modules']:
                destination=folder/'sources'/relative;destination.parent.mkdir(parents=True,exist_ok=True)
                destination.write_bytes((ROOT/relative).read_bytes())
            write_json(folder/'best.json',candidate)
        report=validate_candidate(folder/'best.json')
        if report['status']!='passed_patched_conic':
            raise AssertionError([x for x in report['checks'] if not x['passed']])
        plot_candidate(folder/'best.json')
        if horizons:
            hr=crosscheck_candidate(folder/'best.json',True)
            if hr['status']!='consistent_ephemerides':raise AssertionError(hr)
        summary=dict(model=c.model,delta_v_m_s=candidate['total_delta_v_m_s'],duration_days=candidate['duration_days'],
            launch_c3_km2_s2=candidate['launch_C3_km2_s2'],status=report['status'],
            maximum_propagation_error_m=report['maximum_ode_position_error_m'],
            maximum_gooding_error_m_s=report['maximum_gooding_velocity_error_m_s'],
            candidate=str(folder/'best.json'))
        summaries.append(summary);print(json.dumps(summary),flush=True)
    write_json(outdir/'validated-summary.json',summaries)
    return summaries


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outdir',type=Path,default=Path('results/validated-missions'))
    p.add_argument('--backend',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--horizons-refresh',action='store_true')
    a=p.parse_args();reproduce(a.outdir,a.backend,a.horizons_refresh)
