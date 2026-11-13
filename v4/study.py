"""Predeclared ablation then validation-only preprocessing selection and confirmation."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import sys


def worker(job):
    from experiment import run
    return run(**job)


def dispatch(jobs, out, workers):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256((Path(__file__).parent/'experiment.py').read_bytes()).hexdigest()
    path = out/'study_manifest.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['source_sha256'] != signature or previous['jobs'] != jobs:
            raise ValueError('Existing study uses different settings/source')
    status = dict(source_sha256=signature, jobs=jobs, completed=[], errors=[], state='running')
    todo = []
    for job in jobs:
        result_file = Path(job['output'])/job['name']/('seed_'+str(job['seed']))/'result.json'
        if result_file.exists():
            result = json.loads(result_file.read_text())
            if result.get('source_sha256', signature) != signature:
                raise ValueError('Result source mismatch')
            status['completed'].append(dict(name=job['name'], seed=job['seed'], output=job['output']))
        else:
            todo.append(job)
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(worker, job): job for job in todo}
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
                status['completed'].append(dict(name=job['name'], seed=job['seed'], output=job['output']))
                print('DONE {}/{} {} seed={}'.format(len(status['completed']), len(jobs), job['name'], job['seed']), flush=True)
            except Exception as error:
                status['errors'].append(dict(job=job, error=repr(error)))
                print('FAILED {}'.format(repr(error)), flush=True)
            status['elapsed_seconds'] = time.time()-started
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps(status, indent=2))
            temp.replace(path)
    status['state'] = 'complete' if not status['errors'] else 'failed'
    path.write_text(json.dumps(status, indent=2))
    if status['errors']:
        raise RuntimeError('Study jobs failed; inspect '+str(path))


def pilot(out, workers):
    jobs = []
    for mode in ['none', 'meanstd']:
        for seed in range(5):
            jobs.append(dict(name='competition_random', seed=seed, normalization=mode,
                reward_mode='total', output=str(Path(out)/('pilot_'+mode)), min_steps=512000,
                max_steps=512000, check_every=128000, eval_steps=32000,
                validation_only=True, fixed_budget=True))
    dispatch(jobs, Path(out)/'pilot', workers)
    scores = {}
    for mode in ['none', 'meanstd']:
        vals = []
        for seed in range(5):
            path = Path(out)/('pilot_'+mode)/'competition_random'/('seed_'+str(seed))/'result.json'
            vals.append(json.loads(path.read_text())['result']['validation_selection_score'])
        scores[mode] = dict(seed_scores=vals, mean=sum(vals)/len(vals))
    selected = max(scores, key=lambda mode: scores[mode]['mean'])
    decision = dict(rule='Choose larger five-seed mean of final three validation scores; no heldout/test evaluation in pilot',
        scores=scores, selected=selected, selected_on='validation_only', test_used_for_selection=False,
        keep_both_variants=True)
    (Path(out)/'preprocessing_selection.json').write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['pilot', 'confirm', 'diagnostic'], required=True)
    p.add_argument('--output', default='results/v4_study')
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--scenarios', nargs='+')
    args = p.parse_args()
    if args.phase == 'pilot':
        pilot(args.output, args.workers)
        return
    decision = json.loads((Path(args.output)/'preprocessing_selection.json').read_text())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'v3'))
    from protocol import scenarios, Calibration
    names = args.scenarios or list(scenarios(Calibration.load('data/btc_full_day/paper_protocol_calibration.json')))
    if args.phase == 'diagnostic':
        names = ['competition_random', 'competition_random_half']
    jobs = [dict(name=name, seed=seed, normalization=decision['selected'],
        reward_mode='spread_only' if args.phase == 'diagnostic' else 'total',
        output=str(Path(args.output)/args.phase), min_steps=256000,
        max_steps=1024000, check_every=128000, eval_steps=32000,
        validation_only=False, fixed_budget=False) for name in names for seed in range(5)]
    dispatch(jobs, Path(args.output)/args.phase, args.workers)


if __name__ == '__main__':
    main()
