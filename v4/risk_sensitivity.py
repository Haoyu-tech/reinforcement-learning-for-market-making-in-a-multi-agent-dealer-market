"""Exploratory fixed 0.1x risk-penalty sensitivity; original results untouched.

Selected after observing no-trading behavior, so evaluate all fixed models on a
new common random market seed. Does not claim the original coefficients worked.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import pickle
import numpy as np
import experiment
from market import DealerMarket


class ScaledRiskMarket(DealerMarket):
    scale = .1

    def step(self, *args, **kwargs):
        obs, reward, done, info = super().step(*args, **kwargs)
        info['reward'] = info['pnl']+self.scale*(info['reward']-info['pnl'])
        return obs, info['reward'][:, 0].copy(), done, info


def worker(job):
    name, seed, root = job
    root = Path(root)
    mode = json.loads((root/'preprocessing_selection.json').read_text())['selected']
    original_market, original_evaluate = experiment.DealerMarket, experiment.evaluate
    experiment.DealerMarket = ScaledRiskMarket
    # Validation remains distinct. New final market stream, not the original
    # final test that motivated this exploratory sensitivity.
    def evaluate_new(policy, cfg, market_seed, samples, tables=None):
        if market_seed >= 9100000:
            market_seed += 10000000
        return original_evaluate(policy, cfg, market_seed, samples, tables)
    experiment.evaluate = evaluate_new
    ScaledRiskMarket.scale = .1
    destination = root/'risk_sensitivity/scaled'
    try:
        experiment.run(name, seed, mode, 'total', str(destination), min_steps=1024000,
            max_steps=1024000, check_every=128000, eval_steps=32000, fixed_budget=True)
        result_path = destination/name/('seed_'+str(seed))/'result.json'
        doc = json.loads(result_path.read_text())
        doc['risk_sensitivity'] = dict(penalty_multiplier=.1, original_paper_coefficient=False,
            exploratory=True, final_market_seed=19100000+seed, untouched_original_result=True)
        result_path.write_text(json.dumps(doc, indent=2))
        # Fresh common test for frozen original-alpha and no-penalty references.
        ScaledRiskMarket.scale = 1.
        from protocol import scenarios, Calibration
        cases = scenarios(Calibration.load('data/btc_full_day/paper_protocol_calibration.json'))
        for label, reference_name in [('original_alpha', name), ('no_penalty', 'risk_none')]:
            source = root/'confirm'/reference_name/('seed_'+str(seed))/'policy.pkl'
            with source.open('rb') as stream:
                state = pickle.load(stream)
            model = experiment.FilteredPolicy(seed+900000, state['normalization'])
            model.impl.set_weights(state['weights'])
            model.filter = state['filter'].copy() if state['filter'] else None
            cfg, _ = cases[reference_name]
            test = original_evaluate(model, cfg, 19100000+seed, 32000)
            model.close()
            target = root/'risk_sensitivity'/label/name/('seed_'+str(seed))
            target.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target/'evaluation.npz', **test)
            (target/'result.json').write_text(json.dumps(dict(result=experiment.metrics(test), seed=seed,
                label=label, source_model=str(source), final_market_seed=19100000+seed,
                frozen_model=True, additional_training=False), indent=2))
    finally:
        experiment.DealerMarket, experiment.evaluate = original_market, original_evaluate
    return name, seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='results/v4_study')
    p.add_argument('--workers', type=int, default=15)
    a = p.parse_args()
    target = Path(a.output)/'risk_sensitivity'
    target.mkdir(parents=True, exist_ok=True)
    path = target/'study_manifest.json'
    signature = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if path.exists() and json.loads(path.read_text())['source_sha256'] != signature:
        raise ValueError('Risk sensitivity source changed')
    status = dict(source_sha256=signature, penalty_multiplier=.1, exploratory=True,
        common_fresh_market_seeds=[19100000+s for s in range(5)], state='running', completed=[], errors=[])
    jobs = []
    for name in ['risk_stdev', 'risk_square', 'risk_asymmetric']:
        for seed in range(5):
            files = [target/group/name/('seed_'+str(seed))/'result.json' for group in ['scaled', 'original_alpha', 'no_penalty']]
            if all(f.is_file() for f in files):
                status['completed'].append(dict(name=name, seed=seed, resumed=True))
            else:
                jobs.append((name, seed, a.output))
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(worker, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                name, seed = future.result()
                status['completed'].append(dict(name=name, seed=seed))
                print('RISK SENSITIVITY DONE {}/15 {} seed={}'.format(len(status['completed']), name, seed), flush=True)
            except Exception as error:
                status['errors'].append(dict(job=job, error=repr(error)))
                print('RISK SENSITIVITY FAILED '+repr(error), flush=True)
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(status, indent=2))
            temporary.replace(path)
    status['state'] = 'complete' if not status['errors'] else 'failed'
    path.write_text(json.dumps(status, indent=2))
    if status['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
