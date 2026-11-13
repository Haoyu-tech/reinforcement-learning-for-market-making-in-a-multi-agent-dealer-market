"""Exploratory conditional-expectation reward extension; never a paper claim.

For independent investor flow and GBM, replacing inventory PnL by its conditional
mean preserves expected discounted TOTAL PnL for any fixed policy. It reduces
training reward noise; final evaluation still uses realized total PnL.
It is invalid for the toxic flow or nonlinear risk penalties and rejects them.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import numpy as np
import experiment


def collect_conditional(policy, env, reward_mode):
    if env.cfg.toxic or env.cfg.penalty != 'none':
        raise ValueError('Conditional reward extension rejects toxic flow and risk penalties')
    obs = env.observation()
    previous_actions = np.zeros((env.batch, 3), np.float32)
    previous_rewards = np.zeros(env.batch, np.float32)
    records, raw_states, actual = [], [], []
    for _ in range(200):
        a, extra, transformed = policy.act(obs, update=True)
        nxt, original_reward, done, info = env.step(learner=a)
        expected_inventory_pnl = info['inventory'][:, 0]*info['price'][:, 0]*np.expm1(env.cfg.drift*env.cfg.dt)
        reward = info['spread_pnl'][:, 0]-info['hedge_cost'][:, 0]+expected_inventory_pnl
        records.append(dict(obs=transformed, actions=a, rewards=reward.astype(np.float32),
            prev_actions=previous_actions.copy(), prev_rewards=previous_rewards.copy(),
            dones=np.full(env.batch, done), vf_preds=extra['vf_preds'],
            behaviour_logits=extra['behaviour_logits'], action_logp=extra['action_logp']))
        raw_states.append(obs.copy())
        actual.append(original_reward.copy())
        obs = env.reset() if done else nxt
        previous_actions = np.zeros_like(a) if done else a.copy()
        previous_rewards = np.zeros_like(reward, dtype=np.float32) if done else reward.astype(np.float32)
    return {k: np.stack([r[k] for r in records]) for k in records[0]}, np.stack(raw_states), np.stack(actual)


def worker(job):
    name, seed, normalization, output = job
    original = experiment.collect
    experiment.collect = collect_conditional
    try:
        result = experiment.run(name, seed, normalization, 'conditional_total', output,
            min_steps=256000, max_steps=1024000, check_every=128000, eval_steps=32000)
    finally:
        experiment.collect = original
    path = Path(output)/name/('seed_'+str(seed))/'result.json'
    document = json.loads(path.read_text())
    document['extension'] = dict(name='conditional-mean inventory PnL training reward',
        expected_total_pnl_objective_preserved_under_model=True,
        final_evaluation_uses_realized_total_pnl=True, original_paper_implementation=False,
        rejects_toxic_and_risk_penalties=True,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path.write_text(json.dumps(document, indent=2))
    return name, seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='results/v4_study')
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    root = Path(a.output)
    selection = json.loads((root/'preprocessing_selection.json').read_text())
    target = root/'variance_extension'
    target.mkdir(parents=True, exist_ok=True)
    jobs = [(name, seed, selection['selected'], str(target)) for name in ['competition_random', 'competition_random_half'] for seed in range(5)]
    source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path = target/'study_manifest.json'
    if path.exists():
        old = json.loads(path.read_text())
        if old['source_sha256'] != source:
            raise ValueError('Extension source changed')
    status = dict(source_sha256=source, state='running', completed=[], errors=[],
        exploratory_extension_not_paper_implementation=True, expected_runs=10)
    todo = []
    for job in jobs:
        name, seed, _, output = job
        result_path = Path(output)/name/('seed_'+str(seed))/'result.json'
        if result_path.exists() and json.loads(result_path.read_text()).get('extension', {}).get('source_sha256') == source:
            status['completed'].append(dict(name=name, seed=seed))
        else:
            todo.append(job)
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(worker, job): job for job in todo}
        for future in as_completed(futures):
            job = futures[future]
            try:
                name, seed = future.result()
                status['completed'].append(dict(name=name, seed=seed))
                print('EXTENSION DONE {}/10 {} seed={}'.format(len(status['completed']), name, seed), flush=True)
            except Exception as error:
                status['errors'].append(dict(job=job, error=repr(error)))
                print('EXTENSION FAILED {}'.format(repr(error)), flush=True)
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps(status, indent=2))
            temp.replace(path)
    status['state'] = 'complete' if not status['errors'] else 'failed'
    path.write_text(json.dumps(status, indent=2))
    if status['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
