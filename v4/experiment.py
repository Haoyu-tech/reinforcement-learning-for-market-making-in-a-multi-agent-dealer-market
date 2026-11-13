"""Controlled best-effort replication using the real historical RLlib policy.

Changes are explicit: optional training-only MeanStdFilter and a bounded-action
stability diagnostic. Market equations and total-PnL rewards remain unchanged.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'v3'))
from historical import OfficialPolicy, SampleBatch, compute_advantages, metrics, gaussian_kl
from protocol import scenarios, Calibration
from market import DealerMarket, config_dict
from ray.rllib.utils.filter import MeanStdFilter


class FilteredPolicy:
    def __init__(self, seed, normalization='none'):
        self.impl = OfficialPolicy(seed)
        self.normalization = normalization
        self.filter = MeanStdFilter((64,)) if normalization == 'meanstd' else None

    def prepare(self, raw, update=False):
        return np.asarray(self.filter(raw, update=update) if self.filter else raw, dtype=np.float32)

    def act(self, raw, update=False):
        transformed = self.prepare(raw, update)
        a, extra = self.impl.act(transformed)
        return a, extra, transformed

    def copy_from(self, other):
        self.impl.set_weights(other.impl.weights())
        self.filter = other.filter.copy() if other.filter else None

    def close(self):
        self.impl.close()


def executed_distance(old, new, noise=None):
    if noise is None:
        noise = np.random.RandomState(1911).normal(size=(16, len(old), 3))
    m0, s0 = np.split(old, 2, 1)
    m1, s1 = np.split(new, 2, 1)
    low, high = np.array([-1, -1, 0]), np.array([1, 1, 1])
    a0 = np.clip(m0+np.exp(s0)*noise, low, high)
    a1 = np.clip(m1+np.exp(s1)*noise, low, high)
    delta = np.abs(a0-a1)/(high-low)
    return dict(normalized_mae_by_action=delta.mean((0, 1)).tolist(),
        maximum_normalized_mae=float(delta.mean((0, 1)).max()),
        normalized_p95=float(np.quantile(delta, .95)))


def paired_mean_test(previous, current):
    difference = np.asarray(current)-np.asarray(previous)
    se = float(difference.std(ddof=1)/np.sqrt(len(difference)))
    mean = float(difference.mean())
    # Finite Monte Carlo precision is retained, never replaced by treating
    # correlated individual timesteps as independent observations.
    return dict(paired_mean_change=mean, paired_standard_error=se,
                within_mc_interval=bool(abs(mean) <= 1.96*se+1e-8))


def evaluate(policy, cfg, seed, samples, tables=None):
    batch = 20
    if samples % batch:
        raise ValueError('Evaluation requires complete vector steps')
    env = DealerMarket(copy.deepcopy(cfg), batch, seed)
    if tables:
        env.tables = copy.deepcopy(tables)
    else:
        env.warmup(512)
    obs, records = env.reset(), []
    filter_count_before = policy.filter.rs.n if policy and policy.filter else None
    for _ in range(samples//batch):
        a = policy.act(obs, update=False)[0] if policy else None
        obs, _, done, info = env.step(learner=a)
        records.append({key: value.copy() for key, value in info.items()})
        if done:
            obs = env.reset()
    if filter_count_before is not None and policy.filter.rs.n != filter_count_before:
        raise ArithmeticError('Evaluation modified training normalization statistics')
    return {key: np.concatenate([r[key] for r in records]) for key in records[0]}


def collect(policy, env, reward_mode):
    obs = env.observation()
    previous_actions = np.zeros((env.batch, 3), np.float32)
    previous_rewards = np.zeros(env.batch, np.float32)
    records, raw_states, market_rewards = [], [], []
    for _ in range(200):
        a, extra, transformed = policy.act(obs, update=True)
        nxt, original_reward, done, info = env.step(learner=a)
        reward = info['spread_pnl'][:, 0] if reward_mode == 'spread_only' else original_reward
        records.append(dict(obs=transformed, actions=a, rewards=reward.astype(np.float32),
            prev_actions=previous_actions.copy(), prev_rewards=previous_rewards.copy(),
            dones=np.full(env.batch, done), vf_preds=extra['vf_preds'],
            behaviour_logits=extra['behaviour_logits'], action_logp=extra['action_logp']))
        raw_states.append(obs.copy())
        market_rewards.append(info['pnl'][:, 0].copy())
        obs = env.reset() if done else nxt
        previous_actions = np.zeros_like(a) if done else a.copy()
        previous_rewards = np.zeros_like(reward, dtype=np.float32) if done else reward.astype(np.float32)
    return {k: np.stack([r[k] for r in records]) for k in records[0]}, np.stack(raw_states), np.array(market_rewards)


def run(name, seed, normalization, reward_mode, output, min_steps=256000,
        max_steps=1024000, check_every=128000, eval_steps=32000, validation_only=False,
        fixed_budget=False):
    cfg, section = scenarios(Calibration.load('data/btc_full_day/paper_protocol_calibration.json'))[name]
    destination = Path(output)/name/('seed_'+str(seed))
    destination.mkdir(parents=True, exist_ok=True)
    if cfg.agents[0].kind != 'rl':
        test = evaluate(None, cfg, seed+9100000, eval_steps)
        result = metrics(test)
        result.update(seed=seed, trained=False, evaluation_batch=20)
        np.savez_compressed(destination/'evaluation.npz', **test)
        (destination/'result.json').write_text(json.dumps(dict(result=result, config=config_dict(cfg),
            protocol_section=section, normalization='not_applicable', original_numeric_replication=False), indent=2))
        return result
    np.random.seed(seed)
    policy, validator = FilteredPolicy(seed, normalization), FilteredPolicy(seed+2000, normalization)
    env = DealerMarket(copy.deepcopy(cfg), 20, seed)
    env.warmup(512)
    shuffle = np.random.RandomState(seed+777)
    logs, validation, previous_logits, probe = [], [], None, None
    stable_count, steps, next_check = 0, 0, check_every
    started = time.time()
    while steps < max_steps:
        data, raw_states, original_rewards = collect(policy, env, reward_mode)
        batches = []
        for i in range(20):
            one = SampleBatch({k: v[:, i] for k, v in data.items()})
            batches.append(compute_advantages(one, 0., policy.impl.config['gamma'], policy.impl.config['lambda'], use_gae=True))
        batch = SampleBatch.concat_samples(batches)
        av = batch['advantages']
        batch['advantages'] = (av-av.mean())/max(1e-4, av.std())
        order, stats = shuffle.permutation(batch.count), []
        for start in range(0, batch.count-255, 256):
            idx = order[start:start+256]
            stats.append(policy.impl.learn(SampleBatch({k: v[idx] for k, v in batch.items()}))['learner_stats'])
        kl = float(np.mean([s['kl'] for s in stats]))
        policy.impl.update_kl(kl)
        steps += batch.count
        if not all(np.isfinite(s['total_loss']) for s in stats):
            raise FloatingPointError('Nonfinite official PPO loss')
        logs.append(dict(step=steps, reward=float(data['rewards'].mean()),
            original_total_pnl=float(original_rewards.mean()), loss=float(np.mean([s['total_loss'] for s in stats])),
            kl=kl, vf_explained_variance=float(np.mean([s['vf_explained_var'] for s in stats]))))
        if probe is None:
            probe = raw_states.reshape(-1, 64)[::8].copy()
        if steps >= next_check or steps >= max_steps:
            validator.copy_from(policy)
            val_data = evaluate(validator, cfg, seed+321000, 12000, env.tables)
            val = metrics(val_data)
            score_key = 'spread_pnl' if reward_mode == 'spread_only' else 'reward'
            worlds = val_data[score_key][:, 0].reshape(-1, 20).mean(0)
            logits = validator.act(probe)[1]['behaviour_logits']
            distance = None if previous_logits is None else executed_distance(previous_logits, logits)
            paired = None if not validation else paired_mean_test(validation[-1]['world_means'], worlds)
            stable = bool(distance is not None and distance['maximum_normalized_mae'] < .02
                and distance['normalized_p95'] < .1 and paired['within_mc_interval'])
            stable_count = stable_count+1 if stable and steps >= min_steps else 0
            validation.append(dict(step=steps, score=float(worlds.mean()), world_means=worlds.tolist(),
                pnl=val['pnl'][0], inventory_std=val['inventory_std'][0], spread_pnl=val['spread_pnl'][0],
                latent_gaussian_kl=None if previous_logits is None else gaussian_kl(previous_logits, logits),
                executed_policy_distance=distance, paired_score_test=paired,
                stable=stable, consecutive_stable_checks=stable_count,
                filter_count=policy.filter.rs.n if policy.filter else None))
            previous_logits = logits.copy()
            next_check += check_every
            (destination/'validation.json').write_text(json.dumps(validation, indent=2))
            print('{} {} {} seed={} step={} val={:.4f} stable={}'.format(name, normalization,
                reward_mode, seed, steps, worlds.mean(), stable_count), flush=True)
            if stable_count >= 3 and not fixed_budget:
                break
    validator.close()
    if validation_only:
        result = dict(seed=seed, trained=True, validation_only=True, training_steps=steps,
            validation_selection_score=float(np.mean([v['score'] for v in validation[-3:]])),
            empirical_stability_passed=stable_count >= 3)
    else:
        tester = FilteredPolicy(seed+900000, normalization)
        tester.copy_from(policy)
        test = evaluate(tester, cfg, seed+9100000, eval_steps, env.tables)
        result = metrics(test)
        result.update(seed=seed, trained=True, training_steps=steps, evaluation_batch=20,
            convergence=dict(stable=stable_count >= 3, empirical_checks_passed=stable_count >= 3,
                             budget_reached=steps >= max_steps))
        np.savez_compressed(destination/'evaluation.npz', **test)
        tester.close()
    result['seconds'] = time.time()-started
    with (destination/'policy.pkl').open('wb') as f:
        pickle.dump(dict(weights=policy.impl.weights(), filter=policy.filter,
            normalization=normalization, market=config_dict(cfg)), f)
    with (destination/'training.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=logs[0])
        writer.writeheader()
        writer.writerows(logs)
    manifest = dict(result=result, config=config_dict(cfg), normalization=normalization,
        reward_mode=reward_mode, validation_only=validation_only, validation=validation,
        original_numeric_replication=False, author_preprocessing_known=False,
        backend='official Ray 0.7.6 PPOTFPolicy, TensorFlow 1.15',
        preprocessing='official MeanStdFilter, training updates only; evaluation frozen' if policy.filter else 'unchanged v3 observations',
        training_protocol=dict(min_steps=min_steps, max_steps=max_steps, check_every=check_every,
            eval_steps=eval_steps, fixed_budget=fixed_budget, batch=4000, minibatch=256, sgd_passes=1),
        stability='3 consecutive checks: clipped-action CRN MAE<.02, p95<.1 and paired world-mean reward change within 1.96 SE; diagnostic, not proof',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (destination/'result.json').write_text(json.dumps(manifest, indent=2))
    policy.close()
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--scenario', default='competition_random')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--normalization', choices=['none', 'meanstd'], default='meanstd')
    p.add_argument('--reward-mode', choices=['total', 'spread_only'], default='total')
    p.add_argument('--output', default='results/v4')
    p.add_argument('--min-steps', type=int, default=256000)
    p.add_argument('--max-steps', type=int, default=1024000)
    p.add_argument('--check-every', type=int, default=128000)
    p.add_argument('--eval-steps', type=int, default=32000)
    p.add_argument('--validation-only', action='store_true')
    p.add_argument('--fixed-budget', action='store_true')
    args = p.parse_args()
    run(args.scenario, args.seed, args.normalization, args.reward_mode, args.output,
        args.min_steps, args.max_steps, args.check_every, args.eval_steps, args.validation_only, args.fixed_budget)
