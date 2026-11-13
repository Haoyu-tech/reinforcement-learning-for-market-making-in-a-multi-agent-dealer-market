"""Use official RLlib 0.7.6 PPOTFPolicy, without reimplementing its PPO loss.

The author did not disclose the RLlib version. This is a dated, source-verifiable
reference implementation, not a claim to possess the exact author environment.
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
import gym
import tensorflow as tf
import ray
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import compute_advantages
from protocol import scenarios, Calibration
from market import DealerMarket, config_dict


class OfficialPolicy:
    def __init__(self, seed):
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.config.update(seed=seed, num_workers=0, num_gpus=0,
            sgd_minibatch_size=256, num_sgd_iter=1, lr=5e-5, clip_param=.3)
        self.config['model'].update(fcnet_hiddens=[256, 256], fcnet_activation='tanh', vf_share_layers=False)
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(seed)
            self.session = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1, device_count={'GPU': 0}))
            with self.session.as_default():
                self.policy = PPOTFPolicy(
                    gym.spaces.Box(low=-np.inf, high=np.inf, shape=(64,), dtype=np.float32),
                    gym.spaces.Box(low=np.array([-1., -1., 0.], np.float32),
                                   high=np.ones(3, np.float32), dtype=np.float32), self.config)

    def act(self, observations):
        with self.graph.as_default(), self.session.as_default():
            a, _, extra = self.policy.compute_actions(observations)
        return a, extra

    def learn(self, batch):
        with self.graph.as_default(), self.session.as_default():
            return self.policy.learn_on_batch(batch)

    def weights(self):
        with self.graph.as_default(), self.session.as_default():
            return self.policy.get_weights()

    def set_weights(self, weights):
        with self.graph.as_default(), self.session.as_default():
            self.policy.set_weights(weights)

    def update_kl(self, kl):
        with self.graph.as_default(), self.session.as_default():
            self.policy.update_kl(kl)

    def close(self):
        self.session.close()


def rollout(policy, env, count=200):
    rows = []
    obs = env.observation()
    previous_actions = np.zeros((env.batch, 3), np.float32)
    previous_rewards = np.zeros(env.batch, np.float32)
    for _ in range(count):
        actions, extra = policy.act(obs)
        nxt, reward, done, info = env.step(learner=actions)
        rows.append(dict(obs=obs, new_obs=nxt, actions=actions, rewards=reward.astype(np.float32),
            prev_actions=previous_actions.copy(), prev_rewards=previous_rewards.copy(),
            dones=np.full(env.batch, done), vf_preds=extra['vf_preds'],
            behaviour_logits=extra['behaviour_logits'], action_logp=extra['action_logp'],
            competitor_reward=info['pnl'][:, 1]))
        obs = env.reset() if done else nxt
        previous_actions = np.zeros_like(actions) if done else actions.copy()
        previous_rewards = np.zeros_like(reward, dtype=np.float32) if done else reward.astype(np.float32)
    return {key: np.stack([r[key] for r in rows]) for key in rows[0]}


def evaluate(policy, cfg, seed, samples=10000, tables=None, batch=20):
    env = DealerMarket(copy.deepcopy(cfg), batch=batch, seed=seed)
    if tables:
        env.tables = copy.deepcopy(tables)
    else:
        env.warmup(512)
    obs = env.reset()
    data, remaining = [], samples
    while remaining:
        actions = policy.act(obs)[0] if policy is not None else None
        obs, _, done, info = env.step(learner=actions)
        take = min(batch, remaining)
        data.append({key: value[:take].copy() for key, value in info.items()})
        remaining -= take
        if done:
            obs = env.reset()
    return {key: np.concatenate([r[key] for r in data]) for key in data[0]}


def metrics(data):
    result = {key: value.mean(0).tolist() for key, value in data.items()}
    result.update(inventory_std=data['inventory'].std(0).tolist(),
        inventory_pnl_std=data['inventory_pnl'].std(0).tolist(),
        bid_std=data['bid_epsilon'].std(0).tolist(), ask_std=data['ask_epsilon'].std(0).tolist(),
        excess_pnl_x100=float(100*np.mean(data['pnl'][:, 0]-data['pnl'][:, 1])),
        evaluation_samples=len(data['pnl']))
    result['excess_components_x100'] = dict(
        spread=float(100*np.mean(data['spread_pnl'][:, 0]-data['spread_pnl'][:, 1])),
        inventory=float(100*np.mean(data['inventory_pnl'][:, 0]-data['inventory_pnl'][:, 1])),
        hedge=float(100*np.mean(data['hedge_cost'][:, 1]-data['hedge_cost'][:, 0])))
    return result


def gaussian_kl(old, new):
    mean0, logstd0 = np.split(old, 2, axis=1)
    mean1, logstd1 = np.split(new, 2, axis=1)
    return float(np.mean(np.sum(logstd1-logstd0+(np.exp(2*logstd0)+(mean0-mean1)**2)/(2*np.exp(2*logstd1))-.5, axis=1)))


def train_one(name, cfg, seed, out, args):
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    env = DealerMarket(copy.deepcopy(cfg), batch=20, seed=seed)
    env.warmup(512)
    policy, validation_policy = OfficialPolicy(seed), OfficialPolicy(seed+2000)
    initial = copy.deepcopy(policy.weights())
    shuffle = np.random.RandomState(seed+777)
    log, validations, stable_count, steps, next_check = [], [], 0, 0, args.check_every
    started = time.time()
    previous_logits, probe = None, None
    while steps < args.max_steps:
        data = rollout(policy, env)
        batches = []
        for i in range(env.batch):
            one = SampleBatch({k: v[:, i] for k, v in data.items() if k != 'competitor_reward'})
            # Each collected trajectory is exactly a full 200-step episode.
            one = compute_advantages(one, 0., policy.config['gamma'], policy.config['lambda'], use_gae=True)
            batches.append(one)
        batch = SampleBatch.concat_samples(batches)
        advantages = batch['advantages']
        batch['advantages'] = (advantages-advantages.mean())/max(1e-4, advantages.std())
        order = shuffle.permutation(batch.count)
        stats = []
        # Historical multi-GPU optimizer truncates to complete minibatches.
        for start in range(0, batch.count-255, 256):
            idx = order[start:start+256]
            stats.append(policy.learn(SampleBatch({k: v[idx] for k, v in batch.items()}))['learner_stats'])
        kl = float(np.mean([s['kl'] for s in stats]))
        policy.update_kl(kl)
        steps += batch.count
        log.append(dict(step=steps, reward=float(data['rewards'].mean()),
            competitor_pnl=float(data['competitor_reward'].mean()),
            loss=float(np.mean([s['total_loss'] for s in stats])), kl=kl))
        if probe is None:
            probe = data['obs'].reshape(-1, 64)[::8].copy()
        if steps >= next_check or steps >= args.max_steps:
            validation_policy.set_weights(policy.weights())
            val = metrics(evaluate(validation_policy, cfg, seed+10000, samples=10000, tables=env.tables))
            logits = validation_policy.act(probe)[1]['behaviour_logits']
            movement = None if previous_logits is None else gaussian_kl(previous_logits, logits)
            reward_change = None if not validations else abs(val['pnl'][0]-validations[-1]['pnl'])
            stable = movement is not None and movement < .01 and reward_change <= max(.02, abs(val['pnl'][0])*.05)
            stable_count = stable_count+1 if stable and steps >= args.min_steps else 0
            validations.append(dict(step=steps, pnl=val['pnl'][0], probe_kl=movement,
                reward_change=reward_change, stable=bool(stable), consecutive_stable_checks=stable_count))
            previous_logits = logits.copy()
            next_check += args.check_every
            (out/'validation.json').write_text(json.dumps(validations, indent=2))
            print('{} seed={} steps={} val={:.5f} stable={}'.format(name, seed, steps, val['pnl'][0], stable_count), flush=True)
            if stable_count >= 3:
                break
    validation_policy.close()
    # Fresh graph RNG and fresh market seed for final frozen-policy testing.
    test_policy = OfficialPolicy(seed+90000)
    test_policy.set_weights(policy.weights())
    test = evaluate(test_policy, cfg, seed+100000, samples=args.eval_steps, tables=env.tables)
    result = metrics(test)
    result.update(seed=seed, trained=True, training_steps=steps, evaluation_batch=20,
        convergence=dict(empirical_checks_passed=stable_count >= 3, stable=stable_count >= 3,
                         reached_budget=steps >= args.max_steps, validation_history=validations),
        seconds=time.time()-started)
    with (out/'official_policy.pkl').open('wb') as f:
        pickle.dump(dict(weights=policy.weights(), config=policy.config, market=config_dict(cfg)), f)
    np.savez_compressed(out/'evaluation.npz', **test)
    with (out/'training.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log[0])
        writer.writeheader()
        writer.writerows(log)
    config = copy.deepcopy(policy.config)
    # TF config includes optional non-JSON values; save repr as a separate text.
    (out/'official_config.txt').write_text(repr(config))
    source = Path(__file__).read_bytes()
    manifest = dict(result=result, config=config_dict(cfg), seed=seed,
        backend='official ray.rllib.agents.ppo.ppo_policy.PPOTFPolicy', ray_version=ray.__version__,
        tensorflow_version=tf.__version__, python=sys.version, numpy=np.__version__,
        author_exact_RLlib_version_known=False, strict_numeric_replication=False,
        training_protocol=dict(min_steps=args.min_steps, max_steps=args.max_steps, check_every=args.check_every,
            train_batch=4000, sgd_minibatch=256, num_sgd_iter=1, incomplete_minibatch_dropped=160,
            source_defaults='ray-0.7.6, overridden only for disclosed network/optimizer settings and local CPU collection'),
        source_sha256=hashlib.sha256(source).hexdigest())
    (out/'result.json').write_text(json.dumps(manifest, indent=2))
    policy.close()
    test_policy.close()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scenario', default='competition_random')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--calibration', default='data/btc_full_day/paper_protocol_calibration.json')
    p.add_argument('--output', default='results/v3_official')
    p.add_argument('--min-steps', type=int, default=512000)
    p.add_argument('--max-steps', type=int, default=2048000)
    p.add_argument('--check-every', type=int, default=128000)
    p.add_argument('--eval-steps', type=int, default=32000)
    args = p.parse_args()
    if args.max_steps < 4000 or args.check_every < 4000 or args.min_steps > args.max_steps:
        p.error('Invalid training budget')
    cfg, _ = scenarios(Calibration.load(args.calibration))[args.scenario]
    out = Path(args.output)/args.scenario/('seed_'+str(args.seed))
    if cfg.agents[0].kind != 'rl':
        out.mkdir(parents=True, exist_ok=True)
        test = evaluate(None, cfg, args.seed+100000, samples=args.eval_steps)
        result = metrics(test)
        result.update(seed=args.seed, trained=False, evaluation_batch=20)
        np.savez_compressed(out/'evaluation.npz', **test)
        (out/'result.json').write_text(json.dumps(dict(result=result, config=config_dict(cfg),
            strict_numeric_replication=False, backend='paper adaptive/persistent/random rule'), indent=2))
    else:
        train_one(args.scenario, cfg, args.seed, out, args)


if __name__ == '__main__':
    main()
