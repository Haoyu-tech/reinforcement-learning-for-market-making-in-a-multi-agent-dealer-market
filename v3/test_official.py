"""Checks against actual historical RLlib classes, not a replacement PPO."""
import json
from pathlib import Path
import unittest
import numpy as np
import tensorflow as tf
from ray.rllib.agents.ppo.ppo_policy import PPOLoss
from ray.rllib.models.tf.tf_action_dist import DiagGaussian
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.policy.sample_batch import SampleBatch
from historical import OfficialPolicy


class OfficialTests(unittest.TestCase):
    def test_official_loss_against_numpy_equations(self):
        rng = np.random.RandomState(19)
        old = rng.normal(0, .3, (8, 6)).astype(np.float32)
        new = rng.normal(0, .5, (8, 6)).astype(np.float32)
        actions = rng.normal(0, 1, (8, 3)).astype(np.float32)
        adv = np.array([-3, -2, -1, -.5, .5, 1, 2, 3], np.float32)
        previous_values = np.arange(8, dtype=np.float32)
        values = previous_values+np.array([-20, -12, -3, 0, 1, 4, 12, 20], np.float32)
        target = rng.normal(0, 2, 8).astype(np.float32)
        def logp(logits):
            mean, logstd = np.split(logits, 2, 1)
            return -.5*(((actions-mean)/np.exp(logstd))**2).sum(1)-1.5*np.log(2*np.pi)-logstd.sum(1)
        ratio = np.exp(logp(new)-logp(old))
        m0, s0 = np.split(old, 2, 1)
        m1, s1 = np.split(new, 2, 1)
        kl = (s1-s0+(np.exp(2*s0)+(m0-m1)**2)/(2*np.exp(2*s1))-.5).sum(1)
        surrogate = np.minimum(adv*ratio, adv*np.clip(ratio, .7, 1.3))
        clipped_values = previous_values+np.clip(values-previous_values, -10, 10)
        vf_loss = np.maximum((values-target)**2, (clipped_values-target)**2)
        expected = np.mean(-surrogate+.2*kl+vf_loss)
        with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)) as session:
            obj = PPOLoss(None, DiagGaussian, None, tf.constant(target), tf.constant(adv),
                tf.constant(actions), tf.constant(old), tf.constant(logp(old)), tf.constant(previous_values),
                DiagGaussian(tf.constant(new), None), tf.constant(values), tf.constant(.2),
                tf.constant(np.ones(8, dtype=bool)), clip_param=.3, vf_clip_param=10., vf_loss_coeff=1.)
            actual = session.run(obj.loss)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-4)

    def test_original_gae_full_episode(self):
        rollout = SampleBatch({'rewards': np.array([1., 2., 3.], np.float32),
            'actions': np.zeros((3, 3), np.float32),
            'vf_preds': np.array([.1, .2, .3], np.float32), 'dones': np.array([False, False, True])})
        batch = compute_advantages(rollout, 0., .99, 1., use_gae=True)
        expected = np.array([1+.99*2+.99**2*3, 2+.99*3, 3])
        np.testing.assert_allclose(batch['value_targets'], expected, atol=1e-6)

    def test_official_policy_architecture_and_reload(self):
        a, b = OfficialPolicy(1), OfficialPolicy(2)
        self.assertEqual(a.config['model']['fcnet_hiddens'], [256, 256])
        self.assertEqual(a.config['lambda'], 1.)
        self.assertEqual(a.config['kl_coeff'], .2)
        self.assertFalse(a.config['vf_share_layers'])
        self.assertIsNone(a.config['grad_clip'])
        b.set_weights(a.weights())
        observations = np.zeros((5, 64), np.float32)
        ae, be = a.act(observations)[1], b.act(observations)[1]
        np.testing.assert_array_equal(ae['behaviour_logits'], be['behaviour_logits'])
        np.testing.assert_array_equal(ae['vf_preds'], be['vf_preds'])
        a.close()
        b.close()


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(OfficialTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    p = Path('results/v3_official')
    p.mkdir(parents=True, exist_ok=True)
    (p/'official_test_results.json').write_text(json.dumps(dict(tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), passed=result.wasSuccessful()), indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
