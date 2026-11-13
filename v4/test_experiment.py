import sys
from pathlib import Path
import unittest
import numpy as np
from experiment import FilteredPolicy, executed_distance, paired_mean_test, evaluate
from market import Config


class ReplicationChecks(unittest.TestCase):
    def test_filter_updates_only_on_training(self):
        p = FilteredPolicy(7, 'meanstd')
        x = np.random.RandomState(1).normal(size=(20, 64)).astype(np.float32)
        p.act(x, update=True)
        count, mean = p.filter.rs.n, p.filter.rs.mean.copy()
        p.act(x+100, update=False)
        self.assertEqual(p.filter.rs.n, count)
        np.testing.assert_array_equal(p.filter.rs.mean, mean)
        evaluate(p, Config(), 123, 100)
        self.assertEqual(p.filter.rs.n, count)
        p.close()

    def test_filter_and_network_copy_preserve_logits(self):
        a, b = FilteredPolicy(1, 'meanstd'), FilteredPolicy(2, 'meanstd')
        x = np.random.RandomState(2).normal(size=(20, 64)).astype(np.float32)
        a.act(x, update=True)
        b.copy_from(a)
        np.testing.assert_array_equal(a.act(x)[1]['behaviour_logits'], b.act(x)[1]['behaviour_logits'])
        self.assertEqual(a.filter.rs.n, b.filter.rs.n)
        a.close()
        b.close()

    def test_saturated_latent_shift_does_not_change_execution(self):
        old = np.tile([10., -10., -10., -5., -5., -5.], (100, 1))
        new = old.copy()
        new[:, :3] *= 2
        d = executed_distance(old, new)
        self.assertEqual(d['maximum_normalized_mae'], 0.)
        self.assertEqual(d['normalized_p95'], 0.)

    def test_executed_change_is_detected(self):
        old = np.tile([0., 0., .5, -5., -5., -5.], (100, 1))
        new = old.copy()
        new[:, 0] = .5
        self.assertGreater(executed_distance(old, new)['maximum_normalized_mae'], .2)

    def test_paired_test_detects_systematic_shift(self):
        x = np.arange(20, dtype=float)
        self.assertTrue(paired_mean_test(x, x)['within_mc_interval'])
        self.assertFalse(paired_mean_test(x, x+1)['within_mc_interval'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
