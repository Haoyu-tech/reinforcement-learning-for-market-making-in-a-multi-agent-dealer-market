import unittest
import numpy as np
from variance_extension import collect_conditional
from market import Config, DealerMarket


class DummyPolicy:
    def act(self, obs, update=False):
        n = len(obs)
        return np.tile([-.2, -.2, .1], (n, 1)), dict(vf_preds=np.zeros(n),
            behaviour_logits=np.zeros((n, 6)), action_logp=np.zeros(n)), obs


class VarianceExtensionChecks(unittest.TestCase):
    def test_conditional_gbm_inventory_expectation(self):
        rng = np.random.RandomState(1911)
        z = rng.choice([-10., -1., 1., 10.], size=500000)
        p, dt, sigma = 100., 1/(252*26), .1
        for mu in [-1., 0., 1.]:
            shock = rng.normal(size=len(z))
            dp = p*np.expm1((mu-.5*sigma*sigma)*dt+sigma*np.sqrt(dt)*shock)
            residual = z*(dp-p*np.expm1(mu*dt))
            se = residual.std(ddof=1)/np.sqrt(len(residual))
            self.assertLess(abs(residual.mean()), 4*se)

    def test_zero_volatility_reward_matches_realized_pnl(self):
        env = DealerMarket(Config(sigma=0., drift=.5), batch=20, seed=3)
        d, _, actual = collect_conditional(DummyPolicy(), env, 'conditional_total')
        np.testing.assert_allclose(d['rewards'], actual, rtol=1e-6, atol=1e-8)

    def test_rejects_invalid_scenarios(self):
        for cfg in [Config(toxic=True), Config(penalty='square'), Config(penalty='asymmetric'), Config(penalty='stdev')]:
            with self.assertRaises(ValueError):
                collect_conditional(DummyPolicy(), DealerMarket(cfg, batch=1), 'conditional_total')


if __name__ == '__main__':
    unittest.main(verbosity=2)
