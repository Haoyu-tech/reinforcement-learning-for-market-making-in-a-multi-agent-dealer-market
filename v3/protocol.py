"""Experiment configurations; paper disclosures retained, BTC depth only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"v2"))
from market import Agent, Config, Calibration

def scenarios(calibration=None):
    cal = calibration or Calibration()
    cases = {}
    def add(name, agents, section, **kwargs):
        if cal.unit_btc > 0:
            kwargs.update(sigma=cal.fitted_sigma, dt=cal.market_dt)
        cases[name] = (Config(agents=agents, calibration=cal, **kwargs), section)
    for name, peer in [('persistent_half', Agent(bid=-.5, ask=-.5)),
                       ('persistent', Agent()), ('random_half', Agent('random', low=-.5, high=.5)),
                       ('random', Agent('random'))]:
        add('competition_'+name, [Agent('rl'), peer], '6.1 / Tables 1,3 / Figures 1,3,8-11')
    for target in [.25, .5]:
        for risk in [0, 1, 2]:
            add(f'competition_adaptive_ms{target}_risk{risk}',
                [Agent('rl'), Agent('adaptive', target=target, risk=risk)], '6.1 / Table 2 / Figures 1-3')
    for drift in [-1., -.01, 0., .01, 1.]:
        add(f'drift_{drift}', [Agent('rl'), Agent('adaptive'), Agent()], '6.2 / Figure 4', drift=drift)
    for penalty in ['none', 'stdev', 'square', 'asymmetric']:
        add('risk_'+penalty, [Agent('rl'), Agent(bid=-.5, ask=0.)], '6.3 / Figure 5', penalty=penalty)
    add('toxic', [Agent('rl'), Agent('adaptive')], '7.4 / Figure 12', toxic=True,
        free_hedge=True, spread_multiplier=10.)
    for risk in [0, 2]:
        add(f'adaptive_skew_risk{risk}', [Agent('adaptive', target=.25, risk=risk),
            Agent(bid=-.5, ask=0.)], '7.5 / Figures 13,14')
    for peer_name, peer in [('persistent', Agent(bid=-.5, ask=0.)), ('random', Agent('random'))]:
        for target in [.25, .5]:
            for risk in [0, 1, 2]:
                add(f'adaptive_vs_{peer_name}_ms{target}_risk{risk}',
                    [Agent('adaptive', target=target, risk=risk), peer], '7.7 / Tables 4,5')
    for name, risk, hp, ignore in [('random', 1, 'random', False),
                                  ('persistent', 1, 'persistent', False),
                                  ('adaptive_risk1', 1, 'adaptive', False),
                                  ('adaptive_risk2', 2, 'adaptive', False),
                                  ('ignore_flow_risk1', 1, 'adaptive', True),
                                  ('ignore_flow_risk2', 2, 'adaptive', True)]:
        add('hedging_'+name, [Agent('adaptive', risk=risk, hedge_policy=hp,
             ignore_internalization=ignore), Agent()], '7.6 / Table 6')
    return cases
