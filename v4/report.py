"""Report every seed, rejected preprocessing, replicated and failed behaviors."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'v3'))
from protocol import scenarios, Calibration
from market import DealerMarket, Config

def interval(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return dict(mean=None, lower=None, upper=None, samples=0)
    half = float(t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else float('nan')
    return dict(mean=float(a.mean()), lower=float(a.mean() - half), upper=float(a.mean() + half))

def load_phase(base):
    entries = {}
    for file in sorted(base.glob('*/seed_*/result.json')):
        x = json.loads(file.read_text())
        x['_folder'] = file.parent
        entries.setdefault(file.parent.parent.name, []).append(x)
    return entries

def data(entries, name, pooled=False):
    selected = entries[name] if pooled else entries[name][:1]
    result = []
    for x in selected:
        with np.load(x['_folder'] / 'evaluation.npz') as z:
            result.append({k: z[k].copy() for k in z.files})
    return {k: np.concatenate([r[k] for r in result]) for k in result[0]}

def write_csv(path, rows):
    if not rows:
        path.write_text('status\nno_estimable_rows\n', encoding='utf-8')
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)

def make_figures(entries, diagnostics, extension, output, cal):
    output.mkdir(exist_ok=True)

    def save(fig, number, title):
        fig.suptitle(title + ' | official historical PPO, BTC depth calibration', fontsize=10)
        fig.tight_layout()
        fig.savefig(output / 'figure_{:02d}.png'.format(number), dpi=140)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, name, point in zip(axes, ['competition_random', 'competition_random_half'], [0, -0.25]):
        d = data(entries, name, True)
        ax.hist(d['bid_epsilon'][:, 0], bins=50, density=True, histtype='step', label='Total-PnL RL')
        dd = data(diagnostics, name, True)
        ax.hist(dd['bid_epsilon'][:, 0], bins=50, density=True, histtype='step', label='Spread-only diagnostic')
        de = data(extension, name, True)
        ax.hist(de['bid_epsilon'][:, 0], bins=50, density=True, histtype='step', ls=':', label='Conditional-total extension')
        ax.axvline(point, color='black', ls='--', label='Theoretical spread optimum')
        ax.set(xlabel='Bid epsilon', ylabel='Pooled five-seed density', title=name)
        ax.legend(fontsize=8)
    save(fig, 1, 'Learned quotation versus the analytical benchmark')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, risk in zip(axes, [0, 2]):
        name = 'competition_adaptive_ms0.5_risk' + str(risk)
        for entry in entries[name]:
            x = np.genfromtxt(entry['_folder'] / 'training.csv', delimiter=',', names=True)
            ax.plot(x['step'], x['reward'], alpha=0.6)
        ax.set(xlabel='Training transitions', ylabel='RL reward', title='Adaptive competitor, risk=' + str(risk))
    save(fig, 2, 'RL training curves; competitor training rewards were not recorded')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, name in zip(axes, ['competition_persistent', 'competition_adaptive_ms0.5_risk2']):
        d = data(entries, name, True)
        z = d['pre_inventory'][:, 0]
        bins = np.unique(np.quantile(z, np.linspace(0.01, 0.99, 16)))
        for key in ['bid_epsilon', 'ask_epsilon']:
            xy = [(z[(z >= lo) & (z < hi)].mean(), d[key][(z >= lo) & (z < hi), 0].mean()) for lo, hi in zip(bins[:-1], bins[1:]) if ((z >= lo) & (z < hi)).sum() >= 20]
            if xy:
                xx, yy = np.array(xy).T
                ax.plot(xx, yy, 'o-', label=key)
        ax.set(xlabel='Inventory before decision (1%-99% bins)', ylabel='Conditional mean epsilon', title=name)
        ax.legend(fontsize=8)
    save(fig, 3, 'Inventory-sensitive pricing; descriptive pooled bins')
    fig, ax = plt.subplots(figsize=(10, 4))
    for drift in [-1.0, -0.01, 0.0, 0.01, 1.0]:
        d = data(entries, 'drift_' + str(drift))
        ax.plot(d['inventory'][::20, 0][:500], label='mu=' + str(drift), alpha=0.8)
    ax.set(xlabel='Holdout trajectory step, seed 0', ylabel='Inventory; reset every 200 steps')
    ax.legend(fontsize=8)
    save(fig, 4, 'Three-dealer drift scenarios')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    risk_data = {mode: data(entries, 'risk_' + mode, True) for mode in ['none', 'stdev', 'square', 'asymmetric']}
    risk_bins = {key: np.histogram_bin_edges(np.concatenate([d[key][:, 0] for d in risk_data.values()]), bins=70) for key in ['inventory', 'inventory_pnl']}
    for mode, d in risk_data.items():
        for ax, key in zip(axes, ['inventory', 'inventory_pnl']):
            ax.hist(d[key][:, 0], bins=risk_bins[key], density=True, histtype='step', label=mode)
            ax.set(xlabel=key, ylabel='Pooled five-seed density')
            ax.legend(fontsize=8)
    save(fig, 5, 'Risk formulation distributions; full samples retained in metrics')
    env = DealerMarket(Config(calibration=cal), batch=1000, seed=1911)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.step(np.r_[0, np.cumsum(env.depth[0])], np.r_[env.offsets[0], env.offsets[0, -1]], where='post')
    ax.set(xlabel='Cumulative depth', ylabel='Reference offset above mid')
    save(fig, 6, 'Generated reference order book schematic')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i in range(12):
        axes[0].plot(env.depth[i] / env.depth[i, 0], alpha=0.7)
    axes[0].set(xlabel='1bp level', ylabel='Fitted normalized shape')
    axes[1].hist(env.depth[:, 0] / cal.depth_scale, bins=40, density=True)
    axes[1].set(xlabel='Gamma best-level volume', ylabel='Density')
    sizes = np.linspace(0.01, 5, 100)
    curves = env.trade_cost(np.broadcast_to(sizes, (1000, 100))) / sizes
    for i in range(12):
        axes[2].plot(sizes, curves[i])
    axes[2].set(xlabel='Trade size', ylabel='Reference average offset')
    save(fig, 7, 'BTC-derived depth statistics; paper price/spread laws preserved')
    for number, name in zip([8, 9, 10, 11], ['competition_random', 'competition_random_half', 'competition_persistent', 'competition_persistent_half']):
        fig, ax = plt.subplots(figsize=(8, 4))
        for entry in entries[name]:
            x = np.genfromtxt(entry['_folder'] / 'training.csv', delimiter=',', names=True)
            y = x['reward']
            smoothed = [y[max(0, i - 9):i + 1].mean() for i in range(len(y))]
            ax.plot(x['step'], smoothed, label='seed ' + str(entry['result']['seed']))
        ax.set(xlabel='Training transitions', ylabel='Trailing 10-update reward')
        ax.legend(fontsize=8)
        save(fig, number, name)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    d = data(entries, 'toxic')
    for i, name in enumerate(['RL', 'Adaptive']):
        axes[0].plot(d['hedge_fraction'][::20, i][:500], alpha=0.7, label=name)
        axes[1].plot(np.cumsum(d['pnl'][::20, i][:500]), label=name)
    axes[0].set(xlabel='Seed-0 holdout trajectory step', ylabel='Hedge fraction')
    axes[1].set(xlabel='Seed-0 holdout trajectory step', ylabel='Cumulative simulated PnL')
    for ax in axes:
        ax.legend()
    save(fig, 12, 'Perfect-information investor stress experiment')
    for risk, number in [(0, 13), (2, 14)]:
        fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        d = data(entries, 'adaptive_skew_risk' + str(risk))
        for key in ['bid_epsilon', 'ask_epsilon']:
            axes[0].plot(d[key][::20, 0][:200], label=key)
        axes[0].legend(fontsize=8)
        axes[1].plot(d['inventory'][::20, 0][:200])
        axes[0].set(ylabel='Adaptive epsilon')
        axes[1].set(xlabel='Seed-0 holdout trajectory step', ylabel='Inventory')
        save(fig, number, 'Adaptive pricing, risk=' + str(risk))


def frozen_hash_matches(name, expected):
    actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    if actual == expected:
        return True
    approved = json.loads((ROOT / 'documentation_translations.json').read_text()).get(name, {})
    return name.endswith('.md') and approved.get('original_sha256') == expected and approved.get('published_sha256') == actual


def build():
    base = ROOT / 'results/v4_study'
    selection = json.loads((base / 'preprocessing_selection.json').read_text())
    for phase, count in [('pilot', 10), ('confirm', 200), ('diagnostic', 10), ('variance_extension', 10), ('risk_sensitivity', 15)]:
        m = json.loads((base / phase / 'study_manifest.json').read_text())
        if m['state'] != 'complete' or m['errors'] or len(m['completed']) != count:
            raise ValueError('Incomplete phase ' + phase)
    snapshot = json.loads((ROOT / 'v4/protocol_snapshot.json').read_text())
    for name, expected in snapshot.items():
        if not frozen_hash_matches(name, expected):
            raise ValueError('Frozen source/protocol changed: ' + name)
    extension_snapshot = json.loads((ROOT / 'v4/variance_extension_snapshot.json').read_text())
    for name, expected in extension_snapshot.items():
        if not frozen_hash_matches(name, expected):
            raise ValueError('Extension source changed: ' + name)
    risk_snapshot = json.loads((ROOT / 'v4/risk_sensitivity_snapshot.json').read_text())
    for name, expected in risk_snapshot.items():
        if not frozen_hash_matches(name, expected):
            raise ValueError('Risk sensitivity source changed: ' + name)
    entries, diagnostics = (load_phase(base / 'confirm'), load_phase(base / 'diagnostic'))
    extension = load_phase(base / 'variance_extension')
    expected_cases = scenarios(Calibration.load(ROOT / 'data/btc_full_day/paper_protocol_calibration.json'))
    if set(entries) != set(expected_cases):
        raise ValueError('Scenario coverage differs from declared protocol')
    all_rows, effects, quote_rows, stable_count, trained_count = ([], [], [], 0, 0)
    values = {}
    for name, group in entries.items():
        if {x['result']['seed'] for x in group} != set(range(5)):
            raise ValueError('Missing seed ' + name)
        rr = [x['result'] for x in group]
        for item in group:
            r = item['result']
            if r['evaluation_samples'] != 32000:
                raise ValueError('Incomplete final evaluation')
            with np.load(item['_folder'] / 'evaluation.npz') as z:
                if any((not np.isfinite(z[k]).all() for k in z.files)):
                    raise ValueError('Nonfinite evaluation')
                np.testing.assert_allclose(z['pnl'], z['spread_pnl'] - z['hedge_cost'] + z['inventory_pnl'], atol=1e-08)
                np.testing.assert_allclose(z['pnl'].mean(0), r['pnl'], atol=1e-10)
            if r['trained']:
                trained_count += 1
                stable_count += bool(r['convergence']['empirical_checks_passed'])
                if item['normalization'] != selection['selected'] or item['reward_mode'] != 'total':
                    raise ValueError('Selected settings/reward changed in main experiment')
            row = dict(scenario=name, seed=r['seed'], trained=r['trained'], training_steps=r.get('training_steps', 0), own_pnl=r['pnl'][0], competitor_pnl=r['pnl'][1], own_spread_x100=100 * r['spread_pnl'][0], own_inventory_x100=100 * r['inventory_pnl'][0], own_hedge_pnl_x100=-100 * r['hedge_cost'][0], excess_pnl_x100=r['excess_pnl_x100'], inventory_mean=r['inventory'][0], inventory_std=r['inventory_std'][0], inventory_pnl_std=r['inventory_pnl_std'][0], bid_epsilon=r['bid_epsilon'][0], ask_epsilon=r['ask_epsilon'][0], bid_std=r['bid_std'][0], ask_std=r['ask_std'][0], hedge_fraction=r['hedge_fraction'][0], market_share=r['volume'][0] / 20, mean_customer_units_per_step=r['volume'][0], empirical_stability_passed=r.get('convergence', {}).get('empirical_checks_passed'))
            row.update({'excess_' + k + '_x100': v for k, v in r['excess_components_x100'].items()})
            all_rows.append(row)
        values[name] = rr
        band = interval([r['excess_pnl_x100'] for r in rr])
        effects.append(dict(scenario=name, excess_mean_x100=band['mean'], between_seed_ci_lower=band['lower'], between_seed_ci_upper=band['upper'], positive_seeds=sum((r['excess_pnl_x100'] > 0 for r in rr)), empirical_stable_seeds=sum((r.get('convergence', {}).get('empirical_checks_passed', False) for r in rr))))
    write_csv(base / 'all_seed_metrics.csv', all_rows)
    write_csv(base / 'scenario_effects.csv', effects)
    for table, names in [(1, ['competition_persistent_half', 'competition_persistent', 'competition_random_half', 'competition_random']), (2, [n for n in entries if n.startswith('competition_adaptive')]), (3, ['competition_persistent_half', 'competition_persistent', 'competition_random_half', 'competition_random']), (4, [n for n in entries if n.startswith('adaptive_vs_persistent')]), (5, [n for n in entries if n.startswith('adaptive_vs_random')]), (6, [n for n in entries if n.startswith('hedging_')])]:
        write_csv(base / 'table_{}_per_seed.csv'.format(table), [r for r in all_rows if r['scenario'] in names])
    behavior = {}
    behavior['simple_competitors'] = {n: interval([r['excess_pnl_x100'] for r in values[n]]) for n in ['competition_persistent_half', 'competition_persistent', 'competition_random_half', 'competition_random']}
    drift = {str(mu): [r['inventory'][0] for r in values['drift_' + str(mu)]] for mu in [-1.0, -0.01, 0.0, 0.01, 1.0]}
    means = [np.mean(drift[str(mu)]) for mu in [-1.0, -0.01, 0.0, 0.01, 1.0]]
    behavior['drift'] = dict(means=means, monotone=bool(np.all(np.diff(means) >= 0)), extreme_signs_correct=bool(means[0] < 0 < means[-1]), paired_extreme_difference=interval(np.array(drift['1.0']) - np.array(drift['-1.0'])))
    base_risk = values['risk_none']
    behavior['risk'] = {}
    for mode in ['stdev', 'square', 'asymmetric']:
        rr = values['risk_' + mode]
        behavior['risk'][mode] = {key: interval([a[key][0] - b[key][0] for a, b in zip(rr, base_risk)]) for key in ['inventory_std', 'inventory_pnl_std']}
        behavior['risk'][mode]['absolute_inventory_mean_change'] = interval([abs(a['inventory'][0]) - abs(b['inventory'][0]) for a, b in zip(rr, base_risk)])
        behavior['risk'][mode]['market_share'] = interval([a['volume'][0] / 20 for a in rr])
        behavior['risk'][mode]['base_market_share'] = interval([b['volume'][0] / 20 for b in base_risk])
    tox = values['toxic']
    behavior['toxic'] = dict(excess_pnl_x100=interval([r['excess_pnl_x100'] for r in tox]), hedge_fraction_difference=interval([r['hedge_fraction'][0] - r['hedge_fraction'][1] for r in tox]))
    for phase, collection in [('total_pnl', entries), ('spread_only_diagnostic', diagnostics), ('conditional_total_extension', extension)]:
        for name, a, b in [('competition_random', -1.0, 1.0), ('competition_random_half', -0.5, 0.5)]:
            optimum = max(a, (b - 1) / 2)
            optimal_value = (1 + optimum) * (b - optimum) / (b - a)
            for entry in collection[name]:
                with np.load(entry['_folder'] / 'evaluation.npz') as z:
                    eps = np.c_[z['bid_epsilon'][:, 0], z['ask_epsilon'][:, 0]]
                normalized_revenue = (1 + eps) * np.clip((b - eps) / (b - a), 0, 1) / optimal_value
                quote_rows.append(dict(phase=phase, scenario=name, seed=entry['result']['seed'], optimal_epsilon=optimum, mean_epsilon=float(eps.mean()), epsilon_std=float(eps.std()), normalized_spread_efficiency=float(normalized_revenue.mean())))
    write_csv(base / 'analytical_quote_diagnostic.csv', quote_rows)
    behavior['quote_diagnostic'] = {name: {phase: interval([r['normalized_spread_efficiency'] for r in quote_rows if r['scenario'] == name and r['phase'] == phase]) for phase in ['total_pnl', 'spread_only_diagnostic', 'conditional_total_extension']} for name in ['competition_random', 'competition_random_half']}
    behavior['variance_extension'] = {}
    variance_rows = []
    for name, group in extension.items():
        for entry in group:
            if not entry.get('extension', {}).get('final_evaluation_uses_realized_total_pnl'):
                raise ValueError('Extension evaluation reward is not verified')
        rr = [entry['result'] for entry in group]
        behavior['variance_extension'][name] = dict(realized_own_pnl=interval([r['pnl'][0] for r in rr]), realized_excess_pnl_x100=interval([r['excess_pnl_x100'] for r in rr]), inventory_std=interval([r['inventory_std'][0] for r in rr]), original_paper_training_implementation=False)
        for phase, group_to_check in [('original_total', entries[name]), ('conditional_extension', group)]:
            for entry in group_to_check:
                cfg = entry['config']
                with np.load(entry['_folder'] / 'evaluation.npz') as z:
                    conditional = z['spread_pnl'][:, 0] - z['hedge_cost'][:, 0]
                    conditional += z['inventory'][:, 0] * z['price'][:, 0] * np.expm1(cfg['drift'] * cfg['dt'])
                    realized_variance = float(z['pnl'][:, 0].var())
                    conditional_variance = float(conditional.var())
                variance_rows.append(dict(phase=phase, scenario=name, seed=entry['result']['seed'], realized_reward_variance=realized_variance, conditional_reward_variance=conditional_variance, variance_ratio=conditional_variance / realized_variance if realized_variance > 0 else None))
    write_csv(base / 'inventory_noise_decomposition.csv', variance_rows)
    skew = []
    for name in ['competition_persistent', 'competition_adaptive_ms0.5_risk2']:
        for entry in entries[name]:
            with np.load(entry['_folder'] / 'evaluation.npz') as z:
                inv = z['pre_inventory'][:, 0]
                short, long = (inv < 0, inv > 0)
                if short.sum() > 20 and long.sum() > 20:
                    skew.append(dict(scenario=name, seed=entry['result']['seed'], bid_long_minus_short=float(z['bid_epsilon'][long, 0].mean() - z['bid_epsilon'][short, 0].mean()), ask_long_minus_short=float(z['ask_epsilon'][long, 0].mean() - z['ask_epsilon'][short, 0].mean())))
    write_csv(base / 'inventory_skew.csv', skew)
    behavior['skew'] = {name: {key: interval([r[key] for r in skew if r['scenario'] == name]) for key in ['bid_long_minus_short', 'ask_long_minus_short']} for name in ['competition_persistent', 'competition_adaptive_ms0.5_risk2']}
    risk_rows = []
    behavior['risk_sensitivity'] = {}
    for name in ['risk_stdev', 'risk_square', 'risk_asymmetric']:
        summary = {}
        for group in ['no_penalty', 'original_alpha', 'scaled']:
            records = []
            for seed in range(5):
                folder = base / 'risk_sensitivity' / group / name / ('seed_' + str(seed))
                doc = json.loads((folder / 'result.json').read_text())
                r = doc['result']
                if r['evaluation_samples'] != 32000:
                    raise ValueError('Risk sensitivity evaluation incomplete')
                with np.load(folder / 'evaluation.npz') as z:
                    if any((not np.isfinite(z[k]).all() for k in z.files)):
                        raise ValueError('Non-finite risk sensitivity result')
                    np.testing.assert_allclose(z['pnl'], z['spread_pnl'] - z['hedge_cost'] + z['inventory_pnl'], atol=1e-08)
                row = dict(scenario=name, group=group, seed=seed, pnl=r['pnl'][0], market_share=r['volume'][0] / 20, inventory_std=r['inventory_std'][0], inventory_pnl_std=r['inventory_pnl_std'][0])
                risk_rows.append(row)
                records.append(row)
            summary[group] = {key: interval([r[key] for r in records]) for key in ['pnl', 'market_share', 'inventory_std', 'inventory_pnl_std']}
        behavior['risk_sensitivity'][name] = summary
    write_csv(base / 'risk_sensitivity_comparison.csv', risk_rows)
    (base / 'behavior_checks.json').write_text(json.dumps(behavior, indent=2))
    make_figures(entries, diagnostics, extension, base / 'figures', Calibration.load(ROOT / 'data/btc_full_day/paper_protocol_calibration.json'))
    status = dict(best_effort_scope_completed=True, pilot_runs=10, main_scenarios=40, main_runs=200, pricing_diagnostic_runs=10, variance_extension_runs=10, risk_sensitivity_training_runs=15, additional_frozen_risk_reference_evaluations=30, official_backend=True, selected_normalization=selection['selected'], test_used_for_selection=False, main_learning_runs=trained_count, empirical_stability_passed_runs=stable_count, expected_files_and_accounting_verified=True, frozen_protocol_unchanged=True, original_numeric_replication=False, author_exact_settings_known=False, not_exchange_fill_backtest=True)
    (base / 'verification.json').write_text(json.dumps(status, indent=2))
    lines = ['# Best-effort paper reproduction with BTC depth calibration', '', 'Uses official historical RLlib PPO and the disclosed price/spread laws; BTC replaces depth statistics only. Undisclosed author settings remain explicit assumptions. This is not exact numerical replication of the original MSFT study.', '', '## Completed experiments', '', '- Ten preprocessing pilot runs selected using validation only; 200 main runs (40 scenarios x five seeds); ten spread-only pricing diagnostics; ten conditional-reward extension runs; fifteen fixed risk-sensitivity training runs and thirty fresh evaluations of frozen reference policies.', '- Each main final evaluation contains 32,000 fresh simulated records. All seed summaries and training logs are included. Policy weights and per-step arrays are excluded from this public archive and can be regenerated. Failed stability checks remain reported.', '- Five-seed validation mean: raw inputs {:.6f}, MeanStdFilter {:.6f}; selected {} without using final test data.'.format(selection['scores']['none']['mean'], selection['scores']['meanstd']['mean'], selection['selected']), '- Empirical stability checks passed in {}/{} learning runs. These compare clipped actions and validation rewards and do not prove optimality; latent Gaussian KL is retained separately.'.format(stable_count, trained_count), '', '## Competitive returns', '', 'Own minus opponent mean PnL per step, multiplied by 100. Intervals use variation across five model seeds, not individual correlated time steps.', '', '|Opponent|Mean excess PnL x100|Across-seed 95% interval|Positive seeds|', '|---|---:|---|---:|']
    for n, band in behavior['simple_competitors'].items():
        e = next((r for r in effects if r['scenario'] == n))
        lines.append('|{}|{:.3f}|[{:.3f}, {:.3f}]|{}/5|'.format(n, band['mean'], band['lower'], band['upper'], e['positive_seeds']))
    lines.extend(['', '## Pricing diagnostics', '', 'The theoretical spread-revenue optimum is 0 against Uniform[-1,1] and -0.25 against Uniform[-0.5,0.5]. Normalized analytical quote efficiency divides expected spread revenue by its statewise optimum; 1 is optimal. This is not total PnL or live trading return.', '', '|Opponent|Main total-PnL reward|Spread-only diagnostic|Conditional-total extension|', '|---|---:|---:|---:|'])
    for n, pair in behavior['quote_diagnostic'].items():
        lines.append('|{}|{:.2%}|{:.2%}|{:.2%}|'.format(n, pair['total_pnl']['mean'], pair['spread_only_diagnostic']['mean'], pair['conditional_total_extension']['mean']))
    lines.extend(['', 'The conditional-total extension replaces only training inventory PnL with its GBM conditional mean. Final evaluation uses realized random total PnL. Expected total PnL is preserved for a fixed policy under independent investor flow without nonlinear risk penalties. This model-informed exploratory extension is not an author-confirmed method; toxic flow and risk penalties are rejected. See `../../v4/VARIANCE_EXTENSION.md`.', '', '|Opponent|Main policy realized test PnL|Conditional-total extension realized test PnL|', '|---|---:|---:|'])
    for name, result in behavior['variance_extension'].items():
        original_pnl = np.mean([r['pnl'][0] for r in values[name]])
        lines.append('|{}|{:.6f}|{:.6f}|'.format(name, original_pnl, result['realized_own_pnl']['mean']))
    lines.extend(['', 'See `inventory_noise_decomposition.csv` for realized versus conditional reward variance under the same frozen policy. Lower training reward variance does not by itself imply higher test returns.'])
    lines.extend(['', '## Drift and risk findings', '', '- Mean inventory for mu=-1,-0.01,0,0.01,1: ' + ', '.join(('{:.3f}'.format(x) for x in means)) + '.', '- Extreme-drift inventory signs correct: {}; all five means monotone: {}.'.format(behavior['drift']['extreme_signs_correct'], behavior['drift']['monotone']), '- The table reports standard-deviation changes relative to the no-penalty policy. Negative values indicate reductions. Market share must be considered alongside inventory: stopping trading is not sufficient evidence of successful inventory management.', '', '|Penalty|Inventory SD change|Inventory-PnL SD change|Mean market share|', '|---|---:|---:|---:|'])
    for mode, result in behavior['risk'].items():
        lines.append('|{}|{:.4f}|{:.4f}|{:.3%}|'.format(mode, result['inventory_std']['mean'], result['inventory_pnl_std']['mean'], result['market_share']['mean']))
    lines.extend(['', '### Risk-strength sensitivity (post hoc exploratory extension)', '', 'After the main policies largely stopped trading, a separate extension multiplied all three penalty coefficients by 0.1. Main results remain unchanged. Fresh common simulated streams compare frozen no-penalty, original-coefficient and scaled-coefficient policies; the first two groups are not retrained. This investigates BTC calibration scale sensitivity, not successful replication with the original coefficients.', '', '|Penalty|Model group|Market share|Inventory SD|Inventory-PnL SD|', '|---|---|---:|---:|---:|'])
    for name, summary in behavior['risk_sensitivity'].items():
        for group, r in summary.items():
            lines.append('|{}|{}|{:.3%}|{:.3f}|{:.3f}|'.format(name, group, r['market_share']['mean'], r['inventory_std']['mean'], r['inventory_pnl_std']['mean']))
    lines.extend(['', '- Toxic-flow case: mean excess PnL x100 {:.3f}; mean hedge-fraction difference {:.4f}. Positive point estimates alone do not establish a statistically supported joint claim of better returns and more aggressive hedging.'.format(behavior['toxic']['excess_pnl_x100']['mean'], behavior['toxic']['hedge_fraction_difference']['mean']), '', 'Full intervals and inventory-dependent quote skew appear in `behavior_checks.json` and `inventory_skew.csv`. Comparisons are exploratory, without multiplicity adjustment; five model seeds provide limited interval precision.', '', '## Figures and verification', '', 'Figure 1 separates main policies from pricing diagnostics. Figure 2 contains RL training curves only (opponent training rewards were not recorded). Figures 3 and 5 pool seeds descriptively. Figures 4 and 12-14 consistently use seed 0. Figures 6-7 show the BTC-calibrated reference market; figures 8-11 retain all five training seeds.', '', '![Quote distributions](figures/figure_01.png)', '', '![Risk distributions](figures/figure_05.png)', '', 'Per-seed counterparts of paper tables 1-6 are in `table_1_per_seed.csv` through `table_6_per_seed.csv`. See `all_seed_metrics.csv` for main outcomes and `scenario_effects.csv` for seed-level intervals. Model weights and per-step evaluation arrays are generated by rerunning the experiments and are not bundled in the public archive.', '', '## Scope and limitations', '', 'This reproduces a simulated dealer market with synthetic investor orders and best-quote execution. It does not model real exchange queues, cancellation latency, capital/margin constraints or market impact. BTC supplies depth statistics only. Normalized PnL is not a BTC live return. Undisclosed depth scaling, inventory bounds, preprocessing and software configuration remain assumptions. Modified-reward diagnostics remain separate from the main experiment.'])
    (base / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=status, behavior=behavior), indent=2))
if __name__ == '__main__':
    build()
