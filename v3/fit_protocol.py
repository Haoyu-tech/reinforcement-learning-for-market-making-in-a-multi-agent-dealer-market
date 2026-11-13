"""Replace ONLY the appendix's depth calibration; retain disclosed price laws."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import gamma
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'v2'))
from market import Calibration
from dataclasses import asdict


def fit(folder):
    path = folder/'full_depth_1bp.npz'
    with np.load(path) as z:
        depth = z['depth_btc'].reshape(-1, 10)
    if not np.isfinite(depth).all() or np.any(depth < 0) or np.any(depth[:, 0] <= 0):
        raise ValueError('Invalid reconstructed bucket depth')
    best = depth[:, 0]
    normalized = best/best.mean()
    shape, loc, scale = gamma.fit(normalized, floc=0)
    # Mean-one Gamma: retain fitted shape, rescale to mean 1 explicitly.
    n = np.arange(1, 10)
    design = np.c_[n, n*n]
    targets = depth/depth[:, :1]
    coefficients = np.linalg.lstsq(design, (targets[:, 1:]-1).T, rcond=None)[0].T
    selected = np.linspace(0, len(coefficients)-1, min(2048, len(coefficients)), dtype=int)
    retained = coefficients[selected]
    rng = np.random.default_rng(191105892)
    best_draw = rng.gamma(shape, 1/shape, 100000)
    coeff_draw = retained[rng.integers(len(retained), size=100000)]
    levels = np.arange(10)
    drawn_shape = np.maximum(.001, 1+coeff_draw[:, :1]*levels+coeff_draw[:, 1:]*levels**2)
    totals_unscaled = best_draw*drawn_shape.sum(1)
    # Author lambda is undisclosed. Predeclare coverage criterion, do not choose
    # lambda based on trading PnL or agreement with the original tables.
    lam = float(1/np.quantile(totals_unscaled, .05))
    cal = Calibration(provenance='BTC_2026-09-02_full_day_full_L2_1bp_buckets',
        gamma_shape=float(shape), depth_scale=lam,
        shape_coeff_mean=coefficients.mean(0).tolist(), shape_coeff_std=coefficients.std(0).tolist(),
        shape_samples=retained.tolist())
    # Deliberately keep Calibration's original spread mean/std/clamps and 1bp
    # spacing, and Config's disclosed sigma=.1. BTC trade-size normalization,
    # empirical BTC spreads and fitted BTC price sigma are NOT used here.
    (folder/'paper_protocol_calibration.json').write_text(json.dumps(asdict(cal), indent=2), encoding='utf-8')
    cumulative = lam*best_draw[:, None]*np.cumsum(drawn_shape, axis=1)
    hit = np.minimum((cumulative < 1).sum(1), 9)
    record = dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        independent_gamma_shape_sampling=True, gamma_fit='MLE with loc=0, then normalize mean to 1',
        mle_shape=float(shape), mle_scale_before_normalization=float(scale),
        lambda_value=lam, lambda_rule='95% of sampled reference books cover a unit order within ten levels',
        unit_order_deepest_level_counts=np.bincount(hit, minlength=10).tolist(),
        unit_order_extrapolation_fraction=float(np.mean(cumulative[:, -1] < 1)),
        original_spread_law_preserved=True, original_level_spacing_preserved=True,
        original_default_price_sigma_preserved=True,
        original_author_lambda_available=False, original_author_RLlib_version_available=False,
        strict_original_replication=False)
    (folder/'paper_protocol_fit.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('folder', type=Path)
    fit(p.parse_args().folder)
