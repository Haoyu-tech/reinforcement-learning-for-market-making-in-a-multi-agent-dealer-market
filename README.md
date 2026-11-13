# Reinforcement Learning for Market Making in a Multi-agent Dealer Market

An independent, best-effort reproduction of [Ganesh et al. (2019), arXiv:1911.05892](https://arxiv.org/abs/1911.05892), using BTC order-book depth statistics in place of the original MSFT calibration.

**Status: completed experimental suite; partial behavioral reproduction, not exact numerical replication.** Author source code and complete training settings were unavailable. The historical software stack is a documented reference choice, not a recovered author environment.

## Results

- Completed 200 main runs: 40 scenarios with five seeds each. Each final evaluation uses 32,000 fresh simulated records.
- Spread-only pricing diagnostics reached approximately 97.4% and 97.6% of theoretical spread-revenue efficiency against the two random opponents. These use a different reward and are reported separately from the main total-PnL policies.
- Main learning was unstable: only 1 of 100 learning runs passed the declared empirical stability diagnostic. Failures are retained.
- Strong positive/negative drift produced inventory in the expected direction, but the five drift means were not all monotonic.
- Original risk coefficients mostly stopped trading. A separately labeled post hoc sensitivity with coefficients multiplied by 0.1 restored some trading while reducing inventory variability.
- A conditional-reward extension improved analytical quote efficiency but did not improve mean realized own PnL in either random-opponent comparison.

See the [full English report](results/v4_study/REPORT.md), [per-seed main metrics](results/v4_study/all_seed_metrics.csv), and [verification record](results/v4_study/verification.json). Confidence intervals summarize variation across five model seeds; comparisons are exploratory and have no multiplicity correction.

## What is being reproduced?

The simulator generates synthetic investor orders, dealer quotes, hedging and GBM prices according to the disclosed paper mechanisms. BTC supplies depth-distribution calibration only. This is not a chronological exchange-fill backtest, and normalized PnL is not live BTC profitability. Exchange queues, cancellation delays, margin and market impact are not modeled.

The calibration reconstructed 130,007,290 BTC depth deltas from September 2, 2026, sampled 2,880 books at 30-second intervals, and fitted ten 1-basis-point bins on each side. Included fitted parameters allow the dealer-market experiments to run without the private raw catalog. Raw market data and the original paper PDF are not redistributed. Recreating the calibration from raw data requires a compatible external catalog; see the extraction and fitting scripts in v3/.

## Run

Training uses Docker with Python 3.7.17, TensorFlow 1.15.0 and official Ray/RLlib 0.7.6. The legacy image needs network access during build; experiment containers can run without network. Use Docker Desktop with Linux containers on Windows. The complete suite is computationally expensive. Adjust worker counts to available memory and CPU.

```powershell
docker build -t hft-ganesh-rllib-2019:local -f v3/Dockerfile v3
docker run --rm --network none --mount "type=bind,source=${PWD},target=/work" hft-ganesh-rllib-2019:local python v3/test_official.py
docker run --rm --network none --mount "type=bind,source=${PWD},target=/work" hft-ganesh-rllib-2019:local python v4/test_experiment.py
docker run --rm --network none --mount "type=bind,source=${PWD},target=/work" hft-ganesh-rllib-2019:local python v4/test_variance_extension.py
```

Per-run JSON and CSV logs are bundled in `results/v4_study/run_logs.tar.xz`. Keep this archive separate when starting a fresh run: extracted result.json files trigger resume behavior but do not include the weights and per-step arrays needed for a complete resumed study. Root-level published summaries can be overwritten by a fresh run.

```powershell
python -m pip install -r requirements-report.txt
./run_best_effort.ps1
```

The orchestrator runs preprocessing pilots, main experiments, pricing diagnostics, conditional-reward and risk-sensitivity extensions, then verifies results and generates the report. Host reporting uses a modern Python environment; the pinned historical dependencies belong inside Docker only. A freshly built image can have a different image ID even when using the same Dockerfile; v3/image_id.txt records the original experimental image, not a portable image download.

## Repository layout and provenance

|Path|Purpose|
|---|---|
|v2/market.py|Dealer-market mechanisms and calibrated depth distribution|
|v3/|Official historical RLlib integration, paper scenarios and calibration tools|
|v4/|Validation-only selection, main suite, diagnostics, extensions and report generator|
|data/btc_full_day/|Fitted calibration and fit/verification summaries|
|results/v4_study/|All-seed summaries, run logs, 14 figures and English report|
|sources/ray_0_7_6/|Historical upstream source excerpts and upstream license|

Scientific source and calibration bytes are unchanged from the completed local study. Original frozen hashes are retained. English protocol translations are recorded in documentation_translations.json; the reporting verifier accepts only those exact documentation translations in addition to original hashes. Report text was translated separately. Recorded verification describes the completed local study; raw arrays and weights are not included in this curated public archive. Earlier local revisions are outside this publication.

Protocol: [main study](v4/PROTOCOL.md), [conditional-reward extension](v4/VARIANCE_EXTENSION.md), [risk sensitivity](v4/RISK_SENSITIVITY.md). Third-party terms are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

