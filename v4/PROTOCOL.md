# Best-effort reproduction protocol

The study retains the paper's disclosed price and spread laws and official historical RLlib PPO. BTC replaces depth statistics only. Unknown author settings are explicit assumptions, not recovered author configuration.

## Fixed sequence

1. Compare no normalization with official MeanStdFilter: five seeds each, 512,000 training steps each. Select the higher five-seed mean of each seed's last three validation scores. Do not use final test data for selection.
2. Run 40 main scenarios with five seeds each. Learning policies train for at least 256,000 and at most 1,024,000 steps, checking every 128,000. Rule-based baselines do not train. Evaluate on 32,000 fresh simulated records per seed.
3. Train separate spread-only pricing diagnostics against Uniform[-1,1] and Uniform[-0.5,0.5], five seeds each. Their theoretical optimal quote parameters are 0 and -0.25. These diagnostics do not replace the main total-PnL experiment.

The main rewards, opponent settings, original risk coefficients, optimizer and network remain fixed. The selected normalization is not an author-confirmed setting. This is a simulated dealer market, not an exchange matching-engine backtest.

## Empirical stability

Keep 500 raw observations from the initial training sample as fixed probes. Compare executable actions using common Gaussian noise and actual clipping boundaries, normalized by action range. Retain the latent Gaussian KL separately: latent means can change while clipped actions remain identical.

Validation uses 20 independent world paths. Compute paired mean reward changes and standard errors across paths, retaining within-path temporal dependence. Stability requires maximum action MAE below 0.02, action difference 95th percentile below 0.1, and reward change within 1.96 paired standard errors. Require three consecutive passes after the minimum budget. This is a numerical diagnostic, not proof of optimality; failed cases at the maximum budget remain in the results.

## Leakage controls

Update MeanStdFilter only during training; freeze it for validation and test and verify its count does not change. Store filter statistics with policies. Final test seeds are separate from validation and earlier revisions. Report every seed without choosing the best test checkpoint or favorable scenario.

See the root README for execution commands. Original protocol hashes are retained in protocol_snapshot.json; documentation_translations.json records this English translation separately from the unchanged scientific source hashes.
