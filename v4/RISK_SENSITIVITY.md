# Risk-strength sensitivity: exploratory extension

The paper's coefficients 0.4, 0.07 and 0.3 cause almost no trading under this BTC depth calibration. Lower inventory obtained by stopping trading does not establish inventory management while providing liquidity. Preserve the original coefficient results.

After observing this failure, fix one exploratory alternative: multiply all three coefficients by 0.1, giving 0.04, 0.007 and 0.03. Leave the market, algorithm, observations and training budget unchanged. Run five seeds per penalty for 1,024,000 steps each. Do not repeatedly reduce coefficients or select a favorable test checkpoint.

Evaluate three frozen policy groups on common fresh simulated market seeds 19100000 through 19100004: no penalty, original coefficient, and scaled coefficient. The first two groups are not retrained. Report market share, inventory standard deviation, inventory-PnL standard deviation and total PnL together.

This probes reward-scale sensitivity after replacing depth distributions. It is not numerical replication of the original coefficient results. These fresh simulated streams are not new held-out real BTC data.
