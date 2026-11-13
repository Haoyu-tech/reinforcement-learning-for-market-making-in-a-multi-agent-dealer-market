# Conditional-reward variance reduction: exploratory extension

This extension was introduced after observing training noise. It does not replace the main original-reward results and is not an author-confirmed implementation. Reuse the validation-selected preprocessing without selecting another normalization, best seed or test checkpoint.

With independent investor flow and GBM prices, after the current action and client trades, inventory z and price P are known while the next price shock is not:

`E[z*(P_next-P) | current state, action, client trades] = z*P*(exp(mu*dt)-1)`.

Replace only the inventory-PnL term of the training reward with this conditional mean. Iterated expectations preserve expected finite-horizon discounted total PnL for any fixed adapted policy. Actual price paths and the observed previous realized inventory PnL remain unchanged. Future prices are not network inputs.

This uses knowledge of the simulator distribution. The implementation rejects toxic flow, which depends on future price, and nonlinear risk penalties. Run only the two random opponents with five seeds each, using official PPO. Final evaluation uses realized random total PnL, not the conditional training reward.

Keep confirm/, diagnostic/ and variance_extension/ results separate. Better analytical quote efficiency does not establish better realized test PnL or strict recovery of the paper's training implementation.
