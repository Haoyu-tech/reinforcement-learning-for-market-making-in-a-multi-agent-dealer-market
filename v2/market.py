"""Vectorized dealer market. Equations: Ganesh et al., sections 3,4 and 7.2.

Unknown author choices are configuration, never silently called original data.
Only public reference curves and each dealer's own executed trades reach policies.
"""
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
import numpy as np


@dataclass
class Calibration:
    provenance: str = 'synthetic_assumption_not_original_MSFT'
    gamma_shape: float = 2.
    depth_scale: float = .2
    # Independent random quadratic shapes, with shape(0)=1.
    shape_coeff_mean: list = field(default_factory=lambda: [.15, .01])
    shape_coeff_std: list = field(default_factory=lambda: [.03, .003])
    spread_mean: float = .0001
    spread_std: float = .00005
    spread_min: float = .00001
    spread_max: float = .00025
    level_step: float = .0001
    shape_samples: list = field(default_factory=list)
    unit_btc: float = 0.
    price_anchor_usdt: float = 0.
    fitted_sigma: float = .1
    market_dt: float = 1/(252*26)

    @classmethod
    def load(cls, path=None):
        result = cls(**json.loads(Path(path).read_text(encoding='utf-8'))) if path else cls()
        if result.gamma_shape <= 0 or result.depth_scale <= 0:
            raise ValueError('Depth Gamma shape and scale must be positive')
        return result


@dataclass
class Agent:
    kind: str = 'persistent'
    bid: float = 0.
    ask: float = 0.
    hedge: float = .2
    low: float = -1.
    high: float = 1.
    risk: float = 2.
    target: float = .5
    hedge_policy: str = 'adaptive'
    ignore_internalization: bool = False


@dataclass
class Config:
    agents: list = field(default_factory=lambda: [Agent('rl'), Agent('random')])
    drift: float = 0.
    sigma: float = .1
    # Annualization is not provided by the paper. Explicit trading-year convention.
    dt: float = 1/(252*26)
    initial_price: float = 100.
    horizon: int = 200
    investors: int = 20
    penalty: str = 'none'
    toxic: bool = False
    free_hedge: bool = False
    spread_multiplier: float = 1.
    grid_step: float = .1
    taker_fee: float = 0.
    calibration: Calibration = field(default_factory=Calibration)


class ResponseTable:
    """Observed-cell EW moments. Optional warmup uses actual own trades only.

    A nearest-observed-cell interpolation prevents zero/unobserved cells being
    mistaken for known zero-flow cells. Interpolation and warmup are assumptions.
    """
    def __init__(self, batch, step=.1):
        self.grid = np.round(np.arange(-1., 1.+step/2, step), 8)
        self.g = len(self.grid)
        self.batch = batch
        self.moments = np.zeros((batch, self.g*self.g, 5))
        self.visits = np.zeros((batch, self.g*self.g), dtype=int)
        b, a = np.meshgrid(self.grid, self.grid, indexing='ij')
        self.coords = np.stack([b.ravel(), a.ravel()], axis=-1)
        self.lookup = np.tile(np.arange(self.g*self.g), (batch, 1))

    def update(self, actions, volume, flow, normalized_pnl):
        idx = np.argmin(np.abs(actions[:, 0, None]-self.grid), axis=1)*self.g
        idx += np.argmin(np.abs(actions[:, 1, None]-self.grid), axis=1)
        rows = np.arange(self.batch)
        sample = np.stack([volume, flow, flow**2, normalized_pnl, normalized_pnl**2], axis=-1)
        old = self.moments[rows, idx]
        first = self.visits[rows, idx] == 0
        self.moments[rows, idx] = np.where(first[:, None], sample, .65*old+.35*sample)
        self.visits[rows, idx] += 1

    def interpolated(self):
        # Rebuild cheaply only when observed support changes; deterministic ties.
        support = self.visits > 0
        if not hasattr(self, '_support') or not np.array_equal(support, self._support):
            d = ((self.coords[:, None]-self.coords[None])**2).sum(-1)
            for b in range(self.batch):
                known = np.flatnonzero(support[b])
                if known.size:
                    self.lookup[b] = known[np.argmin(d[:, known], axis=1)]
            self._support = support.copy()
        return self.moments[np.arange(self.batch)[:, None], self.lookup].reshape(self.batch, self.g, self.g, 5)

    def action(self, env, agent_index, spec):
        tab, grid, g = self.interpolated(), self.grid, self.g
        rows = np.arange(self.batch)
        # Market share requires GROSS volume, not signed net flow (notation
        # v_epsilon is overloaded in paper section 4.1).
        gross = tab[:, np.arange(g), np.arange(g), 0]
        costs = np.abs(spec.target-gross/env.cfg.investors)
        eligible = costs <= costs.min(1)[:, None]+.05
        center = np.where(eligible, np.arange(g), -1).max(1)
        z = env.inventory[:, agent_index]
        # Candidates decrease bid when short, ask when long. Hold symmetric
        # quote at exactly zero inventory; paper does not specify tie handling.
        ib = np.broadcast_to(center[:, None], (self.batch, g)).copy()
        ia = ib.copy()
        ib[z < 0] = np.arange(g)
        ia[z > 0] = np.arange(g)
        moments = tab[rows[:, None], ib, ia]
        f, f2, s, s2 = [moments[..., i] for i in [1, 2, 3, 4]]
        sigma = env.cfg.sigma*np.sqrt(env.cfg.dt)*env.price
        sref = env.offsets[:, 0]
        variance = np.maximum(0., s2-s*s)*sref[:, None]**2
        variance += sigma[:, None]**2*np.maximum(0., z[:, None]**2+2*z[:, None]*f+f2)
        objective = -sref[:, None]*s+spec.risk*np.sqrt(variance)
        objective[np.arange(g)[None, :] > center[:, None]] = np.inf
        best = objective.argmin(1)
        ib, ia = ib[rows, best], ia[rows, best]
        selected = tab[rows, ib, ia]
        f, f2 = selected[:, 1], selected[:, 2]
        if spec.ignore_internalization:
            f, f2 = np.zeros_like(f), np.zeros_like(f2)
        xs = np.linspace(0., 1., 101)
        post = z[:, None]*(1-xs)
        risk = sigma[:, None]*np.sqrt(np.maximum(0., post**2+2*post*f[:, None]+f2[:, None]))
        hedge_cost = env.trade_cost(np.abs(z[:, None]*xs))
        if env.cfg.free_hedge:
            hedge_cost *= 0
        hedge = xs[(hedge_cost+spec.risk*risk).argmin(1)]
        if spec.hedge_policy == 'persistent':
            hedge[:] = spec.hedge
        elif spec.hedge_policy == 'random':
            hedge = env.policy_rng.uniform(size=self.batch)
        return np.stack([grid[ib], grid[ia], hedge], -1)


class DealerMarket:
    def __init__(self, cfg=None, batch=1, seed=0):
        self.cfg = cfg or Config()
        if self.cfg.penalty not in ['none', 'stdev', 'square', 'asymmetric']:
            raise ValueError('Unknown reward penalty')
        self.batch, self.n = batch, len(self.cfg.agents)
        # Environment randomness is isolated from policy/exploration randomness.
        streams = np.random.SeedSequence(seed).spawn(4)
        self.price_rng, self.flow_rng, self.curve_rng, self.policy_rng = [np.random.default_rng(s) for s in streams]
        self.tables = {i: ResponseTable(batch, self.cfg.grid_step)
                       for i, a in enumerate(self.cfg.agents) if a.kind == 'adaptive'}
        self.reset()

    def curve(self):
        c = self.cfg.calibration
        s0 = np.clip(self.curve_rng.normal(c.spread_mean, c.spread_std, self.batch), c.spread_min, c.spread_max)
        self.offsets = self.price[:, None]*(s0[:, None]/2+np.arange(10)*c.level_step)*self.cfg.spread_multiplier
        best = self.curve_rng.gamma(c.gamma_shape, 1/c.gamma_shape, self.batch)
        if c.shape_samples:
            coefficients = np.asarray(c.shape_samples)[self.curve_rng.integers(len(c.shape_samples), size=self.batch)]
        else:
            coefficients = self.curve_rng.normal(c.shape_coeff_mean, c.shape_coeff_std, (self.batch, 2))
        n = np.arange(10)
        shape = np.maximum(.001, 1+coefficients[:, :1]*n+coefficients[:, 1:]*n*n)
        self.depth = c.depth_scale*best[:, None]*shape
        self.before_depth = np.c_[np.zeros(self.batch), np.cumsum(self.depth, axis=1)[:, :-1]]
        self.total_depth = self.depth.sum(1)

    def trade_cost(self, size, sides=None):
        """Absolute slippage cost; last-level extrapolation is explicit assumption."""
        size = np.asarray(size, dtype=float)
        if size.ndim == 1:
            size = size[:, None]
        def compute(depth, offsets):
            before = np.c_[np.zeros(self.batch), np.cumsum(depth, axis=1)[:, :-1]]
            fills = np.minimum(np.maximum(size[..., None]-before[:, None], 0), depth[:, None])
            cost = np.sum(fills*offsets[:, None], -1)
            cost += np.maximum(0., size-depth.sum(1)[:, None])*offsets[:, -1, None]
            return cost
        if sides is not None and hasattr(self, 'bid_depth'):
            return np.where(np.asarray(sides) == 0, compute(self.bid_depth, self.bid_offsets), compute(self.ask_depth, self.ask_offsets))
        return compute(self.depth, self.offsets)

    def next_price(self):
        return self.price*np.exp((self.cfg.drift-.5*self.cfg.sigma**2)*self.cfg.dt
                                +self.cfg.sigma*np.sqrt(self.cfg.dt)*self.price_rng.normal(size=self.batch))

    def reset(self):
        self.price = np.full(self.batch, self.cfg.initial_price, dtype=float)
        self.inventory = np.zeros((self.batch, self.n))
        self.cash = np.zeros_like(self.inventory)
        self.trades = np.zeros((self.batch, self.n, self.cfg.investors))
        self.last_inv_pnl = np.zeros_like(self.inventory)
        self.history = np.zeros((self.batch, self.n, 10))
        self.t = 0
        self.curve()
        return self.observation()

    def observation(self, dealer=0):
        # Full sufficient piecewise-linear curve is encoded by all 10 sizes and
        # offsets, twice for bid/ask; both sides are symmetric as in the paper.
        bid_curve = np.c_[getattr(self, 'bid_depth', self.depth), getattr(self, 'bid_offsets', self.offsets)/self.price[:, None]*1e4]
        ask_curve = np.c_[getattr(self, 'ask_depth', self.depth), getattr(self, 'ask_offsets', self.offsets)/self.price[:, None]*1e4]
        scalar = np.c_[self.inventory[:, dealer]/20, np.log(self.price/self.cfg.initial_price),
                       np.abs(self.trades[:, dealer]).sum(1)/self.cfg.investors,
                       self.last_inv_pnl[:, dealer]]
        return np.c_[self.trades[:, dealer], scalar, bid_curve, ask_curve].astype(np.float32)

    def actions(self, learner=None, explore=False):
        actions = np.zeros((self.batch, self.n, 3))
        for i, a in enumerate(self.cfg.agents):
            if a.kind == 'rl':
                if learner is None:
                    actions[:, i, :2] = self.policy_rng.uniform(-1, 1, (self.batch, 2))
                    actions[:, i, 2] = self.policy_rng.uniform(size=self.batch)
                else:
                    actions[:, i] = learner
            elif a.kind == 'persistent':
                actions[:, i] = [a.bid, a.ask, a.hedge]
            elif a.kind == 'random':
                actions[:, i, :2] = self.policy_rng.uniform(a.low, a.high, (self.batch, 2))
                actions[:, i, 2] = self.policy_rng.uniform(size=self.batch)
            elif a.kind == 'adaptive':
                if explore:
                    grid = self.tables[i].grid
                    actions[:, i, :2] = self.policy_rng.choice(grid, size=(self.batch, 2))
                    actions[:, i, 2] = self.policy_rng.uniform(size=self.batch)
                else:
                    actions[:, i] = self.tables[i].action(self, i, a)
            else:
                raise ValueError(a.kind)
        return actions

    def step(self, actions=None, learner=None, explore=False):
        acts = self.actions(learner, explore) if actions is None else np.array(actions, dtype=float, copy=True)
        if acts.shape != (self.batch, self.n, 3) or not np.isfinite(acts).all():
            raise ValueError('Invalid actions')
        acts[:, :, :2] = np.clip(acts[:, :, :2], -1, 1)
        acts[:, :, 2] = np.clip(acts[:, :, 2], 0, 1)
        pre_inventory = self.inventory.copy()
        old_wealth = self.cash+self.inventory*self.price[:, None]
        hedge = self.inventory*acts[:, :, 2]
        hedge_cost = self.trade_cost(np.abs(hedge), np.where(hedge >= 0, 0, 1))
        hedge_cost += np.abs(hedge)*self.price[:, None]*self.cfg.taker_fee
        if self.cfg.free_hedge:
            hedge_cost *= 0
        self.inventory -= hedge
        self.cash += hedge*self.price[:, None]-hedge_cost
        next_price = self.next_price()
        directions = self.flow_rng.choice([-1, 1], (self.batch, self.cfg.investors))
        if self.cfg.toxic:
            # Perfectly informed orders: new dealer flow loses at next mark.
            # No artificial sign flip of total held-inventory PnL is applied.
            directions[:] = np.where(next_price[:, None] >= self.price[:, None], -1, 1)
        self.trades.fill(0.)
        for side, direction in enumerate([1, -1]):
            quotes = acts[:, :, side]
            # Random priorities give uniform allocation for any number of ties.
            priorities = self.flow_rng.uniform(size=(self.batch, self.cfg.investors, self.n))
            priorities = np.where(quotes[:, None] == quotes.min(1)[:, None, None], priorities, -1.)
            winners = priorities.argmax(-1)
            for i in range(self.n):
                mask = (directions == direction) & (winners == i)
                self.trades[:, i] += mask*direction
        flow = self.trades.sum(-1)
        volume = np.abs(self.trades).sum(-1)
        unit_bid_spread = self.trade_cost(np.ones((self.batch, 1)), np.zeros((self.batch, 1)))[:, 0]
        unit_ask_spread = self.trade_cost(np.ones((self.batch, 1)), np.ones((self.batch, 1)))[:, 0]
        buys = (self.trades > 0).sum(-1)
        sells = (self.trades < 0).sum(-1)
        spread_pnl = unit_bid_spread[:, None]*buys*(1+acts[:, :, 0])+unit_ask_spread[:, None]*sells*(1+acts[:, :, 1])
        self.cash += -flow*self.price[:, None]+spread_pnl
        self.inventory += flow
        inv_pnl = self.inventory*(next_price-self.price)[:, None]
        pnl = spread_pnl-hedge_cost+inv_pnl
        # Independent cash ledger check; do not merely test the same formula.
        wealth_change = self.cash+self.inventory*next_price[:, None]-old_wealth
        if not np.allclose(wealth_change, pnl, atol=1e-8, rtol=1e-8):
            raise ArithmeticError('Cash/inventory ledger does not reconcile')
        for i, table in self.tables.items():
            table.update(acts[:, i], volume[:, i], flow[:, i], spread_pnl[:, i]/self.offsets[:, 0])
        self.history[:, :, self.t % 10] = inv_pnl
        count = min(10, self.t+1)
        penalty = np.zeros_like(pnl)
        if self.cfg.penalty == 'stdev':
            penalty = -.4*self.history[:, :, :count].std(-1)
        elif self.cfg.penalty == 'square':
            penalty = -.07*inv_pnl**2
        elif self.cfg.penalty == 'asymmetric':
            penalty = .3*np.minimum(0., inv_pnl)
        info = dict(pnl=pnl, spread_pnl=spread_pnl, hedge_cost=hedge_cost,
                    inventory_pnl=inv_pnl, reward=pnl+penalty, inventory=self.inventory.copy(),
                    pre_inventory=pre_inventory, bid_epsilon=acts[:, :, 0], ask_epsilon=acts[:, :, 1],
                    hedge_fraction=acts[:, :, 2], volume=volume, flow=flow,
                    hedge_volume=np.abs(hedge), price=np.repeat(self.price[:, None], self.n, axis=1))
        self.price, self.last_inv_pnl = next_price, inv_pnl
        self.t += 1
        done = self.t >= self.cfg.horizon
        self.curve()
        return self.observation(), info['reward'][:, 0].copy(), done, info

    def warmup(self, steps=512):
        """Empirical-table initialization; consumes actual trades, never rival quotes."""
        if self.tables:
            for _ in range(steps):
                _, _, done, _ = self.step(explore=True)
                if done:
                    self.reset()
        self.reset()


def config_dict(cfg):
    return asdict(cfg)


class HistoricalMarket(DealerMarket):
    """BTC reference-tape environment; synthetic clients still use dealer matching.

    Training samples contiguous windows wholly inside the calibration partition.
    Sequential evaluation walks the test partition once, without random wrapping.
    Both real bid and ask depth curves are retained, including their asymmetry.
    """
    def __init__(self, cfg, tape, batch=1, seed=0, sequential=False):
        with np.load(tape) as z:
            self.tape_times, self.tape_book = z['ts_ns'].copy(), z['book'].copy()
        if len(self.tape_times) < cfg.horizon+2:
            raise ValueError('Reference tape shorter than episode horizon')
        if cfg.calibration.unit_btc <= 0 or cfg.calibration.price_anchor_usdt <= 0:
            raise ValueError('BTC unit and price anchor must be calibrated first')
        if sequential and batch != 1:
            raise ValueError('Chronological evaluation requires batch=1')
        self.sequential = sequential
        self.positions = np.zeros(batch, dtype=int)
        self.started = False
        super().__init__(cfg, batch, seed)

    def reset(self):
        if not self.sequential:
            self.positions = self.policy_rng.integers(0, len(self.tape_times)-self.cfg.horizon-1, self.batch)
        elif not self.started:
            self.positions.fill(0)
        self.started = True
        return super().reset()

    def curve(self):
        book = self.tape_book[self.positions]
        anchor = self.cfg.calibration.price_anchor_usdt
        unit = self.cfg.calibration.unit_btc
        mid = (book[:, 0, 0]+book[:, 0, 2])/2
        self.price = mid/anchor*100
        self.bid_depth = book[:, :, 3]/unit
        self.ask_depth = book[:, :, 1]/unit
        self.bid_offsets = (mid[:, None]-book[:, :, 2])/anchor*100
        self.ask_offsets = (book[:, :, 0]-mid[:, None])/anchor*100
        # Symmetric summaries are used only by the paper's adaptive heuristic;
        # actual trading costs and network observations retain separate sides.
        self.depth = (self.bid_depth+self.ask_depth)/2
        self.offsets = (self.bid_offsets+self.ask_offsets)/2
        self.before_depth = np.c_[np.zeros(self.batch), np.cumsum(self.depth, axis=1)[:, :-1]]
        self.total_depth = self.depth.sum(1)

    def next_price(self):
        self.positions += 1
        if np.any(self.positions >= len(self.tape_times)):
            raise IndexError('No future reference price available; tape never wraps')
        book = self.tape_book[self.positions]
        return (book[:, 0, 0]+book[:, 0, 2])/2/self.cfg.calibration.price_anchor_usdt*100
