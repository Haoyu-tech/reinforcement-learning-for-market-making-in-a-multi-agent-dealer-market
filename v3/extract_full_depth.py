"""Read-only vectorized native-L2 reconstruction, sampled into ten 1bp bins.

Uses the previously verified catalog; reconstructs ALL updates in timestamp order.
Sampling observes only updates at/before each target time. Last write per price
is selected explicitly, not by ambiguous duplicate NumPy indexed assignment.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq


def raw(col):
    if hasattr(col, 'combine_chunks'):
        col = col.combine_chunks()
    return np.frombuffer(col.buffers()[1], dtype='<i8', offset=col.offset*8, count=len(col)).copy()


def extract(session, output, cadence):
    conversion = json.loads((session/'conversion.json').read_text())
    verified = json.loads((session/'verification.json').read_text())
    if conversion['state'] != 'converted' or verified['result'] != 'passed':
        raise ValueError('Requires previously verified native catalog')
    root = session/'catalog/data'
    quotes = []
    for p in sorted((root/'quote_tick/BTCUSDT-PERP.BINANCE').glob('*.parquet')):
        t = pq.read_table(p, columns=['bid_price', 'ask_price', 'bid_size', 'ask_size', 'ts_init'])
        quotes.append(np.column_stack([t['ts_init'].to_numpy().astype(np.int64), raw(t['bid_price']), raw(t['ask_price']), raw(t['bid_size']), raw(t['ask_size'])]))
    q = np.concatenate(quotes).astype(np.int64)
    if np.any(np.diff(q[:, 0]) <= 0):
        raise ValueError('Nonmonotone quote catalog')
    target = int(q[0, 0])
    step = int(cadence*1e9)
    # Fixed, documented price band; fail if any sampled BBO/bin exceeds it.
    tick_raw, upper_usdt = 100000000, 200000
    bid = np.zeros(upper_usdt*10+1, dtype=np.int64)
    ask = np.zeros_like(bid)
    times, depths, mids, bbo_checks = [], [], [], 0
    rows = 0
    hashes = []
    def apply(actions, sides, prices, sizes):
        clears = np.flatnonzero(actions == 4)
        if clears.size:
            bid.fill(0)
            ask.fill(0)
            start = clears[-1]+1
            actions, sides, prices, sizes = [x[start:] for x in [actions, sides, prices, sizes]]
        live = (sides == 1) | (sides == 2)
        if np.any(prices[live] % tick_raw):
            raise ValueError('Price not on 0.1 USDT grid; refusing rounding')
        indices = prices//tick_raw
        for side, state in [(1, bid), (2, ask)]:
            mask = (sides == side) & (indices >= 0) & (indices < len(state))
            ii, aa, qq = indices[mask], actions[mask], sizes[mask]
            if len(ii):
                if not np.isin(aa, [1, 2, 3]).all():
                    raise ValueError('Unknown native action')
                unique, reverse_pos = np.unique(ii[::-1], return_index=True)
                last = len(ii)-1-reverse_pos
                state[unique] = np.where(aa[last] == 3, 0, qq[last])
    def sample(stamp):
        nonlocal bbo_checks
        j = np.searchsorted(q[:, 0], stamp, side='right')-1
        _, bp, ap, bq, aq = q[j]
        bi, ai = int(bp//tick_raw), int(ap//tick_raw)
        if bi <= 0 or ai+20000 >= len(ask):
            raise ValueError('Reference outside documented reconstruction band')
        if bid[bi] != bq or ask[ai] != aq:
            raise ValueError(f'Reconstructed BBO size disagrees with verified quote at {stamp}')
        bbo_checks += 1
        mid = (bp+ap)/2/1e9
        width = mid*.0001
        sides_out = []
        for state, start, direction in [(bid, bi, -1), (ask, ai, 1)]:
            count = int(np.ceil(10*width/.1))
            idx = start+direction*np.arange(count)
            bins = np.floor(np.arange(count)*.1/width).astype(int)
            mask = bins < 10
            quantities = state[idx[mask]]/1e9
            sides_out.append(np.bincount(bins[mask], weights=quantities, minlength=10))
        if min(sides_out[0][0], sides_out[1][0]) <= 0:
            raise ValueError('Nonpositive best price bucket')
        times.append(stamp)
        depths.append(sides_out)
        mids.append(mid)
    for file_index, file in enumerate(sorted((root/'order_book_deltas/BTCUSDT-PERP.BINANCE').glob('*.parquet'))):
        with file.open('rb') as stream:
            hashes.append({'file': str(file.resolve()), 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()})
        for batch in pq.ParquetFile(file).iter_batches(batch_size=250000, columns=['action', 'side', 'price', 'size', 'ts_init']):
            actions, sides = batch.column('action').to_numpy(), batch.column('side').to_numpy()
            prices, sizes = raw(batch.column('price')), raw(batch.column('size'))
            ts = batch.column('ts_init').to_numpy().astype(np.int64)
            start = 0
            while target < int(ts[-1]):
                stop = int(np.searchsorted(ts, target, side='right'))
                apply(actions[start:stop], sides[start:stop], prices[start:stop], sizes[start:stop])
                sample(target)
                target += step
                start = stop
            apply(actions[start:], sides[start:], prices[start:], sizes[start:])
            rows += len(ts)
        if file_index % 50 == 0:
            print(f'files={file_index+1} rows={rows} snapshots={len(times)}', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'full_depth_1bp.npz', ts_ns=np.asarray(times, np.int64),
                        depth_btc=np.asarray(depths), mid_usdt=np.asarray(mids))
    record = dict(source_session=str(session.resolve()), source_native_files=hashes,
        reconstructed_delta_rows=rows, samples=len(times), bbo_size_checks=bbo_checks,
        first_utc=datetime.fromtimestamp(times[0]/1e9, timezone.utc).isoformat(),
        last_utc=datetime.fromtimestamp(times[-1]/1e9, timezone.utc).isoformat(),
        cadence_seconds=cadence, depth_bins='10 bins, each 1bp wide from best quote on each side',
        price_grid_usdt=.1, reconstructed_price_band_usdt=[0, upper_usdt],
        aggregation_note='Bucket shape adaptation, not a claim that native exchange levels are 1bp apart',
        source_previously_verified=verified, original_MSFT_data=False)
    (output/'full_depth_provenance.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in record.items() if k not in ['source_native_files', 'source_previously_verified']}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cadence', type=float, default=30.)
    args = p.parse_args()
    if args.cadence <= 0:
        p.error('cadence must be positive')
    extract(args.session, args.output, args.cadence)
