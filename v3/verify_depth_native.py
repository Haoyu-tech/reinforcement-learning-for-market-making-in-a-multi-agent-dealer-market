"""Independent small-window check against Nautilus's native OrderBook engine."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from nautilus_trader.model.book import OrderBook
from nautilus_trader.model.data import OrderBookDelta
from nautilus_trader.model.enums import BookType, OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'Binance-Quant'))
from verify_converted import objects


def verify(session, extracted, count):
    z = np.load(extracted)
    times, expected = z['ts_ns'][:count], z['depth_btc'][:count]
    iid = InstrumentId.from_str('BTCUSDT-PERP.BINANCE')
    book = OrderBook(iid, BookType.L2_MBP)
    catalog = ParquetDataCatalog(str(session/'catalog'))
    next_sample, read = 0, 0
    for delta in objects(catalog, OrderBookDelta):
        while next_sample < len(times) and delta.ts_init > times[next_sample]:
            bp, ap = float(book.best_bid_price()), float(book.best_ask_price())
            width = (bp+ap)/2*.0001
            actual = []
            for side, start, direction in [(OrderSide.SELL, bp, -1), (OrderSide.BUY, ap, 1)]:
                bound = start+direction*(10*width+.2)
                levels = book.get_all_crossed_levels(side, Price.from_str('{:.2f}'.format(bound)), 3)
                totals = np.zeros(10)
                for price, quantity in levels:
                    distance = abs(float(price)-start)
                    # Use exact integer tick distance for bucket boundaries.
                    distance = round(distance*10)/10
                    bucket = int(np.floor(distance/width))
                    if bucket < 10:
                        totals[bucket] += float(quantity)
                actual.append(totals)
            np.testing.assert_allclose(actual, expected[next_sample], atol=1e-8, rtol=1e-10)
            next_sample += 1
        if next_sample >= len(times):
            break
        book.apply_delta(delta)
        read += 1
    if next_sample != len(times):
        raise ValueError('Did not independently verify every requested sample')
    result = dict(native_engine='Nautilus OrderBook L2_MBP', compared_snapshots=count,
        replayed_deltas=read, all_twenty_depth_buckets_match=True,
        method='independent event-by-event engine vs vectorized last-write reconstruction')
    extracted.with_name('native_depth_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--extracted', type=Path, required=True)
    p.add_argument('--count', type=int, default=5)
    a = p.parse_args()
    verify(a.session, a.extracted, a.count)
