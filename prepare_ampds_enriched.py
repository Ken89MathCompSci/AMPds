"""
Builds data/AMPds_enriched_data/{train,val,test}.pkl from AMPds2.h5.

Each pkl stores a single-element tuple (df,) where df has columns:
  main        – mains active power  (W)
  main_Q      – mains reactive power (VAR)   ← new feature
  dish washer – DWE active power    (W)
  fridge      – FGE active power    (W)
  oven        – WOE active power    (W)
  washer dryer– CDE active power    (W)

Meter mapping (verified against existing data/AMPds/train.pkl):
  meter1  = WHE  mains
  meter5  = CDE  clothes dryer  → "washer dryer"
  meter8  = DWE  dishwasher     → "dish washer"
  meter11 = FGE  fridge         → "fridge"
  meter21 = WOE  wall oven      → "oven"

Splits (1-min resolution, same as data/AMPds/):
  train : 2013-11-21
  val   : 2013-12-31
  test  : 2012-08-23
"""

import os
import pickle
import numpy as np
import pandas as pd
import tables

H5_PATH  = 'data/AMPds2.h5'
OUT_DIR  = 'data/AMPds_enriched_data'
FREQ     = '1min'
TZ       = 'America/Vancouver'

# Column indices inside values_block_0
COL_P = 5   # power, active (W)
COL_Q = 7   # power, reactive (VAR)

METER_MAP = {
    'main':         (1,  COL_P),
    'main_Q':       (1,  COL_Q),
    'dish washer':  (8,  COL_P),
    'fridge':       (11, COL_P),
    'oven':         (21, COL_P),
    'washer dryer': (5,  COL_P),
}

SPLITS = {
    'train': '2013-11-21',
    'val':   '2013-12-31',
    'test':  '2012-08-23',
}


def read_meter_column(h5path: str, meter_num: int, col_idx: int) -> pd.Series:
    with tables.open_file(h5path, 'r') as f:
        node = f.get_node(f'/building1/elec/meter{meter_num}/table')
        ts   = node.col('index')
        vals = node.col('values_block_0')[:, col_idx]
    idx = pd.to_datetime(ts, unit='ns', utc=True).tz_convert(TZ)
    return pd.Series(vals.astype(np.float32), index=idx)


def build_split(series_map: dict, day: str) -> pd.DataFrame:
    resampled = {}
    for col, s in series_map.items():
        resampled[col] = (
            s.loc[day]
             .resample(FREQ)
             .mean()
             .ffill()
             .fillna(0)
        )
    df = pd.DataFrame(resampled)
    df.index.name = 'time'
    # strip timezone so downstream code stays simple (matches existing pkl format)
    df.index = df.index.tz_localize(None)
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print('Reading meters from AMPds2.h5 ...')
    series_map = {}
    for col, (meter_num, col_idx) in METER_MAP.items():
        print(f'  meter{meter_num}  col={col_idx}  -> "{col}"')
        series_map[col] = read_meter_column(H5_PATH, meter_num, col_idx)

    for split, day in SPLITS.items():
        print(f'\nBuilding {split} split  ({day}) ...')
        df = build_split(series_map, day)
        print(f'  shape : {df.shape}')
        print(f'  cols  : {list(df.columns)}')
        print(f'  sample:\n{df.head(3).to_string()}')

        out_path = os.path.join(OUT_DIR, f'{split}.pkl')
        with open(out_path, 'wb') as fh:
            pickle.dump((df,), fh)
        print(f'  saved -> {out_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
