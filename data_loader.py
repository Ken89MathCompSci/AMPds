import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os

# Target houses and appliances
TARGET_HOUSES = [1, 2, 5]
TARGET_APPLIANCES = ['fridge', 'dishwasher', 'microwave', 'washer_dryer', 'kettle']

# Fixed date-range splits (6-second intervals → 14,400 samples per day)
RESAMPLE_FREQ = '6s'
SPLIT_RANGES = {
    'train': ('2013-04-10 00:00:00', '2013-04-23 23:59:54'),  # House 2 spring (14 days)
    'val':   ('2013-07-08 00:00:00', '2013-07-08 23:59:54'),  # House 2 summer (1 day)
    'test':  ('2013-10-09 00:00:00', '2013-10-09 23:59:54'),  # House 2 autumn (1 day)
}

# Known meter-to-appliance mapping for UKDALE houses 1, 2, 5
# meter 1 is always the aggregate mains
HOUSE_METER_MAP = {
    1: {
        'kettle':      2,
        'microwave':   3,
        'dishwasher':  4,
        'fridge':      5,
        'washer_dryer': 6,
    },
    2: {
        'kettle':      2,
        'dishwasher':  3,
        'fridge':      4,
        'microwave':   5,
        'washer_dryer': 8,
    },
    5: {
        'kettle':      2,
        'microwave':   4,
        'dishwasher':  5,
        'fridge':      6,
        'washer_dryer': 8,
    },
}


class UKDaleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def read_meter(h5_path, building, meter):
    """
    Read power data for a single meter from the UKDALE HDF5 file.
    Returns a pandas Series with a DatetimeIndex.
    """
    key = f'/building{building}/elec/meter{meter}/table'
    df = pd.read_hdf(h5_path, key=key)
    timestamps = pd.to_datetime(df['index'], unit='ns', utc=True)
    return pd.Series(df['values_block_0'].values.astype(np.float32), index=timestamps)


def slice_and_resample(mains_series, appliance_series, start, end):
    """
    Slice both series to [start, end], resample to RESAMPLE_FREQ, and align.
    Fills gaps with forward-fill then zero.
    """
    # Parse timestamps as UTC to match the timezone-aware index
    start_ts = pd.Timestamp(start, tz='UTC')
    end_ts   = pd.Timestamp(end,   tz='UTC')

    mains_slice = mains_series.loc[start_ts:end_ts]
    app_slice   = appliance_series.loc[start_ts:end_ts]

    mains = mains_slice.resample(RESAMPLE_FREQ).mean().ffill().fillna(0)
    appliance = app_slice.resample(RESAMPLE_FREQ).mean().ffill().fillna(0)

    df = pd.DataFrame({'mains': mains, 'appliance': appliance}).dropna()
    return df['mains'].values.astype(np.float32), df['appliance'].values.astype(np.float32)


def create_sequences(mains, appliance, window_size, target_size=1):
    """
    Create sliding window sequences (sequence-to-point by default).
    """
    X, y = [], []
    stride = 5

    for i in range(0, len(mains) - window_size - target_size + 1, stride):
        X.append(mains[i:i + window_size])
        if target_size == 1:
            midpoint = i + window_size // 2
            y.append(appliance[midpoint:midpoint + 1])
        else:
            y.append(appliance[i + window_size:i + window_size + target_size])

    return np.array(X).reshape(-1, window_size, 1), np.array(y)


def load_house(h5_path, building, window_size=100, target_size=1, normalize=True):
    """
    Load and preprocess all target appliances for one house using fixed
    date-range splits defined in SPLIT_RANGES.

    Returns:
        dict mapping appliance_name -> data dict with DataLoaders and metadata
    """
    print(f"\n=== House {building} ===")
    meter_map = HOUSE_METER_MAP.get(building, {})

    # Load aggregate mains (meter 1)
    mains_series = read_meter(h5_path, building, meter=1)
    print(f"  Mains: {len(mains_series)} samples  "
          f"({mains_series.index[0]} -> {mains_series.index[-1]})")

    results = {}

    for appliance in TARGET_APPLIANCES:
        meter = meter_map.get(appliance)
        if meter is None:
            print(f"  [SKIP] '{appliance}' — no meter mapping for house {building}")
            continue

        try:
            appliance_series = read_meter(h5_path, building, meter)
        except KeyError:
            print(f"  [SKIP] '{appliance}' — meter {meter} not found in file")
            continue

        print(f"  [FOUND] '{appliance}' at meter {meter}: {len(appliance_series)} samples")

        # Slice and resample each split independently
        splits_raw = {}
        skip = False
        for split, (start, end) in SPLIT_RANGES.items():
            m, a = slice_and_resample(mains_series, appliance_series, start, end)
            print(f"    {split}: {len(m)} samples "
                  f"(expected 14400, {start[:10]})")
            if len(m) == 0:
                print(f"  [SKIP] '{appliance}' — no data for {split} split ({start[:10]})")
                skip = True
                break
            splits_raw[split] = (m, a)

        if skip:
            continue

        # Fit scalers on training slice only to avoid data leakage
        train_mains, train_app = splits_raw['train']
        if normalize:
            mains_scaler = MinMaxScaler()
            appliance_scaler = MinMaxScaler()
            mains_scaler.fit(train_mains.reshape(-1, 1))
            appliance_scaler.fit(train_app.reshape(-1, 1))
            def norm_m(x): return mains_scaler.transform(x.reshape(-1, 1)).flatten()
            def norm_a(x): return appliance_scaler.transform(x.reshape(-1, 1)).flatten()
        else:
            mains_scaler = appliance_scaler = None
            def norm_m(x): return x
            def norm_a(x): return x

        loaders = {}
        for split, (m, a) in splits_raw.items():
            X, y = create_sequences(norm_m(m), norm_a(a), window_size, target_size)
            shuffle = (split == 'train')
            loaders[split] = DataLoader(UKDaleDataset(X, y), batch_size=32, shuffle=shuffle)
            print(f"    {split}_loader: {X.shape}")

        results[appliance] = {
            'train_loader':      loaders['train'],
            'val_loader':        loaders['val'],
            'test_loader':       loaders['test'],
            'mains_scaler':      mains_scaler,
            'appliance_scaler':  appliance_scaler,
            'appliance_name':    appliance,
            'window_size':       window_size,
            'target_size':       target_size,
            'input_size':        1,
            'output_size':       target_size,
        }

    return results


def load_all_houses(h5_path, window_size=100, target_size=1, normalize=True):
    """
    Load and preprocess houses 1, 2, and 5 for all target appliances.

    Returns:
        dict: { house_id (int) -> { appliance_name -> data dict } }
    """
    all_data = {}
    for building in TARGET_HOUSES:
        all_data[building] = load_house(
            h5_path, building, window_size, target_size, normalize
        )
    return all_data


def check_coverage(h5_path):
    """
    Print a table showing how many resampled samples each house/appliance
    has on each split date.  Flags entries below MIN_SAMPLES as [LOW] or [MISSING].
    """
    MIN_SAMPLES = 14000  # ~97 % of a full 14,400-sample day

    header = f"{'House':<7} {'Appliance':<14}" + "".join(
        f"  {split:>6}({start[:10]})" for split, (start, _) in SPLIT_RANGES.items()
    )
    print("\n" + "=" * len(header))
    print("Coverage check  (6s resampling, expected 14,400 samples/day)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for building in TARGET_HOUSES:
        meter_map = HOUSE_METER_MAP.get(building, {})
        try:
            mains_series = read_meter(h5_path, building, meter=1)
        except Exception:
            print(f"H{building}  [mains not readable]")
            continue

        for appliance in TARGET_APPLIANCES:
            meter = meter_map.get(appliance)
            if meter is None:
                continue
            try:
                app_series = read_meter(h5_path, building, meter)
            except KeyError:
                continue

            counts = []
            for split, (start, end) in SPLIT_RANGES.items():
                m, _ = slice_and_resample(mains_series, app_series, start, end)
                counts.append(len(m))

            flags = []
            for c in counts:
                if c == 0:
                    flags.append(f"{'MISSING':>20}")
                elif c < MIN_SAMPLES:
                    flags.append(f"{c:>17}[LOW]")
                else:
                    flags.append(f"{c:>20}")

            row = f"H{building:<6} {appliance:<14}" + "".join(flags)
            print(row)

    print("=" * len(header) + "\n")


def explore_h5_structure(h5_path):
    """
    Print the meter structure of the UKDALE HDF5 file.
    Useful for verifying meter numbers before loading.
    """
    import h5py
    print(f"Exploring: {h5_path}\n")
    with h5py.File(h5_path, 'r') as f:
        for building in TARGET_HOUSES:
            key = f'building{building}/elec'
            if key not in f:
                print(f"  House {building}: not found")
                continue
            meters = sorted(f[key].keys())
            print(f"  House {building}: {meters}")


# ---------------------------------------------------------------------------
# Compatibility wrappers — used by train_*.py scripts
# ---------------------------------------------------------------------------
H5_PATH = "preprocessed_datasets/ukdale/ukdale.h5"


def explore_available_appliances(_file_path):
    """
    Compat wrapper: returns {index: appliance_name} for TARGET_APPLIANCES.
    _file_path is ignored (kept for API compatibility with train_*.py).
    """
    return {i: name for i, name in enumerate(TARGET_APPLIANCES)}


def load_and_preprocess_ukdale(file_path, appliance_index, window_size=100,
                                target_size=1, normalize=True, **_kwargs):
    """
    Compat wrapper: maps (file_path, appliance_index) to the new H5 loader.
    Extracts house number from file_path (e.g. 'ukdale1.mat' -> house 1).
    """
    import re
    match = re.search(r'ukdale(\d+)', os.path.basename(file_path))
    building = int(match.group(1)) if match else 1

    appliance_name = TARGET_APPLIANCES[appliance_index]
    house_data = load_house(H5_PATH, building, window_size, target_size, normalize)

    if appliance_name not in house_data:
        raise ValueError(f"'{appliance_name}' not found in house {building}")

    return house_data[appliance_name]


if __name__ == "__main__":
    h5_path = "preprocessed_datasets/ukdale/ukdale.h5"

    # Explore structure first
    explore_h5_structure(h5_path)

    # Coverage check — shows which house/appliance combos have data on the split dates
    check_coverage(h5_path)

    # Load all houses and appliances
    all_data = load_all_houses(h5_path, window_size=100, target_size=1)
    print(f"\nSplit ranges (6s intervals, ~14400 samples/day):")
    for split, (start, end) in SPLIT_RANGES.items():
        print(f"  {split}: {start}  ->  {end}")

    print("\n=== Summary ===")
    for house, appliances in all_data.items():
        print(f"House {house}:")
        for appliance, d in appliances.items():
            print(f"  {appliance}: "
                  f"train={len(d['train_loader'])} batches, "
                  f"val={len(d['val_loader'])} batches, "
                  f"test={len(d['test_loader'])} batches")
