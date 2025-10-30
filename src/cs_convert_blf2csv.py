#!/usr/bin/env python3
"""
blf2csv.py — Convert a .BLF CAN log to a wide, binned CSV using one or more DBC files.

Example:
  python blf2csv.py \
    --blf "IBS Full load 2025-08-20_22-17-18_L063.blf" \
    --dbc "FM29_EVBUS_Matrix_CANFD_V390.6_20230411.dbc" \
    --dbc "FM29_BodyBUS_Matrix_CAN_V390.6_20230411.dbc" \
    --indir "can_data" \
    --outdir "can_data"

This will produce:
  can_data/IBS Full load 2025-08-20_22-17-18_L063.csv
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional
import numpy as np
import pandas as pd

import logging
logging.getLogger("cantools.database").setLevel(logging.ERROR)

# Third-party libs
try:
    import can  # python-can
except Exception as e:
    raise SystemExit("ERROR: python-can is required. Install with: pip install python-can") from e

try:
    from cantools import database as cdb  # cantools
except Exception as e:
    raise SystemExit("ERROR: cantools is required. Install with: pip install cantools") from e

# Progress bars (optional)
try:
    from tqdm import tqdm
except Exception:
    # Fallback: no-op progress if tqdm isn't installed
    class tqdm:  # type: ignore
        def __init__(self, total=None, unit="", unit_scale=False, desc="", leave=True):
            self.total = total
            self.n = 0
        def update(self, n=1):
            self.n += n
        def refresh(self): ...
        def write(self, s): print(s)
        def close(self): ...

# Timezone support
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # Not critical; we'll skip local time column if missing.

MASK_29BIT = 0x1FFFFFFF


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert a .BLF CAN log to CSV using one or more DBC files (wide, binned).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--blf", required=True, help="BLF filename (basename or path)")
    p.add_argument("--dbc", required=True, action="append",
                   help="DBC filename(s). Repeat this flag for multiple DBCs.")
    p.add_argument("--indir", default=".", help="Input directory containing BLF/DBC files")
    p.add_argument("--outdir", default=".", help="Output directory for CSV")
    p.add_argument("--bin-ms", type=int, default=1000, help="Bin size in milliseconds")
    p.add_argument("--tz", default="America/Detroit", help="Local timezone for convenience column")
    p.add_argument("--no-prefix", action="store_true",
                   help="Do not prefix signal names with <MessageName>.")
    return p.parse_args(argv)


def load_dbcs(dbc_paths: Iterable[Path]) -> cdb.Database:
    db = cdb.Database(frame_id_mask=MASK_29BIT, strict=True)
    for path in dbc_paths:
        if not path.exists():
            raise FileNotFoundError(f"DBC not found: {path}")
        db.add_dbc_file(path)
    return db


def _get_file_handle(reader) -> Optional[object]:
    """Best-effort: find an underlying file handle to show byte-level progress."""
    for attr in ("_file", "file", "_fh", "_f", "fp", "f", "raw_file"):
        fh = getattr(reader, attr, None)
        if fh is not None and hasattr(fh, "tell"):
            return fh
    return None


def decode_blf_to_rows(blf_path: Path, db: cdb.Database, use_prefix: bool) -> List[Tuple[float, Dict[str, float]]]:
    total_bytes = os.path.getsize(blf_path)
    msg_by_id = {m.frame_id & MASK_29BIT: m for m in db.messages}

    rows: List[Tuple[float, Dict[str, float]]] = []
    with can.BLFReader(str(blf_path)) as reader:
        fh = _get_file_handle(reader)
        if fh and total_bytes > 0:
            pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, desc="Decoding BLF")
            update_mode = "bytes"
        else:
            pbar = tqdm(total=None, unit="msg", desc="Decoding BLF")
            update_mode = "msgs"

        for i, msg in enumerate(reader, 1):
            try:
                ts = float(msg.timestamp)  # seconds since epoch (float)
                frame_id = msg.arbitration_id & MASK_29BIT
                m = msg_by_id.get(frame_id)
                if m is not None:
                    decoded = m.decode(msg.data)
                    if use_prefix:
                        decoded = {f"{m.name}.{k}": v for k, v in decoded.items()}
                    rows.append((ts, decoded))
            except Exception:
                # Ignore undecodable frames.
                pass

            if i % 1000 == 0:
                if update_mode == "bytes" and fh:
                    try:
                        pbar.n = min(fh.tell(), total_bytes)
                        pbar.refresh()
                    except Exception:
                        update_mode = "msgs"
                        pbar.total = None
                        pbar.update(1000)
                else:
                    pbar.update(1000)

        if update_mode == "bytes":
            pbar.n = total_bytes
            pbar.refresh()
        pbar.close()

    return rows


def build_dataframe(rows: List[Tuple[float, Dict[str, float]]], bin_ms: int,
                    tz_name: Optional[str]) -> pd.DataFrame:
    """Bin to `bin_ms`, forward-fill, and add time columns."""
    from collections import defaultdict
    bins = defaultdict(dict)

    with tqdm(total=len(rows), desc="Transforming (bin/merge)", unit="msg") as t2:
        for ts, decoded in rows:
            bin_idx = int(ts * 1000) // bin_ms
            bins[bin_idx].update(decoded)
            t2.update(1)

    with tqdm(total=5, desc="Post-transform prep", unit="step") as t3:
        bin_keys = np.array(sorted(bins.keys()), dtype=np.int64)
        data_rows = [bins[k] for k in bin_keys]
        wide = pd.DataFrame.from_records(data_rows)

        # Convert mostly-numeric object cols early
        obj_cols = [c for c in wide.columns if wide[c].dtype == "object"]
        for c in obj_cols:
            s = pd.to_numeric(wide[c], errors="coerce")
            if s.notna().mean() >= 0.90:
                wide[c] = s

        # Downcast floats early
        fcols = [c for c in wide.columns if pd.api.types.is_float_dtype(wide[c])]
        if fcols:
            wide[fcols] = wide[fcols].astype("float32")

        # Build sparse -> full time index
        idx_sparse = pd.to_datetime(bin_keys * bin_ms, unit="ms", utc=True)
        wide.index = idx_sparse
        full_index = pd.date_range(idx_sparse.min(), idx_sparse.max(), freq=f"{bin_ms}ms", tz="UTC")
        wide = wide.reindex(full_index, method="ffill")

        idx = wide.index
        out = wide.copy()

        # time columns
        out.insert(0, "time", (idx.asi8 / 1e9).astype("float64"))
        out.insert(1, "time_utc_iso", idx.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3])

        # local timezone if available
        if tz_name and ZoneInfo is not None:
            try:
                tz = ZoneInfo(tz_name)
                out.insert(2, "time_local_iso", idx.tz_convert(tz).strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3])
            except Exception:
                out.insert(2, "time_local_iso", out["time_utc_iso"])
        else:
            out.insert(2, "time_local_iso", out["time_utc_iso"])

        out.insert(3, "excel_utc", (idx.asi8 / 1e9) / 86400.0 + 25569.0)
        t3.update(1)

        # Optimize dtypes (keep 'time' as float64)
        fcols = out.select_dtypes(include=["float64", "float32"]).columns.difference(["time"])
        if len(fcols):
            out[fcols] = out[fcols].astype("float32")

        icols = out.select_dtypes(include=["int64", "int32", "int"]).columns
        if len(icols):
            out[icols] = out[icols].astype("Int32")

        obj_cols = [c for c in out.columns if out[c].dtype == "object" and c not in ("time_utc_iso", "time_local_iso")]
        for c in tqdm(obj_cols, desc="Coercing object→numeric", unit="col", leave=False):
            s = pd.to_numeric(out[c], errors="coerce")
            if s.notna().mean() >= 0.90:
                out[c] = s.astype("float32")
        t3.update(1)

        rows_count, cols_count = out.shape
        t3.write(f"Grid ready: {rows_count:,} rows × {cols_count:,} cols")
        t3.update(1)

    return out


def write_csv_chunked(df: pd.DataFrame, out_csv: Path, chunk_rows: int = 200_000) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_total = len(df)
    mode = "w"
    header = True
    with tqdm(total=rows_total, unit="row", desc="Writing CSV") as pbar:
        for start in range(0, rows_total, chunk_rows):
            end = min(start + chunk_rows, rows_total)
            df.iloc[start:end].to_csv(out_csv, index=False, mode=mode, header=header)
            mode, header = "a", False
            pbar.update(end - start)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    blf_path = (indir / args.blf).resolve() if not Path(args.blf).is_absolute() else Path(args.blf)
    if not blf_path.exists():
        raise SystemExit(f"BLF not found: {blf_path}")

    dbc_paths = []
    for d in args.dbc:
        p = (indir / d).resolve() if not Path(d).is_absolute() else Path(d)
        dbc_paths.append(p)

    # Output path: <outdir>/<blf_basename>.csv
    out_csv = (outdir / (blf_path.stem + ".csv")).resolve()

    print(f"BLF:    {blf_path}")
    print(f"DBCs:   {', '.join(str(p) for p in dbc_paths)}")
    print(f"OUT:    {out_csv}")
    print(f"BIN_MS: {args.bin_ms}")
    print(f"Prefix message names: {not args.no_prefix}")
    if ZoneInfo is None and args.tz:
        print("Note: zoneinfo not available; 'time_local_iso' will mirror UTC.")

    # Load DBCs and decode
    db = load_dbcs(dbc_paths)
    rows = decode_blf_to_rows(blf_path, db, use_prefix=(not args.no_prefix))

    if not rows:
        print("WARNING: No decodable messages found. Check DBCs or BLF content.")

    df = build_dataframe(rows, bin_ms=args.bin_ms, tz_name=args.tz)
    write_csv_chunked(df, out_csv)

    # Summary
    signal_cols = df.shape[1] - 4  # subtract time columns
    print(f"Done. Rows: {len(df):,}  Cols (signals): {signal_cols:,}  BIN_MS: {args.bin_ms} → {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
