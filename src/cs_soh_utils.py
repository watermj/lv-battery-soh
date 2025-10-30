#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs_soh_utils.py

Utility functions for analyzing and computing vehicle State of Health (SoH).

Author: Jason Waterman
Course: [12V Battery SOH / UC Berkeley / PCAIML]
Created: 2025-10-17
Last Updated: 2025-10-19
Version: 0.1.0
License: MIT

Description:
    This module provides helper functions and classes used in SoH analysis,
    including signal extraction, data cleaning, feature computation,
    and visualization utilities.

    Typical usage example:
        from soh_utils import compute_capacity, detect_cycles

Notes:
    - Designed for use with EV battery and vehicle telemetry datasets.
    - Functions are written to be modular and reusable in Jupyter notebooks
      as well as production pipelines.

"""

# ===== Imports =====
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta, datetime
from typing import List, Tuple, Optional

# ===== Public API =====
__all__ = [
    "calculate_soh1",
    "detect_cycles_from_voltage",
    "count_cycles_voltage",
    "merge_short_gaps", 
    "parse_log_date",
    "compute_log_soh", 
    "ensure_dt_index",
    "attach_soh1_series",
    "ensure_time_index_1s", 
    "ensure_time_utc"
]

# ===== Functions =====

def calculate_soh1(
    dfr: pd.DataFrame,
    return_components: bool = False,
    return_series: bool = False,
    *,
    resample_s: int = 2,
    return_vendor: bool = False,
    output_s: int | None = 1
) -> float | dict | pd.DataFrame:
    """
    Compute SOH1 (fraction in [0,1]) for a single log.

    Parameters
    ----------
    dfr : DataFrame
        Raw or cleaned log; must contain at least IBS_V, IBS_I, IBS_T.
    return_components : bool
        If False -> return a single float (median SOH1).
        If True  -> return medians (tuple) or a DataFrame if return_series=True.
    return_series : bool
        If True and return_components=True -> return full time series DataFrame.
    resample_s : int, default 2
        Resampling cadence in seconds for the internal pipeline.
    return_vendor : bool, default False
        When returning a series, include vendor IBS SoH as 'SOH_VENDOR' (fraction in [0,1])
        aligned to the same index.

    Returns
    -------
    float | tuple | DataFrame
        - float: median SOH1
        - tuple: (median_SOH1, median_SOH_C, median_SOH_R)
        - DataFrame: columns ['SOH1','SOH_C','SOH_R'] (+ optional 'SOH_VENDOR')
    """

    # ------------------- CONFIG -------------------
    RESAMPLE_SECONDS = max(1, int(resample_s))
    EARLY_DAYS       = 14
    MIN_WARM_TEMP    = 15
    DOD_MIN_PERCENT  = 10
    STEP_MIN_DI      = 5.0
    REST_I_MAX_STD   = 0.5     # standard rest threshold
    REST_DV_MV_STD   = 3.0
    REST_I_MAX_FBK   = 1.0     # fallback (looser) rest threshold
    REST_DV_MV_FBK   = 10.0
    ALPHA_CAP        = 0.005
    BETA_R           = 0.007
    C_NOMINAL_FALLBACK_AH = 75.0

    # Column map
    COLS = dict(
        V="IBS_BatteryVoltage",
        I="IBS_BatteryCurrent",
        T="IBS_BatteryTemperature",
        SOC="IBS_StateOfCharge",           # optional
        SOH_IBS="IBS_StateOfHealth",       # optional (%)
        RI="IBS_AvgRi",                    # optional (mΩ)
        CAP_AVAIL="IBS_AvailableCapacity", # optional (Ah)
        CAP_DISCH="IBS_DischargeableAh",   # optional (Ah)
        CAP_NOM="IBS_NominalCapacity",     # optional (Ah)
        SULF="IBS_Sulfation",              # optional
        DEFECT="IBS_BatteryDefect",        # optional
    )
    SLEEP_COL = "VCU_ChrgSysOperCmd"       # 0=awake, 1=sleep

    # ------------------- Helpers -------------------
    def _ensure_time_index(df: pd.DataFrame) -> pd.DataFrame:
        time_candidates = ["time_utc", "time", "time_utc_iso", "timestamp"]
        tcol = next((c for c in time_candidates if c in df.columns), None)
        if tcol is not None:
            idx = pd.to_datetime(df[tcol], utc=True, errors="coerce")
        else:
            idx = pd.to_datetime(df.index, utc=True, errors="coerce")
        good = ~idx.isna()
        if good.sum() < 3:
            return pd.DataFrame(index=pd.DatetimeIndex([], tz=None))
        df2 = df.loc[good].copy()
        df2.index = idx[good]
        try:
            df2.index = df2.index.tz_convert(None)
        except Exception:
            pass
        return df2.sort_index()

    def _nominal_capacity(df: pd.DataFrame) -> float:
        if COLS["CAP_NOM"] in df.columns:
            val = float(pd.to_numeric(df[COLS["CAP_NOM"]], errors="coerce").dropna().median())
            if np.isfinite(val):
                return val
        return C_NOMINAL_FALLBACK_AH

    def _temp_correct_R_to_25C(Rs: pd.Series, Ts: pd.Series) -> pd.Series:
        return Rs / (1 + BETA_R * (25.0 - Ts))

    def _normalize_percent_to_fraction(x: float) -> float:
        if np.isfinite(x) and x > 1.5:
            return x / 100.0
        return x

    def _compute_soh_final(df: pd.DataFrame,
                           use_awake_mask: bool,
                           rest_i_max: float,
                           rest_dv_mv: float) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Core fusion pipeline. If use_awake_mask=True, gate by SLEEP_COL.
        Returns (SOH_final, SOH_C, SOH_R, SOH_vendor) as series aligned on the resampled index.
        """
        if df.empty:
            return (pd.Series(dtype=float),)*4

        # Awake/sleep mask
        if use_awake_mask and (SLEEP_COL in df.columns):
            sleep_raw = pd.to_numeric(df[SLEEP_COL], errors="coerce")
            is_awake_raw = (sleep_raw == 0).astype(int)
        else:
            is_awake_raw = pd.Series(1, index=df.index, dtype=int)

        df2 = df.copy()
        df2["is_awake"] = is_awake_raw

        # Null out key signals during sleep before resampling
        to_null = [COLS.get("V"), COLS.get("I"), COLS.get("T"),
                   COLS.get("SOC"), COLS.get("RI"),
                   COLS.get("CAP_AVAIL"), COLS.get("CAP_DISCH")]
        for c in to_null:
            if c and c in df2.columns:
                df2.loc[df2["is_awake"] == 0, c] = np.nan

        # Light fills on some columns, then re-null during sleep
        numdf = df2.select_dtypes(include=[np.number]).copy()
        for c in [COLS.get("RI"), COLS.get("SOC"), COLS.get("CAP_AVAIL"), COLS.get("CAP_DISCH")]:
            if c and c in numdf.columns:
                numdf[c] = pd.to_numeric(numdf[c], errors="coerce").ffill(limit=5).bfill(limit=5)
        for c in to_null:
            if c and c in numdf.columns:
                numdf.loc[numdf["is_awake"] == 0, c] = np.nan

        r = numdf.resample(f"{RESAMPLE_SECONDS}s").median()
        if r.empty:
            return (pd.Series(dtype=float),)*4

        # Resample vendor SOH (fraction) to r.index
        vendor_series = pd.Series(dtype=float)
        if COLS["SOH_IBS"] in df.columns:
            vendor_raw = pd.to_numeric(df[COLS["SOH_IBS"]], errors="coerce")
            vendor_series = (vendor_raw / 100.0).resample(f"{RESAMPLE_SECONDS}s").median()

        # Awake flag after resample
        if "is_awake" in r.columns:
            AW = (r["is_awake"] >= 0.5).astype(bool)
        else:
            AW = pd.Series(True, index=r.index)

        # Signals
        if (COLS["V"] not in r.columns) or (COLS["I"] not in r.columns) or (COLS["T"] not in r.columns):
            return (pd.Series(dtype=float),)*4
        V = r[COLS["V"]].astype(float)
        I = r[COLS["I"]].astype(float)
        T = r[COLS["T"]].astype(float).clip(-30, 70)

        # Nominal capacity + early window
        C_NOM_REF = _nominal_capacity(df)
        start_ts = r.index[0]
        cutoff   = start_ts + pd.to_timedelta(EARLY_DAYS, unit="D")

        # Charge/discharge flags
        v_on, v_off = 13.6, 13.2
        is_charge = pd.Series(False, index=r.index)
        above_on = (V > v_on) & (AW if use_awake_mask else True)
        below_off = (V < v_off) & (AW if use_awake_mask else True)
        charging_state = False
        for ts in r.index:
            if above_on.loc[ts]:
                charging_state = True
            elif below_off.loc[ts]:
                charging_state = False
            is_charge.loc[ts] = charging_state and (AW.loc[ts] if use_awake_mask else True)
        r["is_charge"] = is_charge.astype(int)
        r["is_discharge"] = ((I < 0) & (~is_charge) & (AW if use_awake_mask else True)).astype(int)

        # Rest detection
        V_filt = V.rolling(int(max(1, 30/RESAMPLE_SECONDS)), min_periods=1).median()
        dv = V_filt.diff().abs()
        dv_avg_60s = dv.rolling(int(max(1, 60/RESAMPLE_SECONDS)), min_periods=1).mean()
        is_rest = (I.abs() <= rest_i_max) & (dv_avg_60s <= (rest_dv_mv/1000.0))
        if use_awake_mask:
            is_rest = is_rest & AW
        r["is_rest"] = is_rest.astype(int)

        # Resistance track
        if COLS["RI"] in r.columns:
            Ri_raw = pd.to_numeric(r[COLS["RI"]], errors="coerce") / 1000.0  # mΩ->Ω
            if use_awake_mask:
                Ri_raw = Ri_raw.where(AW)
            Ri_sm  = Ri_raw.rolling("30min", min_periods=1).median()
        else:
            dI = I.diff(); dV = V.diff()
            step_mask = (dI.abs() >= STEP_MIN_DI) & (~is_charge) & (AW if use_awake_mask else True)
            Ri_step = (dV.abs() / dI.abs()).where(step_mask)
            Ri_sm   = Ri_step.rolling("30min", min_periods=1).median()

        R25 = _temp_correct_R_to_25C(Ri_sm, T)

        # R0_REF seeding
        R0_REF = np.nan
        try:
            warm_mask = pd.to_numeric(df[COLS["T"]], errors="coerce") >= MIN_WARM_TEMP
            early_mask = df.index < (df.index[0] + pd.to_timedelta(EARLY_DAYS, unit="D"))
            _early = df.loc[warm_mask & early_mask]
            if COLS["RI"] in df.columns and len(_early):
                R0_REF = float(pd.to_numeric(_early[COLS["RI"]], errors="coerce").quantile(0.1)) / 1000.0
        except Exception:
            pass

        # === R0_REF seeding ===
        # ### CHANGE: match Set #1: seed from R25 only (early warm quantile),
        # avoid extra df-based seed path that can bias R0 lower/higher.
        warm_on_r25 = T.reindex(R25.index).ge(MIN_WARM_TEMP).fillna(False)
        if use_awake_mask:
            warm_on_r25 &= AW
        cutoff = R25.index[0] + pd.to_timedelta(EARLY_DAYS, unit="D")
        early_on_r25 = R25.index < cutoff
        early_warm   = R25.loc[early_on_r25 & warm_on_r25].dropna()
        if len(early_warm) >= 50:
            R0_REF = float(early_warm.quantile(0.05))
        else:
            R0_REF = float(R25.quantile(0.10))
        if not np.isfinite(R0_REF):
            R0_REF = float(np.clip(float(R25.quantile(0.10)), 3e-3, 20e-3))
        else:
            R0_REF = float(np.clip(R0_REF, 3e-3, 20e-3))

        SOH_R = (R0_REF / R25).clip(0, 1.2)

        # ... (capacity tracks same as your Set #2) ...

        # Capacity tracks
        SOH_C = pd.Series(np.nan, index=r.index)
        if COLS["SOC"] in r.columns:
            SOC = pd.to_numeric(r[COLS["SOC"]], errors="coerce")
            if SOC.dropna().quantile(0.95) <= 1.5:
                SOC = SOC * 100.0
            SOC = SOC.interpolate(limit=int(120/RESAMPLE_SECONDS)).clip(0, 100)

            is_long_discharge = (r["is_discharge"] == 1) & ((AW if use_awake_mask else True))
            grp = (is_long_discharge != is_long_discharge.shift()).cumsum()
            for _, seg in r[is_long_discharge].groupby(grp):
                idx = seg.index
                if len(idx) < int(max(1, 8*60/RESAMPLE_SECONDS)):
                    continue
                soc_drop = float(SOC.loc[idx[0]] - SOC.loc[idx[-1]])
                if soc_drop < DOD_MIN_PERCENT:
                    continue
                if use_awake_mask:
                    Ah = (-I.loc[idx].clip(upper=0).sum() * RESAMPLE_SECONDS) / 3600.0
                else:
                    Ah = (I.loc[idx].abs().sum() * RESAMPLE_SECONDS) / 3600.0
                Tm = float(T.loc[idx].mean())
                C25 = Ah / (soc_drop/100.0) / (1 - ALPHA_CAP * (25.0 - Tm))
                SOH_C.loc[idx] = np.clip(C25 / C_NOM_REF, 0, 1.2)
        SOH_C = SOH_C.clip(0, 1)

        # micro-DoD fallback (rest→rest)
        SOH_C_fb = pd.Series(np.nan, index=r.index)
        if COLS["SOC"] in r.columns:
            SOCi = pd.to_numeric(r[COLS["SOC"]], errors="coerce").interpolate(limit=int(120/RESAMPLE_SECONDS)).clip(0, 100)
            rest_mask = (r["is_rest"] == 1) & ((AW if use_awake_mask else True))
            rest_groups = (rest_mask.ne(rest_mask.shift())).cumsum()
            bounds = [(g.index[0], g.index[-1]) for _, g in r[rest_mask].groupby(rest_groups)]
            for (s1, e1), (s2, e2) in zip(bounds[:-1], bounds[1:]):
                start, end = e1, s2
                if (end - start).total_seconds() < 30:
                    continue
                if use_awake_mask and (~AW.loc[start:end]).any():
                    continue
                seg = r.loc[start:end]
                if seg["is_charge"].any():
                    continue
                soc_drop = float(SOCi.loc[start] - SOCi.loc[end])
                if not np.isfinite(soc_drop) or soc_drop < 1.0:
                    continue
                if use_awake_mask:
                    Ah = (-I.loc[start:end].clip(upper=0).sum() * RESAMPLE_SECONDS) / 3600.0
                else:
                    Ah = (seg[COLS["I"]].abs().sum() * RESAMPLE_SECONDS) / 3600.0
                Tm = float(T.loc[start:end].mean())
                C25 = Ah / (soc_drop/100.0) / (1 - ALPHA_CAP * (25.0 - Tm))
                SOH_C_fb.loc[start:end] = np.clip(C25 / C_NOM_REF, 0, 1.2)

        # capacity-signal fallback
        SOH_C_cap = pd.Series(np.nan, index=r.index)
        cap_source = COLS.get("CAP_AVAIL") if (COLS.get("CAP_AVAIL") in r.columns) else \
                     (COLS.get("CAP_DISCH") if (COLS.get("CAP_DISCH") in r.columns) else None)
        if cap_source:
            cap_raw = pd.to_numeric(r[cap_source], errors="coerce")
            cap_sm  = cap_raw.rolling("15min", min_periods=1).median()
            C25_cap = cap_sm / (1 - ALPHA_CAP * (25.0 - T))
            SOH_C_cap = np.clip(C25_cap / C_NOM_REF, 0, 1.2)

        SOH_C_combined = SOH_C.combine_first(SOH_C_fb).combine_first(SOH_C_cap).clip(0, 1)

        # === Fusion weights ===
        # ### CHANGE: match Set #1 gamma formulas (no AW inside gamma)
        C_WIN = "30min"; R_WIN = "30min"
        gamma_R = (pd.notna(Ri_sm).rolling(R_WIN, min_periods=1).mean().fillna(0)) \
                  * (T.between(10, 35).rolling(R_WIN, min_periods=1).mean().fillna(0))
        gamma_C = (pd.notna(SOH_C_combined).rolling(C_WIN, min_periods=1).mean().fillna(0)) \
                  * (T.between(10, 35).rolling(C_WIN, min_periods=1).mean().fillna(0))
        wC = (gamma_C / (gamma_C + gamma_R + 1e-6)).clip(0.05, 0.95)

        SOH_final = (wC * SOH_C_combined.fillna(0) + (1 - wC) * SOH_R.ffill().bfill()).clip(0, 1)

        # Align vendor to r.index
        vendor_aligned = vendor_series.reindex(r.index).interpolate(limit=int(120/RESAMPLE_SECONDS)) \
                                          if isinstance(vendor_series, pd.Series) else pd.Series(dtype=float, index=r.index)

        return SOH_final, SOH_C_combined, SOH_R.clip(0, 1), vendor_aligned

    # ------------------- MAIN FLOW -------------------
    df0 = _ensure_time_index(dfr)

    # 1) Standard (no awake gating)
    soh_final_std, soh_c_std, soh_r_std, vendor_std = _compute_soh_final(
        df0, use_awake_mask=False, rest_i_max=REST_I_MAX_STD, rest_dv_mv=REST_DV_MV_STD
    )
    # ### CHANGE: “sparse-sample fallback” like Set #1
    std_count = soh_final_std.notna().sum()
    use_fallback = (std_count < 10)  # threshold matches Set #1

    if use_fallback:
        soh_final_fb, soh_c_fb, soh_r_fb, vendor_fb = _compute_soh_final(
            df0, use_awake_mask=True, rest_i_max=REST_I_MAX_FBK, rest_dv_mv=REST_DV_MV_FBK
        )
        # pick the one with more usable points
        if soh_final_fb.notna().sum() >= std_count:
            best = dict(final=soh_final_fb, c=soh_c_fb, r=soh_r_fb, vendor=vendor_fb)
        else:
            best = dict(final=soh_final_std, c=soh_c_std, r=soh_r_std, vendor=vendor_std)
    else:
        best = dict(final=soh_final_std, c=soh_c_std, r=soh_r_std, vendor=vendor_std)

    val = float(np.nanmedian(soh_final_std.values)) if len(soh_final_std) else float("nan")

    # Pick best
    best = dict(final=soh_final_std, c=soh_c_std, r=soh_r_std, vendor=vendor_std)

    # 2) Fallback w/ awake gating if needed
    if (not np.isfinite(val)) or np.isnan(val):
        soh_final_fb, soh_c_fb, soh_r_fb, vendor_fb = _compute_soh_final(
            df0, use_awake_mask=True, rest_i_max=REST_I_MAX_FBK, rest_dv_mv=REST_DV_MV_FBK
        )
        if len(soh_final_fb):
            val = float(np.nanmedian(soh_final_fb.values))
            best = dict(final=soh_final_fb, c=soh_c_fb, r=soh_r_fb, vendor=vendor_fb)

    # 3) Final fallbacks
    if (not np.isfinite(val)) or np.isnan(val):
        if best["vendor"].notna().any():
            val = float(best["vendor"].median(skipna=True))
        if (not np.isfinite(val)) or np.isnan(val):
            if best["r"].notna().any():
                val = float(best["r"].median(skipna=True))
        if (not np.isfinite(val)) or np.isnan(val):
            val = 1.0

    # Normalize/clamp winner
    val = float(np.clip(_normalize_percent_to_fraction(val), 0.0, 1.0))

    # --- Return section (backward compatible) ---
    if not return_components:
        return val

    if return_series:
        base = pd.DataFrame({
            "SOH1":  best["final"],
            "SOH_C": best["c"],
            "SOH_R": best["r"],
        }).sort_index()

        if return_vendor:
            base["SOH_VENDOR"] = best["vendor"].clip(0, 1)

        if output_s is None:
            # keep native (resampled) cadence
            return base

        # Build a 1-second target index spanning the original input timestamps
        start = df0.index[0].ceil("S")
        end   = df0.index[-1].floor("S")
        target_index = pd.date_range(start, end, freq=f"{output_s}s")

        # Option A (smooth line): time interpolation
        df_out = (base.reindex(target_index)
                        .interpolate(method="time", limit_direction="both")
                        .ffill().bfill())

        # Optional – if you prefer step-like behavior instead:
        # df_out = base.reindex(target_index, method="nearest", tolerance=pd.Timedelta(seconds=output_s))

        return df_out
    
    # medians
    med_soh1 = float(np.nanmedian(best["final"].values)) if len(best["final"]) else np.nan
    med_soh_c = float(np.nanmedian(best["c"].values)) if len(best["c"]) else np.nan
    med_soh_r = float(np.nanmedian(best["r"].values)) if len(best["r"]) else np.nan
    # Normalize and clamp
    def _norm(x): 
        return float(np.clip(x/100.0 if (np.isfinite(x) and x>1.5) else x, 0.0, 1.0)) if np.isfinite(x) else np.nan
    return (_norm(med_soh1), _norm(med_soh_c), _norm(med_soh_r))




def _resolve_time_index(dfr: pd.DataFrame, time_col: Optional[str]) -> pd.DataFrame:
    df = dfr.copy()
    if time_col is not None and time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        df = df.sort_values(time_col).set_index(time_col)
    elif not isinstance(df.index, pd.DatetimeIndex):
        # try common names if time_col not provided
        for c in ["time_utc","time","timestamp","Timestamp","time_utc_iso"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
                df = df.sort_values(c).set_index(c)
                break
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("No DatetimeIndex and no usable time column found.")
    return df

def _pick_voltage_column(df: pd.DataFrame, user_col: Optional[str] = None) -> str:
    if user_col and user_col in df.columns:
        return user_col
    candidates = [
        "VCU_IBS_UIT.IBS_BatteryVoltage",
        "IBS_BatteryVoltage",
        "battery_voltage",
        "Voltage", "voltage", "V"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError("Voltage column not found. Pass vol_col explicitly.")


def detect_cycles_from_voltage(
    v: pd.Series,
    th_on_v: float = 13.4,   # rise above => charging
    th_off_v: float = 13.2,  # fall below => not charging
    min_len_s: int = 10
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Hysteresis on voltage; returns list of (start_ts, end_ts) for charge windows."""
    v = v.sort_index()
    charging = False
    t0 = None
    cycles: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for t, val in zip(v.index, v.values):
        if not charging and val >= th_on_v:
            charging = True; t0 = t
        elif charging and val <= th_off_v:
            charging = False
            if t0 is not None:
                dur = (t - t0).total_seconds()
                if dur >= min_len_s:
                    cycles.append((t0, t))
            t0 = None
    # if still charging at the end, close the window
    if charging and t0 is not None:
        t1 = v.index[-1]
        if (t1 - t0).total_seconds() >= min_len_s:
            cycles.append((t0, t1))
    return cycles


def count_cycles_voltage(
    dfr: pd.DataFrame,
    time_col: Optional[str] = None,
    vol_col: Optional[str] = None,
    resample: str = "1S",
    median_win_s: int = 3,       # small de-noise; set 0 to disable
    th_on_v: float = 13.4,
    th_off_v: float = 13.2,
    min_len_s: int = 10,
    enforce_min_one: bool = True # keep your previous rule if you want
) -> Tuple[int, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    """
    Count charge cycles using voltage hysteresis.
    Returns (count, list_of_cycles).
    """
    df = _resolve_time_index(dfr, time_col)
    v_col = _pick_voltage_column(df, vol_col)

    # numeric + resample to a uniform grid
    v = pd.to_numeric(df[v_col], errors="coerce")
    vs = v.resample(resample).median().ffill()

    # optional light smoothing to suppress single-sample flicker
    if median_win_s and median_win_s > 1:
        vs = vs.rolling(f"{median_win_s}S", min_periods=1).median()

    # detect cycles
    cycles = detect_cycles_from_voltage(vs, th_on_v, th_off_v, min_len_s)
    n = len(cycles)
    if enforce_min_one:
        n = max(1, n)
    return n, cycles


def merge_short_gaps(cycles_windows, max_gap_s=5):
    """Merge consecutive charge windows separated by short gaps."""
    if not cycles_windows:
        return cycles_windows
    merged = [cycles_windows[0]]
    for s, e in cycles_windows[1:]:
        ps, pe = merged[-1]
        gap = (s - pe).total_seconds()
        if gap <= max_gap_s:
            merged[-1] = (ps, e)  # extend previous
        else:
            merged.append((s, e))
    return merged


# Extract log date from first timestamp in cleaned can data log
def parse_log_date(dfr, time_cols=("time","time_utc","time_utc_iso")):
    for c in time_cols:
        if c in dfr.columns:
            t = pd.to_datetime(dfr[c], errors="coerce", utc=True)
            if t.notna().any():
                return t.min().tz_convert(None) if getattr(t.dt, "tz", None) else t.min()
    # fallback: try a YYYY-MM-DD in filename-like strings
    return pd.NaT


# Extract IBS SOH from median IBS_StateOfHealth
def compute_log_soh(dfr):
    for col in ["IBS_StateOfHealth"]:
        if col in dfr.columns:
            s = pd.to_numeric(dfr[col], errors="coerce")
            if s.notna().any():
                # robust central tendency (use median)
                return float(s.median())
    # last resort: NaN
    return np.nan


def ensure_dt_index(dfr: pd.DataFrame, time_col: str | None = "time_utc", assume_utc: bool = True) -> pd.DataFrame:
    df = dfr.copy()
    # Case A: already a DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
        if idx.tz is None and assume_utc:
            df.index = idx.tz_localize("UTC")
        elif idx.tz is not None and assume_utc:
            df.index = idx.tz_convert("UTC")
        return df.sort_index()
    # Case B: parse a known time column
    candidates = [c for c in [time_col, "timestamp", "time", "ts", "time_utc"] if c and c in df.columns]
    if candidates:
        c = candidates[0]
        ts = pd.to_datetime(df[c], errors="coerce", utc=assume_utc)
        df = df.assign(_ts=ts).dropna(subset=["_ts"]).sort_values("_ts").set_index("_ts").drop(columns=[c])
        df.index.name = c
        return df
    # Case C: auto-detect a datetime-like column
    for c in df.columns:
        parsed = pd.to_datetime(df[c], errors="coerce", utc=assume_utc)
        if parsed.notna().sum() >= max(3, int(0.5 * len(df))):
            df = df.assign(_ts=parsed).dropna(subset=["_ts"]).sort_values("_ts").set_index("_ts")
            df.index.name = c
            return df
    raise ValueError("Could not find a datetime index or suitable time column to parse.")


def ensure_time_index_1s(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has a DatetimeIndex with 1-second uniform cadence.
    - If index is not datetime, tries common time columns.
    - Drops NaN timestamps.
    - Removes timezone if present.
    - Resamples all numeric columns to exactly 1 s frequency using median.
    """
    df = df.copy()

    # Step 1: Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        for c in ("time_utc", "time", "timestamp", "DateTime"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
                df = df.dropna(subset=[c]).set_index(c)
                break
        else:
            raise ValueError("No datetime index or usable time column found.")

    # Step 2: Remove timezone info (optional)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    # Step 3: Sort by time
    df = df.sort_index()

    # Step 4: Resample to exactly 1 s cadence
    df_1s = df.resample("1S").median().interpolate(limit=3)

    return df_1s


def attach_soh1_series(df: pd.DataFrame,
                       resample_s: int = 1,
                       tolerance: str = "1s",
                       include_components: bool = True,
                       verbose: bool = True) -> pd.DataFrame:
    """
    Compute SOH1 (and optionally SOH_C/SOH_R) and align to df's 1 s index.
    Robust tz handling + fast-path reindex-nearest to avoid empty merges.
    """
    # --- Ensure df has a DatetimeIndex & normalize tz to NAIVE ---
    # (naive == no tz); this avoids tz-aware vs naive mismatches later
    if not isinstance(df.index, pd.DatetimeIndex):
        for c in ("time","time_utc","timestamp","DateTime"):
            if c in df.columns:
                df = df.copy()
                df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
                df = df.set_index(c)
                break
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df = df.sort_index()

    # Already present?
    if "soh1" in df.columns and pd.to_numeric(df["soh1"], errors="coerce").notna().sum() > 0:
        if verbose:
            print(f"[attach_soh1_series] Using existing soh1 (valid={df['soh1'].notna().sum()}/{len(df)}).")
        return df

    # --- Compute series at 1 s cadence ---
    ser = calculate_soh1(
        df,
        return_components=True,
        return_series=True,
        resample_s=resample_s,
        return_vendor=False
    )
    if not isinstance(ser, pd.DataFrame):
        # fallback: scalar/tuple — attach constant to keep pipeline running
        val = ser[0] if isinstance(ser, (list, tuple)) and len(ser) else ser
        df["soh1"] = pd.to_numeric(val, errors="coerce")
        if verbose:
            print(f"[attach_soh1_series] WARNING: got scalar/tuple; attached constant soh1={df['soh1'].iloc[0]:.3f}")
        return df

    rs = ser.rename(columns={"SOH1":"soh1","SOH_C":"soh_c","SOH_R":"soh_r"}).copy()

    # Normalize tz on returned series too
    if not isinstance(rs.index, pd.DatetimeIndex):
        rs.index = pd.to_datetime(rs.index, errors="coerce", utc=True)
    if rs.index.tz is not None:
        rs.index = rs.index.tz_convert(None)
    rs = rs.sort_index()

    # --- FAST PATH: reindex-nearest onto df.index ---
    tol = pd.to_timedelta(tolerance)
    try:
        aligned = rs.reindex(df.index, method="nearest", tolerance=tol)
    except Exception:
        aligned = None

    # If fast path failed or produced all-NaN, FALL BACK to merge_asof
    if (aligned is None) or (aligned["soh1"].notna().sum() == 0):
        left = df.index.to_series().reset_index(drop=True).to_frame("t")
        right = rs.index.to_series().reset_index(drop=True).to_frame("t_rs")
        right = right.join(rs.reset_index(drop=True))
        aligned = pd.merge_asof(
            left, right, left_on="t", right_on="t_rs",
            direction="nearest", tolerance=tol
        ).drop(columns=["t_rs"]).set_index(df.index)

    # Write columns back
    for c in ("soh1","soh_c","soh_r"):
        if c in aligned.columns and (include_components or c == "soh1"):
            df[c] = pd.to_numeric(aligned[c], errors="coerce")

    # Normalize to fraction if % slipped in
    if "soh1" in df.columns:
        smax = pd.to_numeric(df["soh1"], errors="coerce").max(skipna=True)
        if pd.notna(smax) and smax > 1.5:
            df["soh1"] = pd.to_numeric(df["soh1"], errors="coerce") / 100.0

    # --- Debug prints ---
    if verbose:
        cnt = pd.to_numeric(df["soh1"], errors="coerce").notna().sum()
        total = len(df)
        pct = (cnt/total*100) if total else 0
        print(f"[attach_soh1_series] soh1 coverage: {cnt}/{total}  ({pct:.1f}%)  tol={tolerance}")
        print(f"                     df idx tz: NAIVE, rs idx tz: NAIVE")
        print(f"                     df range: {df.index.min()} → {df.index.max()}")
        print(f"                     rs range: {rs.index.min()} → {rs.index.max()}")

    return df


def ensure_time_utc(
    dfr: pd.DataFrame,
    preferred_sources=("time_utc", "time_utc_iso", "time", "excel_utc", "time_et_iso", "timestamp", "ts"),
    drop_others=True,
    floor_unit="us",
):
    """
    Ensure a tz-aware UTC 'time_utc' column exists, parsing from the best available source.
    Robust to ISO strings, epoch (s/ms/us/ns), and Excel serial dates.

    - If 'time_utc' already exists, it will be normalized and left in place.
    - If not, the first available column in `preferred_sources` will be parsed.
    - Optionally drops other time-like columns.

    Parameters
    ----------
    dfr : pd.DataFrame
    preferred_sources : tuple[str]
        Ordered candidates to parse. First present is used.
    drop_others : bool
        If True, drop other time-ish columns after creating 'time_utc'.
    floor_unit : str
        Precision normalization for time_utc ('us' by default).

    Returns
    -------
    pd.DataFrame
        Copy of input with a normalized 'time_utc' column.
    """
    df = dfr.copy()

    def _as_utc_datetime(s: pd.Series) -> pd.Series:
        # Already datetime?
        if pd.api.types.is_datetime64_any_dtype(s):
            out = s.copy()
            if out.dt.tz is None:
                out = out.dt.tz_localize("UTC")
            else:
                out = out.dt.tz_convert("UTC")
            return out.dt.floor(floor_unit)

        # Try ISO-like parse first
        iso = pd.to_datetime(s, errors="coerce", utc=True)
        if iso.notna().sum() >= max(3, int(0.5 * len(s))):
            return iso.dt.floor(floor_unit)

        # Numeric heuristics (epoch or Excel)
        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.notna().sum() == 0:
            return iso  # all NaT

        vmax = s_num.quantile(0.9)  # robust to outliers

        # Excel serial date range (approx typical range)
        if 20000 <= vmax <= 60000:
            out = pd.to_datetime(s_num, unit="D", origin="1899-12-30", utc=True, errors="coerce")
            return out.dt.floor(floor_unit)

        # Epoch unit by magnitude
        if vmax < 1e11:      unit = "s"   # ~seconds since 1970
        elif vmax < 1e14:    unit = "ms"  # milliseconds
        elif vmax < 1e17:    unit = "us"  # microseconds
        else:                unit = "ns"  # nanoseconds

        out = pd.to_datetime(s_num, unit=unit, utc=True, errors="coerce")
        return out.dt.floor(floor_unit)

    # 1) If time_utc already exists, normalize & return quickly
    if "time_utc" in df.columns:
        df["time_utc"] = _as_utc_datetime(df["time_utc"])
    else:
        # 2) Pick first available source
        source = next((c for c in preferred_sources if c in df.columns), None)
        if source is None:
            raise KeyError(
                "No time source column found. Looked for: " + ", ".join(preferred_sources)
            )
        parsed = _as_utc_datetime(df[source])

        # Insert 'time_utc' right after source (or at front if not possible)
        try:
            insert_at = min(df.columns.get_loc(source) + 1, len(df.columns))
        except KeyError:
            insert_at = 0
        if "time_utc" in df.columns:
            df = df.drop(columns=["time_utc"])
        df.insert(insert_at, "time_utc", parsed)

    # 3) Optional cleanup of other time-ish columns
    if drop_others:
        to_drop = [c for c in ["time", "time_utc_iso", "time_et_iso", "excel_utc", "timestamp", "ts"]
                   if c in df.columns and c != "time_utc"]
        if to_drop:
            df = df.drop(columns=to_drop)

    # 4) Visibility into bad rows
    n_nat = df["time_utc"].isna().sum()
    if n_nat:
        print(f"Warning: {n_nat} rows have unparseable timestamps (time_utc is NaT).")

    return df
