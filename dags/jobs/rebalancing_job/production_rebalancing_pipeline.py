# %% [0] module_docstring
"""
Adaptive regime-switching rebalancing for S&P 500 + defensive assets.

What this script does:
1. Builds the S&P 500 universe from local CSV / public CSV / SPY holdings / Wikipedia / cache, without hardcoded constituent lists.
2. Adds defensive ETFs: Treasuries, TIPS, gold, defensive sectors, USD.
3. Downloads daily OHLCV from Yahoo Finance via yfinance.
4. Creates technical and market-regime features.
5. Selects a forecasting model from a registry using walk-forward validation.
6. Compares several existing rebalancing algorithms.
7. Builds an adaptive strategy that switches rebalancing logic by market regime.

This is research/backtest code, not investment advice.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import time
import warnings
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
import io
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    yf = None


# %% [1] imports_config
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

SP500_CSV_SOURCES = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
    "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv",
)

SP500_SPY_HOLDINGS_XLSX_URL = "https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx"

SP500_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

DEFENSIVE_ASSETS = {
    "BIL": "1-3M Treasury Bill ETF",
    "SHY": "1-3Y Treasury ETF",
    "IEF": "7-10Y Treasury ETF",
    "TLT": "20Y+ Treasury ETF",
    "TIP": "TIPS ETF",
    "GLD": "Gold ETF",
    "IAU": "Gold ETF",
    "UUP": "US Dollar Index ETF",
    "XLU": "Utilities Sector ETF",
    "XLP": "Consumer Staples Sector ETF",
    "XLV": "Health Care Sector ETF",
}

FEATURE_ASSETS = {
    "SPY": "S&P 500 ETF market proxy",
    "^VIX": "CBOE Volatility Index",
}


@dataclass
class BacktestConfig:
    start: str = "2010-01-01"
    end: Optional[str] = None
    horizon: int = 21
    train_window: int = 756
    validation_window: int = 63
    rebalance_freq: Optional[int] = None
    retrain_freq: int = 63
    cov_lookback: int = 252
    transaction_cost: float = 0.001
    max_weight: float = 0.12
    top_n_equity: int = 80
    max_positions_per_strategy: Optional[int] = None
    min_history_rows: int = 300
    cache_path: str = "sp500_defensive_yfinance_cache.pkl"
    refresh_cache: bool = False
    random_state: int = 42

    sp500_local_path: Optional[str] = "sp500_tickers.csv"
    sp500_universe_cache_path: str = "sp500_universe_cache.csv"
    sp500_min_count: int = 450
    sp500_source_priority: Tuple[str, ...] = (
        "local",
        "csv_github",
        "ssga_spy_holdings",
        "wikipedia",
        "universe_cache",
    )

    defensive_tickers: Tuple[str, ...] = field(default_factory=lambda: tuple(DEFENSIVE_ASSETS.keys()))
    feature_tickers: Tuple[str, ...] = field(default_factory=lambda: tuple(FEATURE_ASSETS.keys()))

    production_mode: bool = True
    output_dir: str = "rebalancing_prod_runs"
    run_name: Optional[str] = None
    run_backtest: bool = True
    live_as_of_date: Optional[str] = None
    min_tradable_assets: int = 100
    max_data_lag_days: int = 10
    fail_on_data_quality: bool = False
    strategy_names: Tuple[str, ...] = (
        "equal_weight_all",
        "topn_equal",
        "inverse_vol",
        "min_variance",
        "risk_parity",
        "mean_variance",
        "max_sharpe",
        "defensive_min_variance",
        "adaptive_regime_switch",
    )
    final_strategy: str = "adaptive_regime_switch"
    save_artifacts: bool = True

    download_chunk_size: int = 10
    download_sleep_sec: float = 2.0
    download_max_retries: int = 4
    download_threads: bool = False
    download_timeout: int = 30
    download_retry_failed_individual: bool = True
    download_individual_sleep_sec: float = 0.5
    download_silent: bool = True

    def __post_init__(self) -> None:
        self.max_weight = normalize_weight_limit(self.max_weight)
        if self.max_positions_per_strategy is not None:
            self.max_positions_per_strategy = int(self.max_positions_per_strategy)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}")


def yahoo_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper()
    s = s.replace(".", "-")
    return s


def normalize_weight_limit(max_weight: float) -> float:
    value = float(max_weight)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"max_weight must be positive, got {max_weight}")
    if value > 1.0:
        value = value / 100.0
    if value > 1.0:
        raise ValueError(f"max_weight is too large after percent conversion: {max_weight}")
    return value


def min_positions_for_weight_limit(max_weight: float) -> int:
    limit = normalize_weight_limit(max_weight)
    return int(math.ceil(1.0 / limit))


# %% [2] universe_loading
def _download_bytes(url: str, timeout: int = 30) -> bytes:
    from urllib.request import Request, urlopen

    req = Request(
        url,
        headers={
            "User-Agent": SP500_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml,text/csv,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _download_text(url: str, timeout: int = 30) -> str:
    raw = _download_bytes(url, timeout=timeout)
    return raw.decode("utf-8", errors="replace")


def _first_existing_col(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    normalized = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def _standardize_sp500_table(sp500: pd.DataFrame, source: str = "unknown") -> pd.DataFrame:
    if sp500 is None or sp500.empty:
        raise ValueError(f"{source}: empty S&P 500 table")

    sp500 = sp500.copy()
    sp500.columns = [str(c).strip() for c in sp500.columns]

    ticker_col = _first_existing_col(
        sp500.columns,
        [
            "ticker",
            "ticker_raw",
            "symbol",
            "Symbol",
            "Ticker",
            "Ticker symbol",
            "Ticker Symbol",
            "Holding Ticker",
            "Ticker Identifier",
            "Identifier",
        ],
    )
    name_col = _first_existing_col(
        sp500.columns,
        [
            "Security",
            "security",
            "Name",
            "name",
            "Company",
            "company",
            "Holding Name",
            "Security Name",
            "Description",
        ],
    )
    sector_col = _first_existing_col(
        sp500.columns,
        [
            "GICS Sector",
            "Sector",
            "sector",
            "Fund Sector",
            "Index Sector",
        ],
    )
    sub_industry_col = _first_existing_col(
        sp500.columns,
        [
            "GICS Sub-Industry",
            "GICS Sub Industry",
            "Sub-Industry",
            "Sub Industry",
            "Subindustry",
            "Industry",
        ],
    )
    weight_col = _first_existing_col(sp500.columns, ["Weight", "weight", "% Weight", "Portfolio Weight"])

    if ticker_col is None:
        raise ValueError(f"{source}: ticker column not found. Columns: {list(sp500.columns)}")

    out = pd.DataFrame()
    out["ticker_raw"] = sp500[ticker_col].astype(str).str.strip().str.upper()
    out["ticker"] = out["ticker_raw"].map(yahoo_symbol)

    out["name"] = sp500[name_col].astype(str).str.strip() if name_col else out["ticker_raw"]
    out["gics_sector"] = sp500[sector_col].astype(str).str.strip() if sector_col else "Unknown"
    out["gics_sub_industry"] = sp500[sub_industry_col].astype(str).str.strip() if sub_industry_col else "Unknown"

    if weight_col:
        out["index_weight"] = (
            sp500[weight_col]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.extract(r"([-+]?\d*\.?\d+)", expand=False)
            .astype(float)
        )
    else:
        out["index_weight"] = np.nan

    out["source"] = source
    out["asset_group"] = "sp500_equity"
    out["is_defensive"] = False

    bad_tickers = {
        "",
        "NAN",
        "NONE",
        "NULL",
        "SYMBOL",
        "TICKER",
        "CASH",
        "CASH_USD",
        "USD",
        "US DOLLAR",
        "US DOLLARS",
    }
    out = out[~out["ticker"].isin(bad_tickers)].copy()
    out = out[out["ticker"].str.match(r"^[A-Z\^][A-Z0-9\-\.]{0,12}$", na=False)].copy()
    out = out.drop_duplicates("ticker").reset_index(drop=True)

    keep_cols = [
        "ticker",
        "ticker_raw",
        "name",
        "gics_sector",
        "gics_sub_industry",
        "asset_group",
        "is_defensive",
        "index_weight",
        "source",
    ]
    return out[keep_cols]


def _validate_sp500_universe(sp500: pd.DataFrame, source: str, min_count: int) -> pd.DataFrame:
    if sp500 is None or sp500.empty:
        raise ValueError(f"{source}: empty universe")
    if len(sp500) < min_count:
        raise ValueError(f"{source}: too few tickers after cleaning: {len(sp500)} < {min_count}")
    return sp500


def _sp500_from_local_file(path: Optional[str], min_count: int) -> pd.DataFrame:
    if not path:
        raise FileNotFoundError("local path is not set")

    candidates = [path]
    if not os.path.isabs(path):
        candidates.extend(
            [
                os.path.join(os.getcwd(), path),
                os.path.join("/mnt/data", path),
            ]
        )

    existing_path = next((p for p in dict.fromkeys(candidates) if os.path.exists(p)), None)
    if existing_path is None:
        raise FileNotFoundError(f"local S&P 500 file not found: {path}")

    local = pd.read_csv(existing_path)
    sp500 = _standardize_sp500_table(local, source=f"local:{existing_path}")
    return _validate_sp500_universe(sp500, f"local:{existing_path}", min_count=min_count)


def _sp500_from_csv_sources(min_count: int) -> pd.DataFrame:
    errors = []
    for url in SP500_CSV_SOURCES:
        try:
            text = _download_text(url)
            csv_df = pd.read_csv(StringIO(text))
            sp500 = _standardize_sp500_table(csv_df, source=url)
            return _validate_sp500_universe(sp500, url, min_count=min_count)
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
    raise RuntimeError("CSV sources failed: " + " | ".join(errors))


def _sp500_from_wikipedia(min_count: int) -> pd.DataFrame:
    text = _download_text(SP500_WIKI_URL)
    tables = pd.read_html(StringIO(text))
    if not tables:
        raise ValueError("Wikipedia: no tables found")
    sp500 = _standardize_sp500_table(tables[0], source="wikipedia")
    return _validate_sp500_universe(sp500, "wikipedia", min_count=min_count)


def _find_header_row(raw: pd.DataFrame, max_rows: int = 40) -> Optional[int]:
    max_rows = min(max_rows, len(raw))
    ticker_words = {"ticker", "symbol", "identifier", "holding ticker"}
    name_words = {"name", "security", "holding name", "description"}

    for i in range(max_rows):
        values = [str(x).strip().lower() for x in raw.iloc[i].tolist()]
        has_ticker = any(v in ticker_words or "ticker" in v or "symbol" in v for v in values)
        has_name = any(v in name_words or "name" in v or "security" in v for v in values)
        if has_ticker and has_name:
            return i
    return None


def _sp500_from_ssga_spy_holdings(min_count: int) -> pd.DataFrame:
    from io import BytesIO

    raw_bytes = _download_bytes(SP500_SPY_HOLDINGS_XLSX_URL)
    sheets = pd.read_excel(BytesIO(raw_bytes), sheet_name=None, header=None)
    errors = []

    for sheet_name, raw in sheets.items():
        try:
            header_row = _find_header_row(raw)
            if header_row is None:
                continue
            data = raw.iloc[header_row + 1 :].copy()
            data.columns = [str(x).strip() for x in raw.iloc[header_row].tolist()]
            data = data.dropna(how="all")
            sp500 = _standardize_sp500_table(data, source=f"ssga_spy_holdings:{sheet_name}")
            return _validate_sp500_universe(sp500, f"ssga_spy_holdings:{sheet_name}", min_count=min_count)
        except Exception as exc:
            errors.append(f"{sheet_name}: {type(exc).__name__}: {exc}")

    raise RuntimeError("SSGA SPY holdings failed: " + " | ".join(errors))


def _sp500_from_universe_cache(path: str, min_count: int) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"universe cache not found: {path}")
    cached = pd.read_csv(path)
    sp500 = _standardize_sp500_table(cached, source=f"cache:{path}")
    return _validate_sp500_universe(sp500, f"cache:{path}", min_count=min_count)


def _save_sp500_universe_cache(sp500: pd.DataFrame, path: str) -> None:
    if not path:
        return
    try:
        sp500.to_csv(path, index=False)
        log(f"S&P 500 universe cache saved: {path}")
    except Exception as exc:
        log(f"Could not save S&P 500 universe cache: {exc}")


def get_sp500_universe(config: Optional[BacktestConfig] = None) -> pd.DataFrame:
    config = config or BacktestConfig()

    loaders = {
        "local": lambda: _sp500_from_local_file(config.sp500_local_path, config.sp500_min_count),
        "csv_github": lambda: _sp500_from_csv_sources(config.sp500_min_count),
        "wikipedia": lambda: _sp500_from_wikipedia(config.sp500_min_count),
        "ssga_spy_holdings": lambda: _sp500_from_ssga_spy_holdings(config.sp500_min_count),
        "universe_cache": lambda: _sp500_from_universe_cache(config.sp500_universe_cache_path, config.sp500_min_count),
    }

    errors = []
    for source_name in config.sp500_source_priority:
        loader = loaders.get(source_name)
        if loader is None:
            errors.append(f"{source_name}: unknown source")
            continue
        try:
            sp500 = loader()
            log(f"Loaded S&P 500 universe from {source_name}: {len(sp500)} tickers")
            if source_name != "universe_cache":
                _save_sp500_universe_cache(sp500, config.sp500_universe_cache_path)
            return sp500
        except Exception as exc:
            errors.append(f"{source_name}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "Could not load S&P 500 universe from any source. "
        "Put a file with columns ticker/Symbol and optionally name/Security/Sector into sp500_local_path, "
        "or allow internet access. Errors: " + " | ".join(errors)
    )


def get_defensive_universe(config: Optional[BacktestConfig] = None) -> pd.DataFrame:
    config = config or BacktestConfig()
    rows = []
    for ticker in config.defensive_tickers:
        ticker = yahoo_symbol(ticker)
        rows.append(
            {
                "ticker": ticker,
                "ticker_raw": ticker,
                "name": DEFENSIVE_ASSETS.get(ticker, ticker),
                "gics_sector": "Defensive ETF",
                "gics_sub_industry": DEFENSIVE_ASSETS.get(ticker, ticker),
                "asset_group": "defensive_asset",
                "is_defensive": True,
                "index_weight": np.nan,
                "source": "manual_defensive_assets",
            }
        )
    return pd.DataFrame(rows)


def get_feature_universe(config: Optional[BacktestConfig] = None) -> pd.DataFrame:
    config = config or BacktestConfig()
    rows = []
    for ticker in config.feature_tickers:
        ticker = str(ticker).strip().upper()
        rows.append(
            {
                "ticker": ticker,
                "ticker_raw": ticker,
                "name": FEATURE_ASSETS.get(ticker, ticker),
                "gics_sector": "Market Feature",
                "gics_sub_industry": FEATURE_ASSETS.get(ticker, ticker),
                "asset_group": "market_feature",
                "is_defensive": False,
                "index_weight": np.nan,
                "source": "manual_market_features",
            }
        )
    return pd.DataFrame(rows)


def build_universe(config: Optional[BacktestConfig] = None) -> pd.DataFrame:
    config = config or BacktestConfig()
    sp500 = get_sp500_universe(config)
    defensive = get_defensive_universe(config)
    features = get_feature_universe(config)
    meta = pd.concat([sp500, defensive, features], ignore_index=True)
    meta = meta.drop_duplicates("ticker").reset_index(drop=True)
    return meta


def _normalize_yfinance_download(raw: pd.DataFrame, tickers: List[str]) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    frames = []
    failed = []

    if raw is None or raw.empty:
        return pd.DataFrame(), [(t, "empty yfinance response") for t in tickers]

    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0).astype(str)
        lvl1 = raw.columns.get_level_values(1).astype(str)
        ticker_set = set(tickers)
        if len(ticker_set.intersection(set(lvl0))) == 0 and len(ticker_set.intersection(set(lvl1))) > 0:
            raw = raw.swaplevel(axis=1).sort_index(axis=1)
    else:
        if len(tickers) == 1:
            raw = pd.concat({tickers[0]: raw}, axis=1)
        else:
            return pd.DataFrame(), [(t, "unexpected non-MultiIndex response") for t in tickers]

    available = set(raw.columns.get_level_values(0).astype(str))

    for ticker in tickers:
        if ticker not in available:
            failed.append((ticker, "not returned by yfinance"))
            continue

        part = raw[ticker].copy()
        if part.empty:
            failed.append((ticker, "empty ticker frame"))
            continue

        part = part.reset_index()
        rename = {
            "Date": "dt",
            "Datetime": "dt",
            "index": "dt",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
        part = part.rename(columns=rename)
        if "dt" not in part.columns:
            part = part.rename(columns={part.columns[0]: "dt"})
        part["ticker"] = ticker

        keep = [c for c in ["dt", "ticker", "open", "high", "low", "close", "adj_close", "volume"] if c in part.columns]
        part = part[keep]
        part["dt"] = pd.to_datetime(part["dt"], errors="coerce").dt.tz_localize(None).dt.normalize()

        if "close" not in part.columns and "adj_close" in part.columns:
            part["close"] = part["adj_close"]
        if "adj_close" not in part.columns and "close" in part.columns:
            part["adj_close"] = part["close"]

        for col in ["open", "high", "low", "close", "adj_close", "volume"]:
            if col in part.columns:
                part[col] = pd.to_numeric(part[col], errors="coerce")

        part = part.dropna(subset=["dt", "close"])
        part = part[part["close"] > 0]
        if part.empty:
            failed.append((ticker, "empty after cleaning"))
            continue

        frames.append(part)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out, failed



def _yf_download_safe(
    tickers: List[str],
    start: str,
    end: Optional[str],
    threads: bool,
    timeout: int,
    silent: bool,
) -> pd.DataFrame:
    if yf is None:
        raise ImportError("Install yfinance first: pip install yfinance")

    kwargs = dict(
        tickers=tickers if len(tickers) > 1 else tickers[0],
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=threads,
        progress=False,
        timeout=timeout,
        repair=True,
    )

    def call_download() -> pd.DataFrame:
        try:
            return yf.download(**kwargs)
        except TypeError:
            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop("repair", None)
            try:
                return yf.download(**fallback_kwargs)
            except TypeError:
                fallback_kwargs.pop("timeout", None)
                return yf.download(**fallback_kwargs)

    if silent:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return call_download()
    return call_download()


# %% [3] price_download_and_cache
def download_ohlcv(
    tickers: Iterable[str],
    start: str = "2010-01-01",
    end: Optional[str] = None,
    chunk_size: int = 10,
    sleep_sec: float = 2.0,
    max_retries: int = 4,
    threads: bool = False,
    timeout: int = 30,
    retry_failed_individual: bool = True,
    individual_sleep_sec: float = 0.5,
    silent: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if yf is None:
        raise ImportError("Install yfinance first: pip install yfinance")

    tickers = list(dict.fromkeys([str(t).strip().upper() for t in tickers if str(t).strip()]))
    frames: List[pd.DataFrame] = []
    failed_map: Dict[str, str] = {}
    downloaded: set = set()

    for i in range(0, len(tickers), chunk_size):
        original_chunk = tickers[i : i + chunk_size]
        pending = original_chunk.copy()
        chunk_frames: List[pd.DataFrame] = []
        log(f"Downloading {i + 1}-{i + len(original_chunk)} / {len(tickers)}")

        for attempt in range(max_retries + 1):
            if not pending:
                break
            try:
                raw = _yf_download_safe(
                    pending,
                    start=start,
                    end=end,
                    threads=threads,
                    timeout=timeout,
                    silent=silent,
                )
                part, _ = _normalize_yfinance_download(raw, pending)
                if not part.empty:
                    got = set(part["ticker"].unique())
                    chunk_frames.append(part)
                    downloaded.update(got)
                    pending = [t for t in pending if t not in got]
                if not pending:
                    break
                failed_reason = "empty or partial yfinance response"
            except Exception as exc:
                failed_reason = repr(exc)

            if attempt < max_retries:
                time.sleep(max(sleep_sec, 1.0) * (attempt + 1))
            else:
                for t in pending:
                    failed_map[t] = failed_reason

        if pending and retry_failed_individual:
            still_pending = []
            for t in pending:
                try:
                    raw = _yf_download_safe(
                        [t],
                        start=start,
                        end=end,
                        threads=False,
                        timeout=max(timeout, 30),
                        silent=silent,
                    )
                    part, _ = _normalize_yfinance_download(raw, [t])
                    if not part.empty:
                        chunk_frames.append(part)
                        downloaded.add(t)
                        failed_map.pop(t, None)
                    else:
                        still_pending.append(t)
                        failed_map[t] = "empty response on individual retry"
                except Exception as exc:
                    still_pending.append(t)
                    failed_map[t] = repr(exc)
                time.sleep(individual_sleep_sec)
            pending = still_pending

        if chunk_frames:
            chunk_df = pd.concat(chunk_frames, ignore_index=True)
            chunk_df = chunk_df.drop_duplicates(["dt", "ticker"]).sort_values(["dt", "ticker"])
            frames.append(chunk_df)

        if pending:
            log(f"Chunk incomplete after retries: {len(pending)} / {len(original_chunk)} tickers still missing")

        time.sleep(sleep_sec)

    if not frames:
        raise RuntimeError("No market data downloaded. Check internet access and ticker list.")

    df_all = pd.concat(frames, ignore_index=True)
    df_all = (
        df_all.drop_duplicates(["dt", "ticker"])
        .sort_values(["dt", "ticker"])
        .reset_index(drop=True)
    )

    failed_rows = [(t, err) for t, err in sorted(failed_map.items()) if t not in downloaded]
    failed_df = pd.DataFrame(failed_rows, columns=["ticker", "error"]) if failed_rows else pd.DataFrame(columns=["ticker", "error"])
    return df_all, failed_df


def build_price_dataset(config: BacktestConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if config.cache_path and os.path.exists(config.cache_path) and not config.refresh_cache:
        log(f"Reading cache: {config.cache_path}")
        with open(config.cache_path, "rb") as f:
            cached = pickle.load(f)
        return cached["df_all"], cached["meta"], cached["failed"]

    meta = build_universe(config)
    all_tickers = meta["ticker"].tolist()
    df_all, failed = download_ohlcv(
        all_tickers,
        start=config.start,
        end=config.end,
        chunk_size=config.download_chunk_size,
        sleep_sec=config.download_sleep_sec,
        max_retries=config.download_max_retries,
        threads=config.download_threads,
        timeout=config.download_timeout,
        retry_failed_individual=config.download_retry_failed_individual,
        individual_sleep_sec=config.download_individual_sleep_sec,
        silent=config.download_silent,
    )

    good_tickers = set(df_all["ticker"].unique())
    meta = meta[meta["ticker"].isin(good_tickers)].copy().reset_index(drop=True)

    if config.cache_path:
        with open(config.cache_path, "wb") as f:
            pickle.dump({"df_all": df_all, "meta": meta, "failed": failed}, f)
        log(f"Cache saved: {config.cache_path}")

    return df_all, meta, failed


def make_wide_prices(df_all: pd.DataFrame, value_col: str = "close") -> pd.DataFrame:
    return (
        df_all.pivot_table(index="dt", columns="ticker", values=value_col, aggfunc="last")
        .sort_index()
    )


def calc_rsi(price: pd.Series, window: int = 14) -> pd.Series:
    delta = price.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / (loss + 1e-12)
    return 100 - 100 / (1 + rs)


def rolling_last_percentile(series: pd.Series, window: int = 252) -> pd.Series:
    def f(x: np.ndarray) -> float:
        if len(x) == 0 or np.isnan(x[-1]):
            return np.nan
        return float(pd.Series(x).rank(pct=True).iloc[-1])

    return series.rolling(window, min_periods=max(30, window // 4)).apply(f, raw=True)


def create_market_features(df_all: pd.DataFrame) -> pd.DataFrame:
    close = make_wide_prices(df_all, "close")
    market = pd.DataFrame(index=close.index).sort_index()

    if "SPY" in close.columns:
        spy = close["SPY"].dropna()
    elif "^GSPC" in close.columns:
        spy = close["^GSPC"].dropna()
    else:
        numeric_cols = close.columns[close.notna().sum() > 252]
        spy = close[numeric_cols].pct_change().mean(axis=1).add(1).cumprod()

    market["spy_close"] = spy.reindex(market.index).ffill()
    market["mkt_ret_1"] = market["spy_close"].pct_change(1)
    market["mkt_ret_5"] = market["spy_close"].pct_change(5)
    market["mkt_ret_21"] = market["spy_close"].pct_change(21)
    market["mkt_ret_63"] = market["spy_close"].pct_change(63)
    market["mkt_vol_21"] = market["mkt_ret_1"].rolling(21).std()
    market["mkt_vol_63"] = market["mkt_ret_1"].rolling(63).std()
    market["spy_ma_50"] = market["spy_close"].rolling(50).mean()
    market["spy_ma_200"] = market["spy_close"].rolling(200).mean()
    market["spy_trend_200"] = market["spy_close"] / market["spy_ma_200"] - 1
    market["spy_trend_50_200"] = market["spy_ma_50"] / market["spy_ma_200"] - 1
    market["spy_dd_252"] = market["spy_close"] / market["spy_close"].rolling(252).max() - 1
    market["mkt_vol_pctile_252"] = rolling_last_percentile(market["mkt_vol_21"], 252)

    if "^VIX" in close.columns:
        market["vix"] = close["^VIX"].reindex(market.index).ffill()
        market["vix_chg_5"] = market["vix"].pct_change(5)
        market["vix_pctile_252"] = rolling_last_percentile(market["vix"], 252)
    else:
        market["vix"] = np.nan
        market["vix_chg_5"] = np.nan
        market["vix_pctile_252"] = market["mkt_vol_pctile_252"]

    market = market.reset_index().rename(columns={"index": "dt"})
    return market


def assign_market_regime(row: pd.Series) -> str:
    trend_up = row.get("spy_trend_200", np.nan) > 0
    trend_strong = row.get("spy_trend_50_200", np.nan) > 0
    vix_pct = row.get("vix_pctile_252", np.nan)
    vol_pct = row.get("mkt_vol_pctile_252", np.nan)
    drawdown = row.get("spy_dd_252", np.nan)

    high_vol = (pd.notna(vix_pct) and vix_pct >= 0.75) or (pd.notna(vol_pct) and vol_pct >= 0.75)
    crash = (pd.notna(vix_pct) and vix_pct >= 0.90) or (pd.notna(drawdown) and drawdown <= -0.20)
    risk_off = (not trend_up and high_vol) or (pd.notna(drawdown) and drawdown <= -0.12)

    if crash:
        return "crash"
    if risk_off:
        return "risk_off"
    if trend_up and trend_strong and not high_vol:
        return "risk_on"
    if trend_up and high_vol:
        return "high_vol_bull"
    if not trend_up and not high_vol:
        return "sideways_or_recovery"
    return "neutral"


# %% [4] feature_engineering
def prepare_model_dataset(
    df_all: pd.DataFrame,
    meta: pd.DataFrame,
    horizon: int = 21,
    min_history_rows: int = 300,
    require_target: bool = True,
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tradable = meta[meta["asset_group"].isin(["sp500_equity", "defensive_asset"])].copy()
    df = df_all[df_all["ticker"].isin(tradable["ticker"])].copy()
    df = df.merge(
        tradable[["ticker", "gics_sector", "asset_group", "is_defensive"]],
        on="ticker",
        how="left",
    )
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce").dt.normalize()
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df.sort_values(["ticker", "dt"]).reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["dt", "ticker", "close"])
    df = df[df["close"] > 0].copy()

    g = df.groupby("ticker", group_keys=False)

    df["ret_1"] = g["close"].pct_change(1)
    df["ret_5"] = g["close"].pct_change(5)
    df["ret_10"] = g["close"].pct_change(10)
    df["ret_21"] = g["close"].pct_change(21)
    df["ret_63"] = g["close"].pct_change(63)
    df["ret_126"] = g["close"].pct_change(126)

    df["vol_10"] = g["ret_1"].rolling(10).std().reset_index(level=0, drop=True)
    df["vol_21"] = g["ret_1"].rolling(21).std().reset_index(level=0, drop=True)
    df["vol_63"] = g["ret_1"].rolling(63).std().reset_index(level=0, drop=True)

    df["ma_20"] = g["close"].rolling(20).mean().reset_index(level=0, drop=True)
    df["ma_50"] = g["close"].rolling(50).mean().reset_index(level=0, drop=True)
    df["ma_100"] = g["close"].rolling(100).mean().reset_index(level=0, drop=True)
    df["ma_200"] = g["close"].rolling(200).mean().reset_index(level=0, drop=True)

    eps = 1e-12
    df["price_to_ma20"] = df["close"] / (df["ma_20"] + eps) - 1
    df["price_to_ma50"] = df["close"] / (df["ma_50"] + eps) - 1
    df["price_to_ma100"] = df["close"] / (df["ma_100"] + eps) - 1
    df["price_to_ma200"] = df["close"] / (df["ma_200"] + eps) - 1

    df["high_low_range"] = (df["high"] - df["low"]) / (df["close"] + eps) if {"high", "low"}.issubset(df.columns) else np.nan
    df["rsi_14"] = g["close"].apply(calc_rsi).reset_index(level=0, drop=True)
    df["drawdown_252"] = df["close"] / g["close"].rolling(252).max().reset_index(level=0, drop=True) - 1

    if "volume" in df.columns:
        df["volume_ma_20"] = g["volume"].rolling(20).mean().reset_index(level=0, drop=True)
        df["volume_ratio_20"] = df["volume"] / (df["volume_ma_20"] + eps)
    else:
        df["volume_ratio_20"] = np.nan

    market = create_market_features(df_all)
    market["market_regime"] = market.apply(assign_market_regime, axis=1)
    df = df.merge(market, on="dt", how="left")

    df["rel_ret_21_vs_spy"] = df["ret_21"] - df["mkt_ret_21"]
    df["rel_ret_63_vs_spy"] = df["ret_63"] - df["mkt_ret_63"]
    df["is_defensive_num"] = df["is_defensive"].astype(float)

    df["target"] = g["close"].shift(-horizon) / df["close"] - 1

    base_features = [
        "ret_1", "ret_5", "ret_10", "ret_21", "ret_63", "ret_126",
        "vol_10", "vol_21", "vol_63",
        "price_to_ma20", "price_to_ma50", "price_to_ma100", "price_to_ma200",
        "high_low_range", "rsi_14", "drawdown_252", "volume_ratio_20",
        "mkt_ret_1", "mkt_ret_5", "mkt_ret_21", "mkt_ret_63",
        "mkt_vol_21", "mkt_vol_63", "spy_trend_200", "spy_trend_50_200", "spy_dd_252",
        "vix", "vix_chg_5", "vix_pctile_252", "mkt_vol_pctile_252",
        "rel_ret_21_vs_spy", "rel_ret_63_vs_spy", "is_defensive_num",
    ]

    df[base_features + ["target"]] = df[base_features + ["target"]].replace([np.inf, -np.inf], np.nan)

    for col in base_features:
        if col in df.columns:
            lo = df[col].quantile(0.005)
            hi = df[col].quantile(0.995)
            if pd.notna(lo) and pd.notna(hi) and lo < hi:
                df[col] = df[col].clip(lo, hi)

    sector_dummies = pd.get_dummies(df["gics_sector"].fillna("Unknown"), prefix="sector", dtype=float)
    df = pd.concat([df, sector_dummies], axis=1)
    feature_cols = base_features + sector_dummies.columns.tolist()

    history_counts = df.groupby("ticker")["dt"].transform("count")
    df = df[history_counts >= min_history_rows].copy()
    if require_target:
        df = df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)
    else:
        df = df.dropna(subset=feature_cols).reset_index(drop=True)

    prices_wide = make_wide_prices(df_all[df_all["ticker"].isin(tradable["ticker"])], "close")
    returns_wide = prices_wide.pct_change()

    return df, feature_cols, market, prices_wide, returns_wide


# %% [5] forecast_models
def make_model_registry(random_state: int = 42) -> Dict[str, object]:
    return {
        "ridge": make_pipeline(
            StandardScaler(),
            Ridge(alpha=10.0, random_state=random_state),
        ),
        "elastic_net": make_pipeline(
            StandardScaler(),
            ElasticNet(alpha=0.0005, l1_ratio=0.10, max_iter=5000, random_state=random_state),
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=7,
            min_samples_leaf=50,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.035,
            max_leaf_nodes=31,
            min_samples_leaf=60,
            l2_regularization=0.10,
            random_state=random_state,
        ),
    }


def candidate_models_for_horizon(horizon: int, registry: Dict[str, object]) -> Dict[str, object]:
    if horizon <= 5:
        names = ["ridge", "random_forest", "hist_gradient_boosting"]
    elif horizon <= 21:
        names = ["ridge", "elastic_net", "random_forest", "hist_gradient_boosting"]
    else:
        names = ["ridge", "elastic_net", "hist_gradient_boosting"]
    return {name: registry[name] for name in names if name in registry}


def mean_daily_ic(preds: pd.DataFrame, method: str = "spearman") -> float:
    ic = []
    for _, g in preds.groupby("dt"):
        if g["prediction"].nunique() < 2 or g["target"].nunique() < 2:
            continue
        val = g["prediction"].corr(g["target"], method=method)
        if pd.notna(val):
            ic.append(val)
    return float(np.mean(ic)) if ic else -np.inf


def select_and_fit_model(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    horizon: int,
    validation_window: int = 63,
    random_state: int = 42,
) -> Tuple[object, str, pd.DataFrame]:
    registry = make_model_registry(random_state=random_state)
    candidates = candidate_models_for_horizon(horizon, registry)
    dates = np.array(sorted(train_df["dt"].unique()))

    if len(dates) <= validation_window + 30:
        chosen_name = list(candidates.keys())[0]
        model = clone(candidates[chosen_name])
        valid = train_df.dropna(subset=feature_cols + ["target"])
        model.fit(valid[feature_cols], valid["target"])
        return model, chosen_name, pd.DataFrame()

    val_start = dates[-validation_window]
    tr = train_df[train_df["dt"] < val_start].dropna(subset=feature_cols + ["target"])
    val = train_df[train_df["dt"] >= val_start].dropna(subset=feature_cols + ["target"])

    rows = []
    best_name = None
    best_score = -np.inf

    for name, model_template in candidates.items():
        if len(tr) < 1000 or len(val) < 100:
            continue
        model = clone(model_template)
        model.fit(tr[feature_cols], tr["target"])
        pred = val[["dt", "ticker", "target"]].copy()
        pred["prediction"] = model.predict(val[feature_cols])

        rank_ic = mean_daily_ic(pred, method="spearman")
        pearson_ic = mean_daily_ic(pred, method="pearson")
        direction_acc = float((np.sign(pred["prediction"]) == np.sign(pred["target"])).mean())
        rmse = float(math.sqrt(mean_squared_error(pred["target"], pred["prediction"])))
        mae = float(mean_absolute_error(pred["target"], pred["prediction"]))
        score = rank_ic + 0.10 * (direction_acc - 0.50)

        rows.append(
            {
                "model_name": name,
                "rank_ic": rank_ic,
                "pearson_ic": pearson_ic,
                "direction_acc": direction_acc,
                "rmse": rmse,
                "mae": mae,
                "selection_score": score,
            }
        )

        if score > best_score:
            best_score = score
            best_name = name

    if best_name is None:
        best_name = list(candidates.keys())[0]

    final_model = clone(candidates[best_name])
    valid_all = train_df.dropna(subset=feature_cols + ["target"])
    final_model.fit(valid_all[feature_cols], valid_all["target"])
    return final_model, best_name, pd.DataFrame(rows).sort_values("selection_score", ascending=False)


def walk_forward_predictions(
    dataset: pd.DataFrame,
    feature_cols: List[str],
    config: BacktestConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    horizon = config.horizon
    rebalance_freq = config.rebalance_freq or horizon
    dates = np.array(sorted(dataset["dt"].unique()))

    preds = []
    model_log = []
    current_model = None
    current_model_name = None
    last_retrain_idx = -10**9

    first_idx = config.train_window + config.validation_window + horizon
    last_idx = len(dates) - horizon - 1

    for idx in range(first_idx, last_idx, rebalance_freq):
        current_date = dates[idx]
        train_end_idx = idx - horizon
        train_start_idx = max(0, train_end_idx - config.train_window - config.validation_window)

        need_retrain = current_model is None or (idx - last_retrain_idx) >= config.retrain_freq
        if need_retrain:
            train_dates = dates[train_start_idx : train_end_idx + 1]
            train_df = dataset[dataset["dt"].isin(train_dates)].copy()
            if len(train_df) < 1000:
                continue
            current_model, current_model_name, selection_table = select_and_fit_model(
                train_df=train_df,
                feature_cols=feature_cols,
                horizon=horizon,
                validation_window=config.validation_window,
                random_state=config.random_state,
            )
            last_retrain_idx = idx

            if not selection_table.empty:
                selection_table = selection_table.copy()
                selection_table["retrain_dt"] = current_date
                selection_table["chosen_model"] = current_model_name
                model_log.append(selection_table)
            else:
                model_log.append(
                    pd.DataFrame(
                        [{"retrain_dt": current_date, "model_name": current_model_name, "chosen_model": current_model_name}]
                    )
                )

        day = dataset[dataset["dt"] == current_date].dropna(subset=feature_cols + ["target"]).copy()
        if day.empty:
            continue
        day["prediction"] = current_model.predict(day[feature_cols])
        day["model_name"] = current_model_name
        preds.append(
            day[
                [
                    "dt", "ticker", "target", "prediction", "model_name",
                    "asset_group", "is_defensive", "gics_sector", "market_regime",
                    "vol_21", "vol_63",
                ]
            ].copy()
        )

        if len(preds) % 10 == 0:
            log(f"Predicted {len(preds)} rebalance dates. Current date: {pd.Timestamp(current_date).date()}")

    preds_df = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    model_log_df = pd.concat(model_log, ignore_index=True) if model_log else pd.DataFrame()
    return preds_df, model_log_df


# %% [6] portfolio_utils
def safe_cov_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    returns = returns.fillna(0.0)
    cols = returns.columns.tolist()
    if len(cols) == 0:
        return pd.DataFrame()
    if len(returns) < 30 or len(cols) == 1:
        cov = returns.cov().values if len(cols) > 1 else np.array([[float(returns.var().iloc[0])]])
    else:
        try:
            cov = LedoitWolf().fit(returns.values).covariance_
        except Exception:
            cov = returns.cov().values
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov = cov + np.eye(len(cols)) * 1e-8
    return pd.DataFrame(cov, index=cols, columns=cols)


def cap_and_normalize(weights: pd.Series, max_weight: float = 0.12) -> pd.Series:
    w = weights.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    w = w[w > 0]
    if w.empty:
        return w

    max_weight = normalize_weight_limit(max_weight)
    min_required = int(math.ceil(1.0 / max_weight))
    if len(w) < min_required:
        max_weight = max(max_weight, 1.0 / len(w))

    w = w / w.sum()

    for _ in range(100):
        over = w > max_weight + 1e-12
        if not over.any():
            break
        excess = float((w[over] - max_weight).sum())
        w[over] = max_weight
        under = ~over
        if not under.any() or w[under].sum() <= 0:
            break
        w[under] += excess * w[under] / w[under].sum()

    w = w.clip(upper=max_weight)
    if w.sum() > 0:
        w = w / w.sum()

    residual_over = w > max_weight + 1e-10
    if residual_over.any():
        w[residual_over] = max_weight
        if w.sum() > 0:
            w = w / w.sum()

    return w[w > 1e-8].sort_values(ascending=False)


def limit_number_of_positions(weights: pd.Series, max_positions: Optional[int], max_weight: float) -> pd.Series:
    if weights is None or weights.empty or max_positions is None:
        return weights

    max_weight = normalize_weight_limit(max_weight)
    min_required = min_positions_for_weight_limit(max_weight)
    effective_max_positions = max(int(max_positions), min_required)

    if len(weights) <= effective_max_positions:
        return cap_and_normalize(weights, max_weight=max_weight)

    trimmed = weights.sort_values(ascending=False).head(effective_max_positions)
    return cap_and_normalize(trimmed, max_weight=max_weight)


def finalize_strategy_weights(weights: pd.Series, config: BacktestConfig) -> pd.Series:
    if weights is None or weights.empty:
        return pd.Series(dtype=float)
    w = cap_and_normalize(weights, max_weight=config.max_weight)
    w = limit_number_of_positions(w, config.max_positions_per_strategy, config.max_weight)
    return w.sort_values(ascending=False)

def select_candidate_assets(
    day_data: pd.DataFrame,
    top_n_equity: int = 80,
    include_defensive: bool = True,
    positive_only: bool = False,
) -> pd.DataFrame:
    x = day_data.copy()
    equity = x[x["asset_group"] == "sp500_equity"].sort_values("prediction", ascending=False)
    if positive_only:
        equity = equity[equity["prediction"] > 0]
    equity = equity.head(top_n_equity)

    if include_defensive:
        defensive = x[x["asset_group"] == "defensive_asset"].copy()
        out = pd.concat([equity, defensive], ignore_index=True)
    else:
        out = equity.copy()

    out = out.drop_duplicates("ticker")
    return out


def get_cov_for_candidates(
    returns_wide: pd.DataFrame,
    current_date: pd.Timestamp,
    tickers: List[str],
    lookback: int = 252,
    horizon: int = 21,
) -> pd.DataFrame:
    hist = returns_wide.loc[returns_wide.index < current_date, tickers].tail(lookback)
    cov_daily = safe_cov_matrix(hist)
    return cov_daily * horizon


def optimize_long_only(
    mu: pd.Series,
    cov: pd.DataFrame,
    objective: str = "mean_variance",
    max_weight: float = 0.03,
    risk_aversion: float = 5.0,
) -> pd.Series:
    mu = mu.dropna().astype(float)
    common = [t for t in mu.index if t in cov.index]
    mu = mu.loc[common]
    cov = cov.loc[common, common]
    n = len(common)
    if n == 0:
        return pd.Series(dtype=float)
    if n == 1:
        return pd.Series([1.0], index=common)

    mu_values = mu.values.copy()
    lo, hi = np.nanquantile(mu_values, [0.02, 0.98])
    if lo < hi:
        mu_values = np.clip(mu_values, lo, hi)

    cov_values = cov.values
    max_weight = normalize_weight_limit(max_weight)
    if n < min_positions_for_weight_limit(max_weight):
        max_weight = max(max_weight, 1.0 / n)
    bounds = [(0.0, max_weight) for _ in range(n)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    x0 = np.ones(n) / n

    def port_var(w: np.ndarray) -> float:
        return float(w.T @ cov_values @ w)

    def neg_sharpe(w: np.ndarray) -> float:
        vol = math.sqrt(max(port_var(w), 1e-12))
        return -float(w @ mu_values / vol)

    def neg_utility(w: np.ndarray) -> float:
        return -float(w @ mu_values - 0.5 * risk_aversion * port_var(w))

    def min_var(w: np.ndarray) -> float:
        return port_var(w)

    if objective == "max_sharpe":
        fun = neg_sharpe
    elif objective == "min_variance":
        fun = min_var
    elif objective == "mean_variance":
        fun = neg_utility
    else:
        raise ValueError("objective must be max_sharpe, min_variance or mean_variance")

    try:
        res = minimize(fun, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 300})
        w = res.x if res.success else x0
    except Exception:
        w = x0

    return cap_and_normalize(pd.Series(w, index=common), max_weight=max_weight)


def risk_parity_weights(cov: pd.DataFrame, max_weight: float = 0.03) -> pd.Series:
    cov = cov.dropna(axis=0, how="all").dropna(axis=1, how="all")
    common = cov.index.intersection(cov.columns).tolist()
    cov = cov.loc[common, common].fillna(0.0)
    n = len(common)
    if n == 0:
        return pd.Series(dtype=float)
    if n == 1:
        return pd.Series([1.0], index=common)

    cov_values = cov.values + np.eye(n) * 1e-8
    max_weight = normalize_weight_limit(max_weight)
    if n < min_positions_for_weight_limit(max_weight):
        max_weight = max(max_weight, 1.0 / n)
    bounds = [(0.0, max_weight) for _ in range(n)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    x0 = np.ones(n) / n

    def objective(w: np.ndarray) -> float:
        port_var = max(float(w.T @ cov_values @ w), 1e-12)
        mrc = cov_values @ w
        rc = w * mrc / math.sqrt(port_var)
        return float(((rc - rc.mean()) ** 2).sum())

    try:
        res = minimize(objective, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 500})
        w = res.x if res.success else x0
    except Exception:
        w = x0

    return cap_and_normalize(pd.Series(w, index=common), max_weight=max_weight)


def inverse_vol_weights(cov: pd.DataFrame, max_weight: float = 0.03) -> pd.Series:
    vol = pd.Series(np.sqrt(np.diag(cov.values)), index=cov.index).replace(0, np.nan)
    inv = 1.0 / vol
    return cap_and_normalize(inv, max_weight=max_weight)


# %% [7] rebalancing_strategies
def equal_weight_all_strategy(day_data: pd.DataFrame, returns_wide: pd.DataFrame, current_date: pd.Timestamp, config: BacktestConfig) -> pd.Series:
    tickers = day_data["ticker"].tolist()
    return cap_and_normalize(pd.Series(1.0, index=tickers), max_weight=config.max_weight)


def topn_equal_strategy(day_data: pd.DataFrame, returns_wide: pd.DataFrame, current_date: pd.Timestamp, config: BacktestConfig) -> pd.Series:
    candidates = select_candidate_assets(day_data, config.top_n_equity, include_defensive=True, positive_only=False)
    candidates = candidates.sort_values("prediction", ascending=False).head(config.top_n_equity + len(DEFENSIVE_ASSETS))
    return cap_and_normalize(pd.Series(1.0, index=candidates["ticker"]), max_weight=config.max_weight)


def inverse_vol_strategy(day_data: pd.DataFrame, returns_wide: pd.DataFrame, current_date: pd.Timestamp, config: BacktestConfig) -> pd.Series:
    candidates = select_candidate_assets(day_data, config.top_n_equity, include_defensive=True, positive_only=True)
    if candidates.empty:
        candidates = select_candidate_assets(day_data, config.top_n_equity, include_defensive=True, positive_only=False)
    tickers = candidates["ticker"].tolist()
    cov = get_cov_for_candidates(returns_wide, current_date, tickers, config.cov_lookback, config.horizon)
    return inverse_vol_weights(cov, max_weight=config.max_weight)


def min_variance_strategy(day_data: pd.DataFrame, returns_wide: pd.DataFrame, current_date: pd.Timestamp, config: BacktestConfig) -> pd.Series:
    candidates = select_candidate_assets(day_data, config.top_n_equity, include_defensive=True, positive_only=False)
    tickers = candidates["ticker"].tolist()
    cov = get_cov_for_candidates(returns_wide, current_date, tickers, config.cov_lookback, config.horizon)
    mu = pd.Series(0.0, index=cov.index)
    return optimize_long_only(mu, cov, objective="min_variance", max_weight=config.max_weight)


def risk_parity_strategy(day_data: pd.DataFrame, returns_wide: pd.DataFrame, current_date: pd.Timestamp, config: BacktestConfig) -> pd.Series:
    candidates = select_candidate_assets(day_data, config.top_n_equity, include_defensive=True, positive_only=False)
    tickers = candidates["ticker"].tolist()
    cov = get_cov_for_candidates(returns_wide, current_date, tickers, config.cov_lookback, config.horizon)
    return risk_parity_weights(cov, max_weight=config.max_weight)


def mean_variance_strategy(day_data: pd.DataFrame, returns_wide: pd.DataFrame, current_date: pd.Timestamp, config: BacktestConfig) -> pd.Series:
    candidates = select_candidate_assets(day_data, config.top_n_equity, include_defensive=True, positive_only=True)
    if len(candidates) < max(20, int(1 / config.max_weight)):
        candidates = select_candidate_assets(day_data, config.top_n_equity, include_defensive=True, positive_only=False)
    tickers = candidates["ticker"].tolist()
    cov = get_cov_for_candidates(returns_wide, current_date, tickers, config.cov_lookback, config.horizon)
    mu = candidates.set_index("ticker")["prediction"].reindex(cov.index).fillna(0.0)
    return optimize_long_only(mu, cov, objective="mean_variance", max_weight=config.max_weight, risk_aversion=5.0)


def max_sharpe_strategy(day_data: pd.DataFrame, returns_wide: pd.DataFrame, current_date: pd.Timestamp, config: BacktestConfig) -> pd.Series:
    candidates = select_candidate_assets(day_data, config.top_n_equity, include_defensive=True, positive_only=True)
    if candidates.empty:
        return min_variance_strategy(day_data, returns_wide, current_date, config)
    tickers = candidates["ticker"].tolist()
    cov = get_cov_for_candidates(returns_wide, current_date, tickers, config.cov_lookback, config.horizon)
    mu = candidates.set_index("ticker")["prediction"].reindex(cov.index).fillna(0.0)
    return optimize_long_only(mu, cov, objective="max_sharpe", max_weight=config.max_weight)


def defensive_min_variance_strategy(day_data: pd.DataFrame, returns_wide: pd.DataFrame, current_date: pd.Timestamp, config: BacktestConfig) -> pd.Series:
    defensive = day_data[day_data["asset_group"] == "defensive_asset"].copy()
    if len(defensive) < 3:
        return min_variance_strategy(day_data, returns_wide, current_date, config)
    tickers = defensive["ticker"].tolist()
    cov = get_cov_for_candidates(returns_wide, current_date, tickers, config.cov_lookback, config.horizon)
    mu = pd.Series(0.0, index=cov.index)
    return optimize_long_only(mu, cov, objective="min_variance", max_weight=max(config.max_weight, 0.35))


def adaptive_regime_switch_strategy(
    day_data: pd.DataFrame,
    returns_wide: pd.DataFrame,
    current_date: pd.Timestamp,
    config: BacktestConfig,
) -> pd.Series:
    regime = str(day_data["market_regime"].dropna().iloc[0]) if day_data["market_regime"].notna().any() else "neutral"

    if regime in ["crash", "risk_off"]:
        return defensive_min_variance_strategy(day_data, returns_wide, current_date, config)

    if regime == "high_vol_bull":
        tmp_config = BacktestConfig(**{**config.__dict__, "top_n_equity": max(30, config.top_n_equity // 2)})
        return risk_parity_strategy(day_data, returns_wide, current_date, tmp_config)

    if regime == "sideways_or_recovery":
        tmp_config = BacktestConfig(**{**config.__dict__, "top_n_equity": max(40, config.top_n_equity // 2)})
        return inverse_vol_strategy(day_data, returns_wide, current_date, tmp_config)

    if regime == "risk_on":
        return mean_variance_strategy(day_data, returns_wide, current_date, config)

    return max_sharpe_strategy(day_data, returns_wide, current_date, config)


STRATEGY_REGISTRY: Dict[str, Callable[[pd.DataFrame, pd.DataFrame, pd.Timestamp, BacktestConfig], pd.Series]] = {
    "equal_weight_all": equal_weight_all_strategy,
    "topn_equal": topn_equal_strategy,
    "inverse_vol": inverse_vol_strategy,
    "min_variance": min_variance_strategy,
    "risk_parity": risk_parity_strategy,
    "mean_variance": mean_variance_strategy,
    "max_sharpe": max_sharpe_strategy,
    "defensive_min_variance": defensive_min_variance_strategy,
    "adaptive_regime_switch": adaptive_regime_switch_strategy,
}


# %% [8] backtest_and_metrics
def backtest_single_strategy(
    preds_df: pd.DataFrame,
    returns_wide: pd.DataFrame,
    strategy_name: str,
    config: BacktestConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    strategy_func = STRATEGY_REGISTRY[strategy_name]
    results = []
    weights_log = []
    current_weights = pd.Series(dtype=float)

    for current_date, day in preds_df.groupby("dt"):
        current_date = pd.Timestamp(current_date)
        day = day.dropna(subset=["prediction", "target"]).copy()
        if day.empty:
            continue

        new_weights = strategy_func(day, returns_wide, current_date, config)
        new_weights = finalize_strategy_weights(new_weights, config)
        if new_weights.empty:
            continue

        realized = day.set_index("ticker")["target"].reindex(new_weights.index).fillna(0.0)
        gross_return = float((new_weights * realized).sum())

        all_tickers = current_weights.index.union(new_weights.index)
        turnover = float((new_weights.reindex(all_tickers).fillna(0.0) - current_weights.reindex(all_tickers).fillna(0.0)).abs().sum())
        cost = turnover * config.transaction_cost
        net_return = gross_return - cost

        regime = str(day["market_regime"].dropna().iloc[0]) if day["market_regime"].notna().any() else "neutral"
        model_name = str(day["model_name"].dropna().iloc[0]) if day["model_name"].notna().any() else "unknown"

        results.append(
            {
                "dt": current_date,
                "strategy": strategy_name,
                "market_regime": regime,
                "model_name": model_name,
                "gross_return": gross_return,
                "turnover": turnover,
                "cost": cost,
                "net_return": net_return,
                "n_assets": int(len(new_weights)),
            }
        )

        top_weights = new_weights.head(30).reset_index()
        top_weights.columns = ["ticker", "weight"]
        top_weights["dt"] = current_date
        top_weights["strategy"] = strategy_name
        top_weights["market_regime"] = regime
        weights_log.append(top_weights)

        current_weights = new_weights.copy()

    results_df = pd.DataFrame(results).sort_values("dt").reset_index(drop=True)
    if not results_df.empty:
        results_df["equity_curve"] = (1 + results_df["net_return"]).cumprod()
        results_df["cummax"] = results_df["equity_curve"].cummax()
        results_df["drawdown"] = results_df["equity_curve"] / results_df["cummax"] - 1

    weights_df = pd.concat(weights_log, ignore_index=True) if weights_log else pd.DataFrame()
    return results_df, weights_df


def compare_strategies(
    preds_df: pd.DataFrame,
    returns_wide: pd.DataFrame,
    config: BacktestConfig,
    strategies: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strategies = strategies or list(STRATEGY_REGISTRY.keys())
    all_results = []
    all_weights = []

    for strategy_name in strategies:
        log(f"Backtesting strategy: {strategy_name}")
        res, wlog = backtest_single_strategy(preds_df, returns_wide, strategy_name, config)
        if not res.empty:
            all_results.append(res)
        if not wlog.empty:
            all_weights.append(wlog)

    results_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    weights_df = pd.concat(all_weights, ignore_index=True) if all_weights else pd.DataFrame()
    metrics_df = summarize_strategy_metrics(results_df, periods_per_year=252 / (config.rebalance_freq or config.horizon))
    return results_df, weights_df, metrics_df


def _safe_metric_value(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        value = float(value)
        if not np.isfinite(value):
            return float(default)
        return value
    except Exception:
        return float(default)


def summarize_strategy_metrics(results_df: pd.DataFrame, periods_per_year: float = 12.0) -> pd.DataFrame:
    rows = []
    if results_df.empty:
        return pd.DataFrame(
            columns=[
                "strategy", "periods", "metric_status", "total_return", "annual_return",
                "annual_vol", "sharpe", "sortino", "max_drawdown", "calmar",
                "avg_turnover", "avg_n_assets", "hit_rate", "final_equity",
            ]
        )

    for strategy, g in results_df.groupby("strategy"):
        g = g.sort_values("dt").copy()
        r = pd.to_numeric(g["net_return"], errors="coerce").dropna()
        if r.empty:
            continue

        equity = (1.0 + r).cumprod()
        total_return = float(equity.iloc[-1] - 1.0)
        n_periods = int(len(r))
        final_equity = float(equity.iloc[-1])

        annual_return = final_equity ** (float(periods_per_year) / max(n_periods, 1)) - 1.0

        if n_periods >= 2:
            annual_vol = float(r.std(ddof=1) * math.sqrt(periods_per_year))
        else:
            annual_vol = 0.0

        if annual_vol > 0:
            sharpe = annual_return / annual_vol
        else:
            sharpe = 0.0

        downside_returns = r[r < 0]
        if len(downside_returns) >= 2:
            downside = float(downside_returns.std(ddof=1) * math.sqrt(periods_per_year))
        elif len(downside_returns) == 1:
            downside = abs(float(downside_returns.iloc[0])) * math.sqrt(periods_per_year)
        else:
            downside = 0.0

        if downside > 0:
            sortino = annual_return / downside
        else:
            sortino = 0.0

        if "drawdown" in g.columns and g["drawdown"].notna().any():
            max_dd = float(pd.to_numeric(g["drawdown"], errors="coerce").min())
        else:
            running_equity = (1.0 + r).cumprod()
            max_dd = float((running_equity / running_equity.cummax() - 1.0).min())

        if max_dd < 0:
            calmar = annual_return / abs(max_dd)
        else:
            calmar = 0.0

        hit_rate = float((r > 0).mean())

        if n_periods < 3:
            metric_status = "too_few_periods"
        elif n_periods < 10:
            metric_status = "low_confidence"
        else:
            metric_status = "ok"

        rows.append(
            {
                "strategy": strategy,
                "periods": n_periods,
                "metric_status": metric_status,
                "total_return": _safe_metric_value(total_return),
                "annual_return": _safe_metric_value(annual_return),
                "annual_vol": _safe_metric_value(annual_vol),
                "sharpe": _safe_metric_value(sharpe),
                "sortino": _safe_metric_value(sortino),
                "max_drawdown": _safe_metric_value(max_dd),
                "calmar": _safe_metric_value(calmar),
                "avg_turnover": _safe_metric_value(g["turnover"].mean()),
                "avg_n_assets": _safe_metric_value(g["n_assets"].mean()),
                "hit_rate": _safe_metric_value(hit_rate),
                "final_equity": _safe_metric_value(final_equity, default=1.0),
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    return out.sort_values(["sharpe", "final_equity"], ascending=False).reset_index(drop=True)


def plot_results(results_df: pd.DataFrame, metrics_df: Optional[pd.DataFrame] = None) -> None:
    import matplotlib.pyplot as plt

    if results_df.empty:
        print("No results to plot")
        return

    pivot_eq = results_df.pivot(index="dt", columns="strategy", values="equity_curve").sort_index()
    plt.figure(figsize=(14, 7))
    for col in pivot_eq.columns:
        plt.plot(pivot_eq.index, pivot_eq[col], label=col)
    plt.title("Equity curve by strategy")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.legend()
    plt.show()

    pivot_dd = results_df.pivot(index="dt", columns="strategy", values="drawdown").sort_index()
    plt.figure(figsize=(14, 6))
    for col in pivot_dd.columns:
        plt.plot(pivot_dd.index, pivot_dd[col], label=col)
    plt.title("Drawdown by strategy")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.legend()
    plt.show()

    if metrics_df is not None and not metrics_df.empty:
        print(metrics_df)


def run_full_experiment(config: Optional[BacktestConfig] = None) -> Dict[str, pd.DataFrame]:
    config = config or BacktestConfig()

    log("Building/downloading data")
    df_all, meta, failed = build_price_dataset(config)
    log(f"df_all shape: {df_all.shape}; universe rows: {meta.shape}; failed downloads: {len(failed)}")

    log("Preparing features")
    dataset, feature_cols, market, prices_wide, returns_wide = prepare_model_dataset(
        df_all=df_all,
        meta=meta,
        horizon=config.horizon,
        min_history_rows=config.min_history_rows,
    )
    log(f"model dataset shape: {dataset.shape}; features: {len(feature_cols)}")

    log("Walk-forward model selection and predictions")
    preds_df, model_log_df = walk_forward_predictions(dataset, feature_cols, config)
    log(f"predictions shape: {preds_df.shape}")

    strategies = list(getattr(config, "strategy_names", tuple(STRATEGY_REGISTRY.keys())))

    log("Comparing rebalancing strategies")
    results_df, weights_df, metrics_df = compare_strategies(preds_df, returns_wide, config, strategies=strategies)

    return {
        "df_all": df_all,
        "meta": meta,
        "failed": failed,
        "dataset": dataset,
        "feature_cols": pd.DataFrame({"feature": feature_cols}),
        "market": market,
        "preds": preds_df,
        "model_log": model_log_df,
        "results": results_df,
        "weights": weights_df,
        "metrics": metrics_df,
    }


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(obj))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


def make_run_id(config: BacktestConfig) -> str:
    prefix = config.run_name or "rebalance"
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def ensure_run_dir(config: BacktestConfig, run_id: Optional[str] = None) -> str:
    run_id = run_id or make_run_id(config)
    path = os.path.join(config.output_dir, run_id)
    os.makedirs(path, exist_ok=True)
    return path


def _write_csv(df: Optional[pd.DataFrame], path: str) -> Optional[str]:
    if df is None:
        return None
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if not isinstance(df, pd.DataFrame):
        return None
    df.to_csv(path, index=False)
    return path


def _save_pickle(obj: Any, path: str) -> str:
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def save_pipeline_artifacts(outputs: Dict[str, Any], config: BacktestConfig, run_dir: str) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    os.makedirs(run_dir, exist_ok=True)

    table_names = [
        "metrics",
        "results",
        "weights",
        "live_predictions",
        "live_weights",
        "final_live_weights",
        "model_log",
        "latest_model_selection",
        "data_quality",
        "failed",
        "meta",
        "market",
        "feature_cols",
    ]
    for name in table_names:
        obj = outputs.get(name)
        if isinstance(obj, pd.DataFrame) and not obj.empty:
            paths[name] = _write_csv(obj, os.path.join(run_dir, f"{name}.csv"))
        elif isinstance(obj, pd.DataFrame):
            paths[name] = _write_csv(obj, os.path.join(run_dir, f"{name}.csv"))

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config.__dict__, f, ensure_ascii=False, indent=2, default=_json_default)
    paths["config"] = config_path

    if outputs.get("latest_model") is not None:
        paths["latest_model"] = _save_pickle(outputs["latest_model"], os.path.join(run_dir, "latest_model.pkl"))

    summary = {
        "run_dir": run_dir,
        "created_at_utc": str(pd.Timestamp.utcnow()),
        "latest_prediction_date": str(outputs.get("live_prediction_date")),
        "latest_model_name": outputs.get("latest_model_name"),
        "final_strategy": config.final_strategy,
        "artifact_paths": paths,
    }
    summary_path = os.path.join(run_dir, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)
    paths["run_summary"] = summary_path
    return paths


# %% [9] data_validation_and_artifacts
def data_quality_report(df_all: pd.DataFrame, meta: pd.DataFrame, failed: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    def add(check: str, status: str, value: Any, threshold: Any = None, message: str = "") -> None:
        rows.append(
            {
                "check": check,
                "status": status,
                "value": value,
                "threshold": threshold,
                "message": message,
            }
        )

    if df_all is None or df_all.empty:
        add("market_data_not_empty", "FAIL", 0, ">0", "No market data")
        return pd.DataFrame(rows)

    df = df_all.copy()
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce").dt.normalize()
    tradable_meta = meta[meta["asset_group"].isin(["sp500_equity", "defensive_asset"])] if meta is not None and not meta.empty else pd.DataFrame()
    tradable = sorted(set(tradable_meta["ticker"])) if not tradable_meta.empty else sorted(set(df["ticker"]))
    available_tickers = sorted(set(df["ticker"]))
    tradable_available = sorted(set(tradable).intersection(available_tickers))

    add("rows", "PASS" if len(df) > 0 else "FAIL", len(df), ">0")
    add(
        "tradable_assets_available",
        "PASS" if len(tradable_available) >= config.min_tradable_assets else "WARN",
        len(tradable_available),
        config.min_tradable_assets,
        "Number of tradable assets with downloaded prices",
    )
    add("failed_downloads", "PASS" if len(failed) == 0 else "WARN", len(failed), 0)

    min_dt = df["dt"].min()
    max_dt = df["dt"].max()
    add("min_date", "PASS", str(min_dt.date()) if pd.notna(min_dt) else None, config.start)
    add("max_date", "PASS", str(max_dt.date()) if pd.notna(max_dt) else None, config.end)

    if pd.notna(max_dt):
        today = pd.Timestamp.today().normalize()
        lag_days = int((today - max_dt).days)
        add(
            "data_lag_days",
            "PASS" if lag_days <= config.max_data_lag_days else "WARN",
            lag_days,
            config.max_data_lag_days,
            "For historical offline runs this WARN can be ignored",
        )

    dupes = int(df.duplicated(["dt", "ticker"]).sum())
    add("duplicate_dt_ticker", "PASS" if dupes == 0 else "FAIL", dupes, 0)

    price_bad = int((pd.to_numeric(df.get("close"), errors="coerce") <= 0).sum()) if "close" in df.columns else len(df)
    add("non_positive_close", "PASS" if price_bad == 0 else "FAIL", price_bad, 0)

    coverage = df.groupby("ticker")["dt"].nunique()
    if not coverage.empty:
        add("median_history_rows", "PASS", int(coverage.median()), config.min_history_rows)
        low_history = int((coverage < config.min_history_rows).sum())
        add("assets_below_min_history", "PASS" if low_history == 0 else "WARN", low_history, 0)

    required_features = [t for t in config.feature_tickers if t in ["SPY", "^VIX"]]
    missing_features = sorted(set(required_features) - set(available_tickers))
    add("market_feature_assets", "PASS" if not missing_features else "WARN", ",".join(missing_features), "none missing")

    return pd.DataFrame(rows)


def assert_data_quality_ok(quality: pd.DataFrame, config: BacktestConfig) -> None:
    if quality.empty:
        raise RuntimeError("Data quality report is empty")
    bad = quality[quality["status"].eq("FAIL")]
    if not bad.empty:
        raise RuntimeError("Data quality failed: " + bad[["check", "message"]].to_dict("records").__repr__())
    if config.fail_on_data_quality:
        warns = quality[quality["status"].eq("WARN")]
        if not warns.empty:
            raise RuntimeError("Data quality warnings: " + warns[["check", "message"]].to_dict("records").__repr__())


def fit_latest_forecast_model(
    dataset_with_target: pd.DataFrame,
    feature_cols: List[str],
    config: BacktestConfig,
) -> Tuple[Any, str, pd.DataFrame, pd.Timestamp]:
    train_df = dataset_with_target.dropna(subset=feature_cols + ["target"]).copy()
    if train_df.empty:
        raise RuntimeError("No rows with known target for model training")

    dates = np.array(sorted(train_df["dt"].unique()))
    if len(dates) < max(30, config.validation_window + 10):
        raise RuntimeError(
            f"Not enough dates for model training: {len(dates)}. "
            f"Reduce validation_window/train_window/horizon or extend history."
        )

    if len(dates) > config.train_window + config.validation_window:
        train_dates = dates[-(config.train_window + config.validation_window) :]
        train_df = train_df[train_df["dt"].isin(train_dates)].copy()

    model, model_name, selection = select_and_fit_model(
        train_df=train_df,
        feature_cols=feature_cols,
        horizon=config.horizon,
        validation_window=config.validation_window,
        random_state=config.random_state,
    )
    latest_train_dt = pd.Timestamp(train_df["dt"].max())
    if selection is not None and not selection.empty:
        selection = selection.copy()
        selection["latest_train_dt"] = latest_train_dt
        selection["chosen_model"] = model_name
    else:
        selection = pd.DataFrame(
            [{"model_name": model_name, "chosen_model": model_name, "latest_train_dt": latest_train_dt}]
        )
    return model, model_name, selection, latest_train_dt


def get_live_day_frame(
    dataset_live: pd.DataFrame,
    feature_cols: List[str],
    config: BacktestConfig,
) -> Tuple[pd.DataFrame, pd.Timestamp]:
    if dataset_live.empty:
        raise RuntimeError("Live feature dataset is empty")

    live = dataset_live.dropna(subset=feature_cols).copy()
    if live.empty:
        raise RuntimeError("No live rows with complete features")

    live["dt"] = pd.to_datetime(live["dt"], errors="coerce").dt.normalize()
    if config.live_as_of_date:
        as_of = pd.Timestamp(config.live_as_of_date).normalize()
        live = live[live["dt"] <= as_of].copy()
        if live.empty:
            raise RuntimeError(f"No live feature rows on or before {as_of.date()}")

    current_date = pd.Timestamp(live["dt"].max()).normalize()
    day = live[live["dt"].eq(current_date)].copy()
    if len(day) < config.min_tradable_assets:
        log(f"WARN: live day has only {len(day)} assets with complete features")
    return day, current_date


def make_live_predictions(
    model: Any,
    model_name: str,
    dataset_live: pd.DataFrame,
    feature_cols: List[str],
    config: BacktestConfig,
) -> Tuple[pd.DataFrame, pd.Timestamp]:
    day, current_date = get_live_day_frame(dataset_live, feature_cols, config)
    day = day.copy()
    day["prediction"] = model.predict(day[feature_cols])
    day["model_name"] = model_name
    if "target" not in day.columns:
        day["target"] = np.nan

    keep_cols = [
        "dt",
        "ticker",
        "prediction",
        "target",
        "model_name",
        "asset_group",
        "is_defensive",
        "gics_sector",
        "market_regime",
        "vol_21",
        "vol_63",
    ]
    keep_cols = [c for c in keep_cols if c in day.columns]
    pred = day[keep_cols].sort_values("prediction", ascending=False).reset_index(drop=True)
    return pred, current_date


def compute_live_weights(
    live_predictions: pd.DataFrame,
    returns_wide: pd.DataFrame,
    config: BacktestConfig,
    strategies: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if live_predictions.empty:
        return pd.DataFrame()

    strategies = list(strategies or config.strategy_names)
    current_date = pd.Timestamp(live_predictions["dt"].max()).normalize()
    rows = []

    for strategy_name in strategies:
        if strategy_name not in STRATEGY_REGISTRY:
            log(f"WARN: unknown strategy skipped: {strategy_name}")
            continue
        strategy_func = STRATEGY_REGISTRY[strategy_name]
        try:
            w = strategy_func(live_predictions.copy(), returns_wide, current_date, config)
        except Exception as exc:
            log(f"WARN: live strategy failed: {strategy_name}: {exc}")
            continue
        w = finalize_strategy_weights(w, config)
        if w is None or w.empty:
            continue
        part = w.reset_index()
        part.columns = ["ticker", "weight"]
        part["dt"] = current_date
        part["strategy"] = strategy_name
        part["max_weight_limit"] = normalize_weight_limit(config.max_weight)
        part["strategy_position_count"] = int(len(w))
        part["strategy_max_weight"] = float(w.max())
        part["cap_is_binding"] = bool(w.max() >= normalize_weight_limit(config.max_weight) - 1e-8)
        rows.append(part)

    if not rows:
        return pd.DataFrame(columns=["dt", "strategy", "ticker", "weight"])
    out = pd.concat(rows, ignore_index=True)
    ordered_cols = [
        "dt", "strategy", "ticker", "weight", "max_weight_limit",
        "strategy_position_count", "strategy_max_weight", "cap_is_binding",
    ]
    out = out[ordered_cols].sort_values(["strategy", "weight"], ascending=[True, False])
    return out.reset_index(drop=True)


def select_final_live_weights(live_weights: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    if live_weights.empty:
        return live_weights.copy()
    strategy = config.final_strategy
    if strategy not in set(live_weights["strategy"]):
        log(f"WARN: final_strategy={strategy} not found in live weights; using first available strategy")
        strategy = str(live_weights["strategy"].iloc[0])
    out = live_weights[live_weights["strategy"].eq(strategy)].copy()
    out = out.sort_values("weight", ascending=False).reset_index(drop=True)
    return out


# %% [11] production_pipeline
def run_production_pipeline(config: Optional[BacktestConfig] = None) -> Dict[str, Any]:
    config = config or BacktestConfig()
    run_dir = ensure_run_dir(config) if config.save_artifacts else ""

    log("PROD STEP 1/7: build/download market data")
    df_all, meta, failed = build_price_dataset(config)

    log("PROD STEP 2/7: validate data quality")
    quality = data_quality_report(df_all, meta, failed, config)
    assert_data_quality_ok(quality, config)

    log("PROD STEP 3/7: build features for training/backtest")
    dataset_train, feature_cols, market, prices_wide, returns_wide = prepare_model_dataset(
        df_all=df_all,
        meta=meta,
        horizon=config.horizon,
        min_history_rows=config.min_history_rows,
        require_target=True,
    )

    results_df = pd.DataFrame()
    weights_df = pd.DataFrame()
    metrics_df = pd.DataFrame()
    model_log_df = pd.DataFrame()
    preds_df = pd.DataFrame()

    if config.run_backtest:
        log("PROD STEP 4/7: run walk-forward backtest for strategy comparison")
        preds_df, model_log_df = walk_forward_predictions(dataset_train, feature_cols, config)
        if not preds_df.empty:
            results_df, weights_df, metrics_df = compare_strategies(
                preds_df,
                returns_wide,
                config,
                strategies=list(config.strategy_names),
            )
        else:
            log("WARN: no walk-forward predictions; backtest comparison is empty")
    else:
        log("PROD STEP 4/7: backtest skipped by config.run_backtest=False")

    log("PROD STEP 5/7: fit latest production forecast model")
    latest_model, latest_model_name, latest_selection, latest_train_dt = fit_latest_forecast_model(
        dataset_train,
        feature_cols,
        config,
    )

    log("PROD STEP 6/7: build live features and calculate live weights")
    dataset_live, _, _, _, returns_wide_live = prepare_model_dataset(
        df_all=df_all,
        meta=meta,
        horizon=config.horizon,
        min_history_rows=config.min_history_rows,
        require_target=False,
    )
    live_predictions, live_prediction_date = make_live_predictions(
        latest_model,
        latest_model_name,
        dataset_live,
        feature_cols,
        config,
    )
    live_weights = compute_live_weights(live_predictions, returns_wide_live, config, strategies=list(config.strategy_names))
    final_live_weights = select_final_live_weights(live_weights, config)

    outputs: Dict[str, Any] = {
        "run_dir": run_dir,
        "df_all": df_all,
        "meta": meta,
        "failed": failed,
        "data_quality": quality,
        "dataset": dataset_train,
        "feature_cols": pd.DataFrame({"feature": feature_cols}),
        "market": market,
        "preds": preds_df,
        "model_log": model_log_df,
        "results": results_df,
        "weights": weights_df,
        "metrics": metrics_df,
        "latest_model": latest_model,
        "latest_model_name": latest_model_name,
        "latest_model_selection": latest_selection,
        "latest_train_dt": latest_train_dt,
        "live_prediction_date": live_prediction_date,
        "live_predictions": live_predictions,
        "live_weights": live_weights,
        "final_live_weights": final_live_weights,
    }

    log("PROD STEP 7/7: save artifacts")
    if config.save_artifacts:
        outputs["artifact_paths"] = save_pipeline_artifacts(outputs, config, run_dir)
        log(f"Artifacts saved to: {run_dir}")
    else:
        outputs["artifact_paths"] = {}

    return outputs


# %% [12] run_configs
def make_prod_config_2024_2026() -> BacktestConfig:
    return BacktestConfig(
        start="2024-01-01",
        end=None,
        horizon=10,
        train_window=126,
        validation_window=21,
        rebalance_freq=10,
        retrain_freq=42,
        cov_lookback=63,
        transaction_cost=0.001,
        max_weight=12.0,
        top_n_equity=50,
        max_positions_per_strategy=None,
        min_history_rows=120,
        cache_path="sp500_defensive_yfinance_cache.pkl",
        refresh_cache=True,
        random_state=42,
        sp500_local_path="sp500_tickers.csv",
        sp500_universe_cache_path="sp500_universe_cache.csv",
        sp500_min_count=450,
        sp500_source_priority=(
            "local",
            "csv_github",
            "ssga_spy_holdings",
            "wikipedia",
            "universe_cache",
        ),
        download_chunk_size=10,
        download_sleep_sec=2.0,
        download_max_retries=4,
        download_threads=False,
        download_timeout=30,
        download_retry_failed_individual=True,
        download_individual_sleep_sec=0.5,
        download_silent=True,
        production_mode=True,
        output_dir="rebalancing_prod_runs",
        run_name="sp500_prod_compare",
        run_backtest=True,
        live_as_of_date=None,
        min_tradable_assets=100,
        max_data_lag_days=10,
        fail_on_data_quality=False,
        final_strategy="adaptive_regime_switch",
        save_artifacts=True,
    )



def make_prod_config_2024_2026_monthly() -> BacktestConfig:
    cfg = make_prod_config_2024_2026()
    cfg.horizon = 21
    cfg.train_window = 252
    cfg.validation_window = 42
    cfg.rebalance_freq = 21
    cfg.retrain_freq = 63
    cfg.cov_lookback = 126
    cfg.min_history_rows = 180
    cfg.run_name = "sp500_prod_compare_2024_2026_monthly"
    return cfg

def make_prod_config_2014_plus() -> BacktestConfig:
    return BacktestConfig(
        start="2014-01-01",
        end=None,
        horizon=21,
        train_window=756,
        validation_window=63,
        rebalance_freq=21,
        retrain_freq=63,
        cov_lookback=252,
        transaction_cost=0.001,
        max_weight=12.0,
        top_n_equity=80,
        max_positions_per_strategy=None,
        min_history_rows=300,
        cache_path="sp500_defensive_yfinance_cache.pkl",
        refresh_cache=True,
        random_state=42,
        sp500_local_path="sp500_tickers.csv",
        sp500_universe_cache_path="sp500_universe_cache.csv",
        sp500_min_count=450,
        sp500_source_priority=(
            "local",
            "csv_github",
            "ssga_spy_holdings",
            "wikipedia",
            "universe_cache",
        ),
        download_chunk_size=10,
        download_sleep_sec=2.0,
        download_max_retries=4,
        download_threads=False,
        download_timeout=30,
        download_retry_failed_individual=True,
        download_individual_sleep_sec=0.5,
        download_silent=True,
        production_mode=True,
        output_dir="rebalancing_prod_runs",
        run_name="sp500_prod_compare",
        run_backtest=True,
        live_as_of_date=None,
        min_tradable_assets=100,
        max_data_lag_days=10,
        fail_on_data_quality=False,
        final_strategy="adaptive_regime_switch",
        save_artifacts=True,
    )


if __name__ == "__main__":
    cfg = make_prod_config_2024_2026()
    outputs = run_production_pipeline(cfg)
    print(outputs["metrics"])
    print(outputs["final_live_weights"].head(30))
