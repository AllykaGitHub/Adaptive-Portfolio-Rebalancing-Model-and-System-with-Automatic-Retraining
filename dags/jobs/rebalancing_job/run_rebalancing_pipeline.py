from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from production_rebalancing_pipeline import (
    BacktestConfig,
    make_prod_config_2024_2026,
    make_prod_config_2024_2026_monthly,
    make_prod_config_2014_plus,
    run_production_pipeline,
)


def _str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _apply_env_overrides(cfg: BacktestConfig) -> BacktestConfig:
    env = os.environ

    scalar_overrides = {
        "REBALANCING_START": ("start", str),
        "REBALANCING_END": ("end", str),
        "REBALANCING_HORIZON": ("horizon", int),
        "REBALANCING_TRAIN_WINDOW": ("train_window", int),
        "REBALANCING_VALIDATION_WINDOW": ("validation_window", int),
        "REBALANCING_REBALANCE_FREQ": ("rebalance_freq", int),
        "REBALANCING_RETRAIN_FREQ": ("retrain_freq", int),
        "REBALANCING_COV_LOOKBACK": ("cov_lookback", int),
        "REBALANCING_TRANSACTION_COST": ("transaction_cost", float),
        "REBALANCING_MAX_WEIGHT": ("max_weight", float),
        "REBALANCING_TOP_N_EQUITY": ("top_n_equity", int),
        "REBALANCING_MAX_POSITIONS": ("max_positions_per_strategy", int),
        "REBALANCING_MIN_HISTORY_ROWS": ("min_history_rows", int),
        "REBALANCING_CACHE_PATH": ("cache_path", str),
        "REBALANCING_OUTPUT_DIR": ("output_dir", str),
        "REBALANCING_RUN_NAME": ("run_name", str),
        "REBALANCING_FINAL_STRATEGY": ("final_strategy", str),
        "REBALANCING_LIVE_AS_OF_DATE": ("live_as_of_date", str),
        "REBALANCING_MIN_TRADABLE_ASSETS": ("min_tradable_assets", int),
        "REBALANCING_MAX_DATA_LAG_DAYS": ("max_data_lag_days", int),
        "REBALANCING_DOWNLOAD_CHUNK_SIZE": ("download_chunk_size", int),
        "REBALANCING_DOWNLOAD_SLEEP_SEC": ("download_sleep_sec", float),
        "REBALANCING_DOWNLOAD_MAX_RETRIES": ("download_max_retries", int),
        "REBALANCING_DOWNLOAD_TIMEOUT": ("download_timeout", int),
        "REBALANCING_INDIVIDUAL_SLEEP_SEC": ("download_individual_sleep_sec", float),
    }

    for env_name, (attr, caster) in scalar_overrides.items():
        if env_name in env and env[env_name] != "":
            setattr(cfg, attr, caster(env[env_name]))

    bool_overrides = {
        "REBALANCING_REFRESH_CACHE": "refresh_cache",
        "REBALANCING_RUN_BACKTEST": "run_backtest",
        "REBALANCING_FAIL_ON_DATA_QUALITY": "fail_on_data_quality",
        "REBALANCING_SAVE_ARTIFACTS": "save_artifacts",
        "REBALANCING_DOWNLOAD_THREADS": "download_threads",
        "REBALANCING_RETRY_FAILED_INDIVIDUAL": "download_retry_failed_individual",
        "REBALANCING_DOWNLOAD_SILENT": "download_silent",
    }
    for env_name, attr in bool_overrides.items():
        if env_name in env and env[env_name] != "":
            setattr(cfg, attr, _str_to_bool(env[env_name]))

    if env.get("REBALANCING_STRATEGIES"):
        cfg.strategy_names = tuple(s.strip() for s in env["REBALANCING_STRATEGIES"].split(",") if s.strip())

    if env.get("REBALANCING_SP500_SOURCE_PRIORITY"):
        cfg.sp500_source_priority = tuple(
            s.strip() for s in env["REBALANCING_SP500_SOURCE_PRIORITY"].split(",") if s.strip()
        )

    return cfg


def make_config(profile: str) -> BacktestConfig:
    if profile == "2024_2026":
        cfg = make_prod_config_2024_2026()
    elif profile == "2024_2026_monthly":
        cfg = make_prod_config_2024_2026_monthly()
    elif profile == "2014_plus":
        cfg = make_prod_config_2014_plus()
    else:
        cfg = BacktestConfig()

    cfg.sp500_local_path = os.environ.get("REBALANCING_SP500_LOCAL_PATH", "sp500_tickers.csv")
    return _apply_env_overrides(cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=os.environ.get("REBALANCING_PROFILE", "2024_2026"))
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()

    cfg = make_config(args.profile)
    outputs = run_production_pipeline(cfg)

    summary = {
        "run_dir": outputs.get("run_dir"),
        "latest_model_name": outputs.get("latest_model_name"),
        "live_prediction_date": str(outputs.get("live_prediction_date")),
        "metrics_rows": int(len(outputs.get("metrics", []))),
        "final_weights_rows": int(len(outputs.get("final_live_weights", []))),
        "artifact_paths": outputs.get("artifact_paths", {}),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
