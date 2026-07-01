from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


def _import_make_config():
    from run_rebalancing_pipeline import make_config
    return make_config


def _import_pipeline_functions():
    from production_rebalancing_pipeline import (
        assert_data_quality_ok,
        build_price_dataset,
        data_quality_report,
        make_prod_config_2024_2026,
        run_production_pipeline,
    )
    return {
        "assert_data_quality_ok": assert_data_quality_ok,
        "build_price_dataset": build_price_dataset,
        "data_quality_report": data_quality_report,
        "run_production_pipeline": run_production_pipeline,
        "make_prod_config_2024_2026": make_prod_config_2024_2026,
    }


def _str_to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_run_id(value: Any) -> str:
    return str(value or "manual").replace(":", "_").replace("/", "_")


def _dag_id(context: Dict[str, Any]) -> str:
    dag_obj = context.get("dag")
    if hasattr(dag_obj, "dag_id"):
        return str(dag_obj.dag_id)
    return "portfolio_rebalancing_pipeline"


def _run_id(context: Dict[str, Any]) -> str:
    return _safe_run_id(context.get("run_id", "manual"))


def _work_dir(work_dir: Optional[str] = None, **context: Any) -> str:
    if work_dir:
        path = work_dir
    else:
        root = os.environ.get(
            "REBALANCING_AIRFLOW_WORK_ROOT",
            os.environ.get("REBALANCING_OUTPUT_DIR", "/tmp/rebalancing_airflow_runs"),
        )
        path = os.path.join(root, _dag_id(context), _run_id(context))
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def _config(profile: str = "2024_2026", work_dir: Optional[str] = None):
    make_config = _import_make_config()
    cfg = make_config(profile)
    if work_dir:
        wd = Path(work_dir)
        cfg.output_dir = str(wd / "artifacts")
        cfg.cache_path = str(wd / Path(str(cfg.cache_path)).name)
        cfg.sp500_universe_cache_path = str(wd / Path(str(cfg.sp500_universe_cache_path)).name)
    return cfg


def _summary(stage: str, status: str, **kwargs: Any) -> Dict[str, Any]:
    out = {"stage": stage, "status": status}
    out.update(kwargs)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    return out


def _stateless_mode() -> bool:
    return _str_to_bool(os.environ.get("REBALANCING_STATELESS_MODE", "true"), default=True)


def _light_stage_mode() -> bool:
    return _str_to_bool(os.environ.get("REBALANCING_LIGHT_STAGE_TASKS", "true"), default=True)


def load_market_data_task(profile: str = "2024_2026", work_dir: Optional[str] = None, **context: Any) -> Dict[str, Any]:
    """
    Airflow/K8S task containers do not share local files. In the default stateless mode this task
    performs only a cheap preflight and returns a small XCom dict. The full market-data load is
    executed in save_artifacts_task inside one process, so no intermediate .pkl is passed between pods.

    If your Airflow instance has a real shared mounted storage and you want this task to do the heavy
    download as a separate stage, set REBALANCING_LIGHT_STAGE_TASKS=false. Even then, downstream tasks
    will not read local pkl files from this pod; use this mode only as an isolated data-source check.
    """
    wd = _work_dir(work_dir, **context)
    cfg = _config(profile, wd)

    if _light_stage_mode():
        return _summary(
            "load_market_data",
            "preflight_ok",
            profile=profile,
            work_dir=wd,
            note="Stateless Airflow mode: heavy data loading runs in the final save_artifacts task.",
        )

    funcs = _import_pipeline_functions()
    df_all, meta, failed = funcs["build_price_dataset"](cfg)
    return _summary(
        "load_market_data",
        "ok",
        profile=profile,
        work_dir=wd,
        rows=len(df_all),
        universe_rows=len(meta),
        failed_downloads=len(failed),
    )


def validate_data_task(profile: str = "2024_2026", work_dir: Optional[str] = None, **context: Any) -> Dict[str, Any]:
    wd = _work_dir(work_dir, **context)
    cfg = _config(profile, wd)

    if _light_stage_mode():
        return _summary(
            "validate_data",
            "preflight_ok",
            profile=profile,
            work_dir=wd,
            min_tradable_assets=cfg.min_tradable_assets,
            max_data_lag_days=cfg.max_data_lag_days,
            note="No local artifact is read from upstream task. Full validation runs in save_artifacts.",
        )

    funcs = _import_pipeline_functions()
    df_all, meta, failed = funcs["build_price_dataset"](cfg)
    quality = funcs["data_quality_report"](df_all, meta, failed, cfg)
    funcs["assert_data_quality_ok"](quality, cfg)
    return _summary(
        "validate_data",
        "ok",
        profile=profile,
        work_dir=wd,
        quality_rows=len(quality),
        failed_downloads=len(failed),
    )


def build_features_task(profile: str = "2024_2026", work_dir: Optional[str] = None, **context: Any) -> Dict[str, Any]:
    wd = _work_dir(work_dir, **context)
    cfg = _config(profile, wd)
    return _summary(
        "build_features",
        "preflight_ok",
        profile=profile,
        work_dir=wd,
        horizon=cfg.horizon,
        min_history_rows=cfg.min_history_rows,
        note="Feature engineering runs inside save_artifacts in stateless mode.",
    )


def compare_strategies_task(profile: str = "2024_2026", work_dir: Optional[str] = None, **context: Any) -> Dict[str, Any]:
    wd = _work_dir(work_dir, **context)
    cfg = _config(profile, wd)
    return _summary(
        "compare_strategies",
        "preflight_ok",
        profile=profile,
        work_dir=wd,
        run_backtest=cfg.run_backtest,
        strategies=list(cfg.strategy_names),
        note="Strategy comparison runs inside save_artifacts in stateless mode.",
    )


def fit_live_weights_task(profile: str = "2024_2026", work_dir: Optional[str] = None, **context: Any) -> Dict[str, Any]:
    wd = _work_dir(work_dir, **context)
    cfg = _config(profile, wd)
    return _summary(
        "fit_live_weights",
        "preflight_ok",
        profile=profile,
        work_dir=wd,
        final_strategy=cfg.final_strategy,
        max_weight=cfg.max_weight,
        max_positions_per_strategy=cfg.max_positions_per_strategy,
        note="Live model fitting and weights run inside save_artifacts in stateless mode.",
    )


def save_artifacts_task(profile: str = "2024_2026", work_dir: Optional[str] = None, **context: Any) -> Dict[str, Any]:
    """Final task that executes the complete production pipeline in one pod.

    This keeps the multi-task Airflow graph from the example, but avoids broken local-file handoff
    between K8S pods. Only small JSON summaries are passed through XCom.
    """
    wd = _work_dir(work_dir, **context)
    cfg = _config(profile, wd)
    cfg.output_dir = str(Path(wd) / "artifacts")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    funcs = _import_pipeline_functions()
    outputs = funcs["run_production_pipeline"](cfg)

    artifact_paths = outputs.get("artifact_paths", {}) or {}
    summary = {
        "stage": "save_artifacts",
        "status": "ok",
        "profile": profile,
        "work_dir": wd,
        "run_dir": outputs.get("run_dir"),
        "latest_model_name": outputs.get("latest_model_name"),
        "live_prediction_date": str(outputs.get("live_prediction_date")),
        "metrics_rows": int(len(outputs.get("metrics", []))),
        "final_weights_rows": int(len(outputs.get("final_live_weights", []))),
        "artifact_paths": artifact_paths,
    }

    summary_path = Path(cfg.output_dir) / "airflow_task_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    summary["summary_path"] = str(summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return summary
