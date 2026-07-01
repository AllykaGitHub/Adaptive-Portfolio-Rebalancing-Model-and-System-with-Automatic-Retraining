from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

JOB_DIR = Path(__file__).resolve().parent / "jobs" / "rebalancing_job"
if str(JOB_DIR) not in sys.path:
    sys.path.insert(0, str(JOB_DIR))

DEFAULT_PROFILE = os.environ.get("REBALANCING_PROFILE", "2024_2026")
DEFAULT_WORK_ROOT = os.environ.get(
    "REBALANCING_AIRFLOW_WORK_ROOT",
    os.environ.get("REBALANCING_OUTPUT_DIR", "/tmp/rebalancing_airflow_runs"),
)
DEFAULT_WORK_DIR = (
    DEFAULT_WORK_ROOT.rstrip("/")
    + "/{{ dag.dag_id }}/{{ dag_run.run_id | replace(':', '_') | replace('/', '_') }}"
)


def load_market_data_callable(profile: str, work_dir: str, **context):
    from dag_steps import load_market_data_task
    return load_market_data_task(profile=profile, work_dir=work_dir, **context)


def validate_data_callable(profile: str, work_dir: str, **context):
    from dag_steps import validate_data_task
    return validate_data_task(profile=profile, work_dir=work_dir, **context)


def build_features_callable(profile: str, work_dir: str, **context):
    from dag_steps import build_features_task
    return build_features_task(profile=profile, work_dir=work_dir, **context)


def compare_strategies_callable(profile: str, work_dir: str, **context):
    from dag_steps import compare_strategies_task
    return compare_strategies_task(profile=profile, work_dir=work_dir, **context)


def fit_live_weights_callable(profile: str, work_dir: str, **context):
    from dag_steps import fit_live_weights_task
    return fit_live_weights_task(profile=profile, work_dir=work_dir, **context)


def save_artifacts_callable(profile: str, work_dir: str, **context):
    from dag_steps import save_artifacts_task
    return save_artifacts_task(profile=profile, work_dir=work_dir, **context)


default_args = {
    "owner": "data_science",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="portfolio_rebalancing_pipeline",
    description="Portfolio rebalancing pipeline split into Airflow PythonOperator stages",
    tags=["portfolio", "rebalancing", "ml", "backtest"],
    schedule_interval="0 9 * * 1-5",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
) as dag:
    load_market_data = PythonOperator(
        task_id="load_market_data",
        python_callable=load_market_data_callable,
        op_kwargs={"profile": DEFAULT_PROFILE, "work_dir": DEFAULT_WORK_DIR},
        executor_config={"flavor": "2cpu-16ram", "time_limit": "30m"},
    )

    validate_data = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data_callable,
        op_kwargs={"profile": DEFAULT_PROFILE, "work_dir": DEFAULT_WORK_DIR},
        executor_config={"flavor": "2cpu-16ram", "time_limit": "30m"},
    )

    build_features = PythonOperator(
        task_id="build_features",
        python_callable=build_features_callable,
        op_kwargs={"profile": DEFAULT_PROFILE, "work_dir": DEFAULT_WORK_DIR},
        executor_config={"flavor": "2cpu-16ram", "time_limit": "30m"},
    )

    compare_strategies = PythonOperator(
        task_id="compare_strategies",
        python_callable=compare_strategies_callable,
        op_kwargs={"profile": DEFAULT_PROFILE, "work_dir": DEFAULT_WORK_DIR},
        executor_config={"flavor": "2cpu-16ram", "time_limit": "30m"},
    )

    fit_live_weights = PythonOperator(
        task_id="fit_live_weights",
        python_callable=fit_live_weights_callable,
        op_kwargs={"profile": DEFAULT_PROFILE, "work_dir": DEFAULT_WORK_DIR},
        executor_config={"flavor": "2cpu-16ram", "time_limit": "30m"},
    )

    save_artifacts = PythonOperator(
        task_id="save_artifacts",
        python_callable=save_artifacts_callable,
        op_kwargs={"profile": DEFAULT_PROFILE, "work_dir": DEFAULT_WORK_DIR},
        executor_config={"flavor": "8cpu-128ram", "time_limit": "360m"},
    )

    load_market_data >> validate_data >> build_features >> compare_strategies >> fit_live_weights >> save_artifacts
