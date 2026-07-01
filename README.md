# Portfolio Rebalancing Airflow DAG

Multi-task Airflow DAG with the same visual architecture as the MLC Airflow example:

```text
load_market_data
    ↓
validate_data
    ↓
build_features
    ↓
compare_strategies
    ↓
fit_live_weights
    ↓
save_artifacts
```

Important for MLCore/K8S: Airflow task pods do not share local filesystem state. Therefore this version does not pass local `.pkl` files between tasks. The first five tasks are lightweight preflight gates and return only small JSON XCom values. The final `save_artifacts` task runs the full production pipeline in one pod and saves all artifacts.

## Publish

```bash
mlc airflow publish -p mlcpsando <airflow_instance_name>
```

## Manual job submit

```bash
mlc job submit -p mlcpsando --preset-file ./dags/jobs/rebalancing_job/preset.yml
```

## Useful env

```yaml
env:
  REBALANCING_PROFILE: "2024_2026"
  REBALANCING_STATELESS_MODE: "true"
  REBALANCING_LIGHT_STAGE_TASKS: "true"
  REBALANCING_REFRESH_CACHE: "true"
  REBALANCING_RUN_BACKTEST: "true"
  REBALANCING_FINAL_STRATEGY: "adaptive_regime_switch"
  REBALANCING_MAX_WEIGHT: "12"
  REBALANCING_OUTPUT_DIR: "/tmp/rebalancing_airflow_runs"
```

If your instance has a real shared persistent mount and you want every stage to read/write intermediate files, you need to implement storage on that mount or object storage. Local `/tmp` and often `/work` are pod-local and are not safe for cross-task artifacts.
