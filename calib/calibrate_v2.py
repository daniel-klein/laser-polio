import json
import subprocess
import sys
from functools import partial
from pathlib import Path

import calib_db
import click
import optuna
import yaml
from logic import compute_fit
from logic import process_data

import laser_polio as lp

# ------------------- USER CONFIG -------------------
num_trials = 2
study_name = "calib_demo_zamfara_r0"
calib_config_path = lp.root / "calib/calib_configs/calib_pars_r0.yaml"
model_config_path = lp.root / "calib/model_configs/config_zamfara.yaml"
setup_sim_path = lp.root / "calib/setup_sim_v2.py"
results_path = lp.root / "calib/results" / study_name
PARAMS_FILE = "params.json"
ACTUAL_DATA_FILE = lp.root / "examples/calib_demo_zamfara/synthetic_infection_counts_zamfara_250.csv"
# ---------------------------------------------------


def objective(trial, calib_config, model_config_path):
    """Optuna objective function that runs the simulation and evaluates the fit."""
    results_file = results_path / "simulation_results.csv"
    if Path(results_file).exists():
        try:
            Path(results_file).unlink()
        except PermissionError as e:
            print(f"[WARN] Cannot delete file: {e}")

    # Generate suggested parameters from calibration config
    suggested_params = {}
    for name, spec in calib_config["parameters"].items():
        low = spec["low"]
        high = spec["high"]

        if isinstance(low, int) and isinstance(high, int):
            suggested_params[name] = trial.suggest_int(name, low, high)
        elif isinstance(low, float) or isinstance(high, float):
            suggested_params[name] = trial.suggest_float(name, float(low), float(high))
        else:
            raise TypeError(f"Cannot infer parameter type for '{name}'")

    # Save parameters to file (used by setup_sim)
    with open(PARAMS_FILE, "w") as f:
        json.dump(suggested_params, f, indent=4)

    # Run simulation using subprocess
    try:
        subprocess.run(
            [
                sys.executable,
                str(setup_sim_path),
                "--model-config",
                str(model_config_path),
                "--params-file",
                str(PARAMS_FILE),
                "--results-path",
                str(results_path),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Simulation failed: {e}")
        return float("inf")

    # Load results and compute fit
    actual = process_data(ACTUAL_DATA_FILE)
    predicted = process_data(results_file)
    return compute_fit(actual, predicted)


@click.command()
@click.option("--study-name", default=study_name, help="Name of the Optuna study.")
@click.option("--num-trials", default=num_trials, type=int, help="Number of optimization trials.")
@click.option("--calib-config", default=str(calib_config_path), help="Path to calibration parameter file.")
@click.option("--model-config", default=str(model_config_path), help="Path to model base config.")
def run_worker(study_name, num_trials, calib_config, model_config):
    """Run Optuna trials using setup_sim.py."""
    storage_url = calib_db.get_storage()

    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    except Exception:
        print(f"Creating new study: '{study_name}'")
        study = optuna.create_study(study_name=study_name, storage=storage_url)

    with open(calib_config) as f:
        calib_config_dict = yaml.safe_load(f)

    model_config_path = Path(model_config)

    # Attach metadata
    study.set_user_attr("parameter_spec", calib_config_dict.get("parameters", {}))
    for k, v in calib_config_dict.get("metadata", {}).items():
        study.set_user_attr(k, v)

    wrapped_objective = partial(
        objective,
        calib_config=calib_config_dict,
        model_config_path=model_config_path,
    )

    study.optimize(wrapped_objective, n_trials=num_trials)

    # Report best trial
    best = study.best_trial
    print("\nBest Trial:")
    print(f"  Value: {best.value}")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # Save results
    output_dir = Path(study_name)
    output_dir.mkdir(exist_ok=True)

    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df.to_csv(output_dir / "calibration_results.csv", index=False)

    with open(output_dir / "best_params.json", "w") as f:
        json.dump(best.params, f, indent=4)
    with open(output_dir / "study_metadata.json", "w") as f:
        json.dump(study.user_attrs, f, indent=4)

    print("✅ Calibration complete. Results saved.")


if __name__ == "__main__":
    run_worker()
