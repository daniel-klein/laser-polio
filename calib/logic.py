import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def process_data(filename):
    """Load simulation results and extract features for comparison."""
    df = pd.read_csv(filename)
    return {
        "total_infected": df["I"].sum(),
        "peak_infection_time": df.loc[df["I"].idxmax(), "Time"],
    }


def compute_fit(actual, predicted, use_squared=False, normalize=False, weights=None):
    """Compute distance between actual and predicted summary metrics."""
    fit = 0
    weights = weights or {}

    for key in actual:
        v1 = np.array(actual[key], dtype=float)
        v2 = np.array(predicted[key], dtype=float)
        gofs = np.abs(v1 - v2)

        if normalize and v1.max() > 0:
            gofs /= v1.max()
        if use_squared:
            gofs **= 2

        loss_weight = weights.get(key, 1)
        fit += (gofs * loss_weight).sum()

    return fit


def objective(trial, calib_config, model_config_path, sim_path, results_path, params_file, actual_data_file):
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
    with open(params_file, "w") as f:
        json.dump(suggested_params, f, indent=4)

    # Run simulation using subprocess
    try:
        subprocess.run(
            [
                sys.executable,
                str(sim_path),
                "--model-config",
                str(model_config_path),
                "--params-file",
                str(params_file),
                "--results-path",
                str(results_path),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Simulation failed: {e}")
        return float("inf")

    # Load results and compute fit
    actual = process_data(actual_data_file)
    predicted = process_data(results_file)
    return compute_fit(actual, predicted)


# # Default paths can live here too if you want
# DEFAULT_STUDY_NAME = "calib_demo_zamfara_r0"
# DEFAULT_CALIB_PATH = lp.root / "calib/calib_configs/calib_pars_r0.yaml"
# DEFAULT_MODEL_PATH = lp.root / "calib/model_configs/config_zamfara.yaml"
# DEFAULT_RESULTS_PATH = lp.root / "calib/results" / DEFAULT_STUDY_NAME
# DEFAULT_SIM_PATH = lp.root / "calib/setup_sim_v2.py"
# DEFAULT_PARAMS_FILE = "params.json"
# DEFAULT_ACTUAL_FILE = lp.root / "examples/calib_demo_zamfara/synthetic_infection_counts_zamfara_250.csv"


# @click.command()
# @click.option("--study-name", default=DEFAULT_STUDY_NAME)
# @click.option("--num-trials", default=10, type=int)
# @click.option("--calib-config", default=str(DEFAULT_CALIB_PATH))
# @click.option("--model-config", default=str(DEFAULT_MODEL_PATH))
# @click.option("--results-path", default=str(DEFAULT_RESULTS_PATH))
# @click.option("--sim-path", default=str(DEFAULT_SIM_PATH))
# @click.option("--params-file", default=DEFAULT_PARAMS_FILE)
# @click.option("--actual-data-file", default=str(DEFAULT_ACTUAL_FILE))
# def run_worker(
#     study_name,
#     num_trials,
#     calib_config,
#     model_config,
#     results_path,
#     sim_path,
#     params_file,
#     actual_data_file,
# ):
#     """Run Optuna trials using setup_sim.py."""
#     storage_url = calib_db.get_storage()

#     try:
#         study = optuna.load_study(study_name=study_name, storage=storage_url)
#     except Exception:
#         print(f"Creating new study: '{study_name}'")
#         study = optuna.create_study(study_name=study_name, storage=storage_url)

#     with open(calib_config) as f:
#         calib_config_dict = yaml.safe_load(f)

#     model_config_path = Path(model_config)

#     study.set_user_attr("parameter_spec", calib_config_dict.get("parameters", {}))
#     for key, value in calib_config_dict.get("metadata", {}).items():
#         study.set_user_attr(key, value)

#     wrapped_objective = partial(
#         objective,
#         calib_config=calib_config_dict,
#         model_config_path=model_config_path,
#         sim_path=Path(sim_path),
#         results_path=Path(results_path),
#         params_file=params_file,
#         actual_data_file=Path(actual_data_file),
#     )

#     study.optimize(wrapped_objective, n_trials=num_trials)

#     best = study.best_trial
#     print("\nBest Trial:")
#     print(f"  Value: {best.value}")
#     for k, v in best.params.items():
#         print(f"    {k}: {v}")

#     output_dir = Path(study_name)
#     output_dir.mkdir(exist_ok=True)

#     df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
#     df.to_csv(output_dir / "calibration_results.csv", index=False)

#     with open(output_dir / "best_params.json", "w") as f:
#         json.dump(best.params, f, indent=4)
#     with open(output_dir / "study_metadata.json", "w") as f:
#         json.dump(study.user_attrs, f, indent=4)

#     print("✅ Calibration complete. Results saved.")
