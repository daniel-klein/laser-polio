import click
from logic import run_worker_main
from pathlib import Path

# TODO: Pass the process_data function to the calibrator


# ------------------- USER CONFIG -------------------
num_trials = 2
study_name = "my_polio_calibration_study"
calib_config_path = "calib_pars.yaml"
model_config_path = "config.yaml"
sim_path = "laser.py"
results_path = Path( "calib/results" ) / study_name
params_file = "params.json"
actual_data_file = "examples/calib_demo/synthetic_infection_counts.csv"
# ---------------------------------------------------


@click.command()
@click.option("--study-name", default=study_name, show_default=True, help="Name of the Optuna study.")
@click.option("--num-trials", default=num_trials, show_default=True, type=int, help="Number of optimization trials.")
@click.option("--calib-config", default=str(calib_config_path), show_default=True, help="Path to calibration parameter file.")
@click.option("--model-config", default=str(model_config_path), show_default=True, help="Path to model base config.")
@click.option("--results-path", default=str(results_path), show_default=True )
@click.option("--sim-path", default=str(sim_path), show_default=True )
@click.option("--params-file", default=str(params_file), show_default=True )
@click.option("--actual-data-file", default=str(actual_data_file), show_default=True )
def run_worker(**kwargs):
    run_worker_main(**kwargs)


if __name__ == "__main__":
    run_worker()
