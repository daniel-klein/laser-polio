import subprocess
import os
import json
import yaml
import optuna
from pathlib import Path

# From inside container
STORAGE_URL = "mysql://root@optuna-mysql:3306/optuna_db"
# From outside container
STORAGE_URL2 = "mysql+pymysql://root@127.0.0.1:3306/optuna_db"

def create_study_directory(study_name, model_config, calib_config):
    """Create a study directory and dump the model_config and calib_config."""
    study_dir = Path(study_name)
    study_dir.mkdir(parents=True, exist_ok=True)

    with open(study_dir / "model_config.yaml", "w") as model_file:
        model_file.write(model_config)
    with open(study_dir / "calib_config.yaml", "w") as calib_file:
        calib_file.write(calib_config)

    print(f"✅ Study directory '{study_name}' created with config files")

def get_default_config_values():
    """Run docker container with --help to retrieve default values for configs."""
    result = subprocess.run(
        ['docker', 'run', '--rm', 'idm-docker-staging.packages.idmod.org/laser/laser-polio:latest', '--help'],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise Exception("Docker command failed: " + result.stderr)

    help_output = result.stdout

    # Naive parsing: adjust based on actual help output format
    def extract_path(flag_name):
        for line in help_output.splitlines():
            if flag_name in line:
                parts = line.strip().split()
                for i, part in enumerate(parts):
                    if part.startswith(flag_name):
                        return parts[i+1] if i+1 < len(parts) else None
        return None

    model_config_path = extract_path("--model-config")
    calib_config_path = extract_path("--calib-config")

    # Read contents of the default config files (assumes they are part of the container image)
    model_config = f"# Default path: {model_config_path}\n"
    calib_config = f"# Default path: {calib_config_path}\n"

    return model_config, calib_config

def collect_study_results(study_name, output_dir):
    """Connect to the Optuna DB and write post-execution results to the output directory."""
    study = optuna.load_study(study_name=study_name, storage=STORAGE_URL2)

    best = study.best_trial
    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))

    with open(output_dir / "best_params.json", "w") as f:
        json.dump(best.params, f, indent=4)

    with open(output_dir / "study_metadata.json", "w") as f:
        json.dump(study.user_attrs, f, indent=4)

    trials_df.to_csv(output_dir / "trials.csv", index=False)

    print(f"Post-execution Optuna results saved to '{output_dir}'")

def collect_study_results_cli(study_name, output_dir, storage_url="mysql://root@optuna-mysql:3306/optuna_db"):
    """Run Optuna CLI commands to extract study info and save them to files."""
    output_dir.mkdir(exist_ok=True)

    def run_optuna_cmd(args, outfile):
        """Helper to run an optuna CLI command and write output to a file."""
        full_cmd = ['optuna'] + args + ['--study-name', study_name, '--storage', storage_url]
        result = subprocess.run(full_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Optuna CLI failed: {' '.join(full_cmd)}\n{result.stderr}")
        print(f"Saved {outfile}")

    # Collect stats, best trial, and trials list
    #run_optuna_cmd(['studies', 'stats'], "study_stats.txt")
    run_optuna_cmd(['studies', 'best-trial'], "best_trial.txt")
    run_optuna_cmd(['studies', 'trials'], "trials.txt")  # Not CSV, but text summary

    print("✅ Post-execution CLI-based Optuna reporting complete.")

def get_laser_polio_deps( study_name ):
    try:
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "cat",
                "idm-docker-staging.packages.idmod.org/laser/laser-polio:latest",
                "/app/laser_polio_deps.txt",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        deps_output = result.stdout
        with open(study_name + "/laser_polio_deps.txt", "w") as f:
            f.write(deps_output)
        print("✅ laser_polio_deps.txt retrieved and saved locally.")

    except subprocess.CalledProcessError as e:
        print("❌ Failed to retrieve laser_polio_deps.txt:")
        print(e.stderr)


def run_docker_calibration(study_name, num_trials=2):
    """Run the docker container to perform the calibration with a study."""
    model_config, calib_config = get_default_config_values()
    study_path = Path(study_name)

    # Step 1: Save initial config inputs
    create_study_directory(study_name, model_config, calib_config)
    get_laser_polio_deps(study_name)

    # Step 2: Launch container
    docker_command = [
        'docker', 'run', '--rm', '--name', 'calib_worker', '--network', 'optuna-network',
        '-e', f'STORAGE_URL={STORAGE_URL}',
        'idm-docker-staging.packages.idmod.org/laser/laser-polio:latest',
        '--study-name', study_name, '--num-trials', str(num_trials)
    ]

    result = subprocess.run(docker_command, capture_output=True, text=True)

    if result.returncode != 0:
        raise Exception("Docker calibration failed: " + result.stderr)

    print(f"✅ Calibration complete for study: {study_name}")

    # Step 3: Post-execution study reporting
    collect_study_results(study_name, study_path)
    #collect_study_results_cli(study_name, study_path) # , storage_url="sqlite:///optuna.db")
    from calib_report import plot_stuff
    plot_stuff( study_name, STORAGE_URL2 )

if __name__ == "__main__":
    run_docker_calibration("test_polio_calb", num_trials=2)
