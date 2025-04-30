# calabaria_optuna.py
import calabaria as cb
import polars as pl
import yaml
from calabaria_model import LaserPolioConfig
from calabaria_model import LaserPolioModel

import laser_polio as lp

NUM_TRIALS = 5
NUM_REPS = 1

# --- Load model ---
default_config = LaserPolioConfig("zamfara_test")

# --- Parameter Spec ---

# TODO: Read from yml: calib_config or lp.root / "calib/calib_configs/calib_pars_r0.yaml"
calib_config = lp.root / "calib/calib_configs/r0.yaml"
with open(calib_config) as f:
    d = yaml.safe_load(f)
    params = d["parameters"]
    # rename 'low' and 'high' to 'lower' and 'upper'
    for param in params.values():
        param["lower"] = param.pop("low")
        param["upper"] = param.pop("high")
    specs = cb.parameters.core.ParameterSpecs.from_dict(params)
"""
specs = cb.parameters.core.ParameterSpecs(
    {
        "r0": cb.ParameterSpec(name="r0", lower=5.0, upper=15.0),
    }
)
"""

model = LaserPolioModel(config=default_config, pars=cb.ParameterSet(specs=specs))

# --- Load (Synthetic) Data ---
actual_data_file = lp.root / "examples/calib_demo_zamfara/synthetic_infection_counts_zamfara_250.csv"
obs_df = pl.read_csv(actual_data_file).with_columns((pl.col("S") + pl.col("E") + pl.col("I") + pl.col("R")).alias("N"))

# --- Target ---
target = cb.Target(
    name="infected_over_time",
    data=obs_df,
    alignment=cb.JoinAlignment(on_cols="Time", mode="exact"),
    evaluation=cb.eval.beta_binomial_nll(x_col="I", n_col="N"),
)

targets = cb.Targets([target])

# --- Build CalibrationTask ---
dispatcher = cb.SerialSimDispatcher(model=model)
task = cb.CalibrationTask(model=model, targets=targets, sim_dispatcher=dispatcher, replicates=NUM_REPS)

# --- Set up Optuna with parallel dispatcher ---
engine = cb.OptunaEngine(
    n_trials=NUM_TRIALS,
    # dispatcher=cb.JoblibCalibrationDispatcher(n_jobs=4),
    dispatcher=cb.SerialCalibrationDispatcher(),
    if_study_exists="overwrite",
    verbose=True,
)

# --- Run the calibration ---
result = engine.fit(task)

# --- Print results ---
print("\n📈 Best Parameters:", result.study.best_params)
print("📉 Best Loss:", result.study.best_value)
