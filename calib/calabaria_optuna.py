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
default_config = LaserPolioConfig("zamfara_test", model_config=lp.root / "calib/model_configs/config_zamfara.yaml")

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
    specs = cb.ParameterSpecs.from_dict(params)

print("WARNING")
specs = cb.ParameterSpecs(
    {
        "x_r0": cb.ParameterSpec(name="x_r0", lower=0.5, upper=2.0, transform="log10"),
    }
)


model = LaserPolioModel(config=default_config, pars=cb.ParameterSet(specs=specs))

# --- Load (Synthetic) Data ---
actual_data_file = lp.root / "examples/calib_demo_zamfara/synthetic_infection_counts_zamfara_250.csv"
obs_df = pl.read_csv(actual_data_file).with_columns((pl.col("S") + pl.col("E") + pl.col("I") + pl.col("R")).alias("N"))

# --- Target ---
# total_infected = obs_df["I"].sum()
total_infected = obs_df.select(pl.col("I").sum().alias("total_infected"))
target = cb.Target(
    model_output="total_infected",
    data=total_infected,
    alignment=cb.JoinAlignment(on_cols=[], mode="exact"),
    # evaluation=cb.eval.beta_binomial_nll(x_col="I", n_col="N"),
    evaluation=cb.eval.Evaluator(
        aggregator=cb.eval.IdentityAggregator(),
        loss_fn=cb.loss.NormalNLL(obs_stdev=10, x_col="total_infected"),
        reducer=cb.eval.MeanReducer(),
        weight=1.0,
    ),
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
