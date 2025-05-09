# calabaria_optuna.py
import calabaria as cb
import yaml
from calabaria_model import LaserPolioConfig
from calabaria_model import LaserPolioModel
from optuna.samplers import GPSampler
from targets import calc_calib_targets_paralysis

import laser_polio as lp

debug = True
NUM_TRIALS = [10, 2][debug]  # Per task
NUM_REPS = [1, 1][debug]
NUM_JOBLIB_TASKS = [10, 1][debug]  # Uses SerialDispatcher if NUM_JOBLIB_TASKS == 1

# --- Load model ---
laser_config = LaserPolioConfig("nigeria_6y", model_config=lp.root / "calib/model_configs/config_nigeria_6y.yaml")

# --- Parameter Spec ---
calib_config = lp.root / "calib/calib_configs/r0_k_ssn.yaml"
with open(calib_config) as f:
    d = yaml.safe_load(f)
    params = d["parameters"]
    # rename 'low' and 'high' to 'lower' and 'upper'
    for param in params.values():
        param["lower"] = param.pop("low")
        param["upper"] = param.pop("high")
    specs = cb.ParameterSpecs.from_dict(params)

model = LaserPolioModel(config=laser_config, pars=cb.ParameterSet(specs=specs))

# --- Load or create (Synthetic) Data ---
# actual_data_file = lp.root / "examples/calib_demo_zamfara/synthetic_infection_counts_zamfara_250.csv"
# actual_data_file = lp.root / "results/demo_zamfara/actual_data.csv"
# obs_df = pl.read_csv(actual_data_file).with_columns((pl.col("S") + pl.col("E") + pl.col("I") + pl.col("R")).alias("N"))
obs_df = model.extract_actual_data()

obs_df.to_csv(model.config.results_path / "actual_data.csv", index=False)

lp_targets = calc_calib_targets_paralysis(filename=obs_df, model_config_path=laser_config.model_config, is_actual_data=True)

# --- Target ---
# total_infected = obs_df.select(pl.col("I").sum().alias("total_infected"))
total_tgt = cb.Target(
    model_output="total_infected",
    data=lp_targets["total_infected"],
    alignment=cb.JoinAlignment(on_cols=[], mode="exact"),
    # evaluation=cb.eval.beta_binomial_nll(x_col="I", n_col="N"),
    evaluation=cb.eval.Evaluator(
        aggregator=cb.eval.IdentityAggregator(),
        loss_fn=cb.loss.NormalNLL(obs_stdev=10, x_col="total_infected"),
        reducer=cb.eval.MeanReducer(),
        weight=1.0,
    ),
)

if False:
    # Compute monthly summary of the "I" column and filter for months <= 12
    """
    yearly_cases = (
        obs_df.with_columns((pl.col("Time") // 30 + 1).alias("month"))
        .filter(pl.col("month") <= 12)  # Time is 0 to 365
        .group_by("month")
        .agg(pl.col("I").sum().alias("monthly_cases"))
        .sort("month")
        .select("monthly_cases")  # Select only the "monthly_cases" column
        .transpose(include_header=True, column_names=[str(i) for i in range(1, 13)])
    )
    """
    yearly_tgt = cb.Target(
        model_output="yearly_cases",
        data=lp_targets["yearly_cases"],
        alignment=cb.JoinAlignment(on_cols=[], mode="exact"),
        # evaluation=cb.eval.beta_binomial_nll(x_col="I", n_col="N"),
        evaluation=cb.eval.Evaluator(
            aggregator=cb.eval.IdentityAggregator(),
            loss_fn=cb.loss.DirichletMultinomialNLL(cols=range(1, 13)),
            reducer=cb.eval.MeanReducer(),
            weight=1.0,
        ),
    )

    # Compute monthly summary of the "I" column and filter for months <= 12
    """
    monthly_cases = (
        obs_df.with_columns((pl.col("Time") // 30 + 1).alias("month"))
        .filter(pl.col("month") <= 12)  # Time is 0 to 365
        .group_by("month")
        .agg(pl.col("I").sum().alias("monthly_cases"))
        .sort("month")
        .select("monthly_cases")  # Select only the "monthly_cases" column
        .transpose(include_header=True, column_names=[str(i) for i in range(1, 13)])
    )
    """
    monthly_tgt = cb.Target(
        model_output="monthly_cases",
        data=lp_targets["monthly_cases"],
        alignment=cb.JoinAlignment(on_cols=[], mode="exact"),
        # evaluation=cb.eval.beta_binomial_nll(x_col="I", n_col="N"),
        evaluation=cb.eval.Evaluator(
            aggregator=cb.eval.IdentityAggregator(),
            loss_fn=cb.loss.DirichletMultinomialNLL(cols=range(1, 13)),
            reducer=cb.eval.MeanReducer(),
            weight=1.0,
        ),
    )

    monthly_timeseries_tgt = cb.Target(
        model_output="monthly_timeseries",
        data=lp_targets["monthly_timeseries"],
        alignment=cb.JoinAlignment(on_cols=[], mode="exact"),
        # evaluation=cb.eval.beta_binomial_nll(x_col="I", n_col="N"),
        evaluation=cb.eval.Evaluator(
            aggregator=cb.eval.IdentityAggregator(),
            loss_fn=cb.loss.DirichletMultinomialNLL(cols=range(1, 13)),
            reducer=cb.eval.MeanReducer(),
            weight=1.0,
        ),
    )

    regional_tgt = []
    if "summary_config" in laser_config.model_config:
        regional_tgt = [
            cb.Target(
                model_output="regional_cases",
                data=lp_targets["regional_cases"],
                alignment=cb.JoinAlignment(on_cols=[], mode="exact"),
                # evaluation=cb.eval.beta_binomial_nll(x_col="I", n_col="N"),
                evaluation=cb.eval.Evaluator(
                    aggregator=cb.eval.IdentityAggregator(),
                    loss_fn=cb.loss.DirichletMultinomialNLL(cols=range(1, 13)),
                    reducer=cb.eval.MeanReducer(),
                    weight=1.0,
                ),
            )
        ]


# targets = cb.Targets([total_tgt, yearly_tgt, monthly_tgt, monthly_timeseries_tgt] + regional_tgt)  # total_tgt, monthly_tgt])
targets = cb.Targets([total_tgt])

# --- Build CalibrationTask ---
dispatcher = cb.SerialSimDispatcher(model=model)
task = cb.CalibrationTask(model=model, targets=targets, sim_dispatcher=dispatcher, replicates=NUM_REPS)

# --- Set up Optuna with parallel dispatcher ---
engine = cb.OptunaEngine(
    study_name="calabaria_test",
    n_trials=NUM_TRIALS,
    dispatcher=cb.SerialCalibrationDispatcher() if NUM_JOBLIB_TASKS == 1 else cb.JoblibCalibrationDispatcher(n_jobs=NUM_JOBLIB_TASKS),
    sampler=GPSampler(n_startup_trials=10, deterministic_objective=False),
    if_study_exists="overwrite",
    verbose=True,
)

# --- Run the calibration ---
result = engine.fit(task)

# --- Print results ---
print("\n📈 Best Parameters:", result.study.best_params)
print("📉 Best Loss:", result.study.best_value)

# --- Results ---
print(result.summary())
print(task.telemetry.report())
result.save_diagnostics()

# Save trials to csv
trials_df = result.study.trials_dataframe()
trials_df.to_csv("trials.csv", index=False)
print("Trials saved to trials.csv")
