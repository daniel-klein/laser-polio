import numpy as np
import pandas as pd
import yaml

import laser_polio as lp


def calc_calib_targets_paralysis(filename, model_config_path=None, is_actual_data=True):
    """Load simulation results and extract features for comparison."""

    # Load the data & config
    if isinstance(filename, pd.DataFrame):
        df = filename
    else:
        df = pd.read_csv(filename)
    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)

    # Parse dates to datetime object if needed
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Choose the column to summarize
    if is_actual_data:
        case_col = "P"
        scale_factor = 1.0
    else:
        case_col = "new_exposed"
        scale_factor = 1 / 2000.0
        # The actual data is in months & the sim has a tendency to rap into the next year (e.g., 2020-01-01) so we need to exclude and dates beyond the last month of the actual data
        max_date = lp.find_latest_end_of_month(df["date"])
        df = df[df["date"] <= max_date]

    targets = {}

    # 1. Total infected (scaled if simulated)
    targets["total_infected"] = np.array([df[case_col].sum() * scale_factor])

    # 2. Yearly cases
    targets["yearly_cases"] = df.groupby("year")[case_col].sum().values * scale_factor

    # 3. Monthly cases
    targets["monthly_cases"] = df.groupby("month")[case_col].sum().values * scale_factor

    # 4. Monthly timeseries (sorted full date x month time series)
    monthly_df = df.groupby([df["date"].dt.to_period("M")])[case_col].sum().sort_index().astype(float) * scale_factor
    targets["monthly_timeseries"] = monthly_df.values

    # 5. Regional group cases
    if model_config and "summary_config" in model_config:
        region_groups = model_config["summary_config"].get("region_groups", {})
        regional_cases = []
        for name in region_groups:
            node_list = region_groups[name]
            total = df[df["node"].isin(node_list)][case_col].sum() * scale_factor
            regional_cases.append(total)
        targets["regional_cases"] = np.array(regional_cases)

    print(f"{targets=}")
    return targets


def get_smoothed_node_case_presence(sim_df, case_column="new_exposed", cir=1 / 2000, n_draws=100, seed=42):
    np.random.seed(seed)
    sim_df = sim_df.copy()
    sim_df["date"] = pd.to_datetime(sim_df["date"])
    sim_df["month"] = sim_df["date"].dt.to_period("M")

    all_months = sim_df["month"].sort_values().unique()

    monthly_counts = np.zeros((n_draws, len(all_months)))
    total_counts = np.zeros(n_draws)

    for i in range(n_draws):
        expected = sim_df[case_column].values * cir
        paralytic_cases = np.random.poisson(expected)

        sim_df["has_case"] = paralytic_cases > 0

        total_counts[i] = sim_df.loc[sim_df["has_case"], "node"].nunique()

        monthly_group = sim_df[sim_df["has_case"]].groupby("month")["node"].nunique()
        monthly_group = monthly_group.reindex(all_months, fill_value=0)
        monthly_counts[i, :] = monthly_group.values

    return {"nodes_with_cases_total": total_counts.mean(), "nodes_with_cases_timeseries": monthly_counts.mean(axis=0)}


def calc_targets_temporal_regional_nodes(filename, model_config_path=None, is_actual_data=True):
    """Load simulation results and extract features for comparison."""

    # Load the data & config
    df = pd.read_csv(filename)
    model_config = {}
    if model_config_path is not None:
        with open(model_config_path) as f:
            model_config = yaml.safe_load(f)

    # Parse dates to datetime object if needed
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Choose the column to summarize
    if is_actual_data:
        case_col = "P"
        scale_factor = 1.0
    else:
        case_col = "new_exposed"
        scale_factor = 1 / 2000.0
        # The actual data is in months & the sim has a tendency to rap into the next year (e.g., 2020-01-01) so we need to exclude and dates beyond the last month of the actual data
        max_date = lp.find_latest_end_of_month(df["date"])
        df = df[df["date"] <= max_date]

    targets = {}

    # --- Temporal aggregation ---
    targets["total_infected"] = np.array([df[case_col].sum() * scale_factor])
    targets["yearly_cases"] = df.groupby("year")[case_col].sum().values * scale_factor
    targets["monthly_cases"] = df.groupby("month")[case_col].sum().values * scale_factor
    monthly_df = df.groupby([df["date"].dt.to_period("M")])[case_col].sum().sort_index().astype(float) * scale_factor
    targets["monthly_timeseries"] = monthly_df.values

    # --- Regional aggregation ---
    if model_config and "summary_config" in model_config:
        region_groups = model_config["summary_config"].get("region_groups", {})
        regional_cases = []
        for name in region_groups:
            node_list = region_groups[name]
            total = df[df["node"].isin(node_list)][case_col].sum() * scale_factor
            regional_cases.append(total)
        targets["regional_cases"] = np.array(regional_cases)

    # --- Number of nodes with cases ---
    if is_actual_data:
        has_case = df[case_col] > 0
        targets["nodes_with_cases_total"] = np.array([df.loc[has_case, "node"].nunique()])

        df["month_period"] = df["date"].dt.to_period("M")
        all_months = df["month_period"].sort_values().unique()
        monthly_node_counts = df.loc[has_case].groupby("month_period")["node"].nunique().sort_index()

        monthly_node_counts = monthly_node_counts.reindex(all_months, fill_value=0)
        targets["nodes_with_cases_timeseries"] = monthly_node_counts.values
    if not is_actual_data:
        node_case_metrics = get_smoothed_node_case_presence(df)
        targets["nodes_with_cases_total"] = np.array([node_case_metrics["nodes_with_cases_total"]])
        targets["nodes_with_cases_timeseries"] = node_case_metrics["nodes_with_cases_timeseries"]

    print(f"{targets=}")
    return targets


def calc_calib_targets(filename, model_config_path=None):
    """Load simulation results and extract features for comparison."""

    # Load the data & config
    df = pd.read_csv(filename)
    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)

    # Parse dates to datetime object if needed
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month

    targets = {}

    # 1. Total infected
    targets["total_infected"] = df["I"].sum()

    # 2. Yearly cases

    # 3. Monthly cases
    targets["monthly_cases"] = df.groupby("month")["I"].sum().values

    # 4. Regional group cases as a single array
    if model_config and "summary_config" in model_config:
        region_groups = model_config["summary_config"].get("region_groups", {})
        regional_cases = []
        for name in region_groups:
            node_list = region_groups[name]
            total = df[df["node"].isin(node_list)]["I"].sum()
            regional_cases.append(total)
        targets["regional_cases"] = np.array(regional_cases)

    print(f"{targets=}")
    return targets


def process_data(filename):
    """Load simulation results and extract features for comparison."""
    df = pd.read_csv(filename)
    return {
        "total_infected": df["I"].sum(),
        "peak_infection_time": df.loc[df["I"].idxmax(), "Time"],
    }
