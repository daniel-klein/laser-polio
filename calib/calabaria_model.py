# calabaria_model.py
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import calabaria as cb
import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import yaml
from laser_core.propertyset import PropertySet

import laser_polio as lp


@dataclass
class LaserPolioConfig:
    study_name: str
    model_config: Path
    pop: int = 100_000
    init_inf: int = 1
    beta: float = 0.05
    verbose: bool = True

    def __post_init__(self):
        with open(self.model_config) as f:
            self.parameters = PropertySet(yaml.safe_load(f))

        if "results_path" in self.parameters:
            self.results_path = Path(self.parameters["results_path"])
        else:
            self.results_path = Path(lp.root / "calib/results" / self.study_name)

        Path(self.results_path).mkdir(parents=True, exist_ok=True)
        return


@dataclass
class LaserPolioModel(cb.BaseModel):
    config: LaserPolioConfig
    pars: cb.ParameterSet

    def extract_actual_data(self):
        # Extract "actual_data" from the h5 file
        config = self.config
        cp = config.parameters.to_dict()

        actual_data = cp.pop("actual_data", lp.root / "data/epi_africa_20250421.h5")
        regions = cp.pop("regions", ["ZAMFARA"])
        dot_names = lp.find_matching_dot_names(regions, lp.root / "data/compiled_cbr_pop_ri_sia_underwt_africa.csv", verbose=config.verbose)
        node_lookup = lp.get_node_lookup(lp.root / "data/node_lookup.json", dot_names)
        start_year = cp.pop("start_year", 2018)
        n_days = cp.pop("n_days", 365)

        epi = lp.get_epi_data(actual_data, dot_names, node_lookup, start_year, n_days)
        epi.rename(columns={"cases": "P"}, inplace=True)
        return epi

    def parameters(self):
        return self.pars

    def build_sim(self, param_set, seed):
        config = self.config

        # Extract simulation setup parameters with defaults or overrides
        cp = config.parameters.to_dict()

        # Extract simulation setup parameters with defaults or overrides
        regions = cp.pop("regions", ["ZAMFARA"])
        start_year = cp.pop("start_year", 2018)
        n_days = cp.pop("n_days", 365)
        pop_scale = cp.pop("pop_scale", 1)
        init_region = cp.pop("init_region", "ANKA")
        init_prev = cp.pop("init_prev", 0.01)
        results_path = cp.pop("results_path", "results/demo")
        actual_data = cp.pop("actual_data", lp.root / "data/epi_africa_20250421.h5")
        # save_plots = cp.pop("save_plots", False)
        # save_data = cp.pop("save_data", False)
        # init_pop_file = cp.pop("init_pop_file", init_pop_file)

        # Geography
        dot_names = lp.find_matching_dot_names(regions, lp.root / "data/compiled_cbr_pop_ri_sia_underwt_africa.csv", verbose=config.verbose)
        node_lookup = lp.get_node_lookup(lp.root / "data/node_lookup.json", dot_names)
        # dist_matrix = lp.get_distance_matrix(lp.root / "data/distance_matrix_africa_adm2.h5", dot_names)
        shp = gpd.read_file(filename=lp.root / "data/shp_africa_low_res.gpkg", layer="adm2")
        shp = shp[shp["dot_name"].isin(dot_names)]
        # Sort the GeoDataFrame by the order of dot_names
        shp.set_index("dot_name", inplace=True)
        shp = shp.loc[dot_names].reset_index()

        # Immunity
        init_immun = pd.read_hdf(lp.root / "data/init_immunity_0.5coverage_january.h5", key="immunity")
        init_immun = init_immun.set_index("dot_name").loc[dot_names]
        init_immun = init_immun[init_immun["period"] == start_year]

        # Initial infection seeding
        init_prevs = np.zeros(len(dot_names))
        prev_indices = [i for i, dot_name in enumerate(dot_names) if init_region in dot_name]
        if not prev_indices:
            raise ValueError(f"No nodes found containing '{init_region}'")
        init_prevs[prev_indices] = init_prev
        # Make dtype match init_prev type
        if isinstance(init_prev, int):
            init_prevs = init_prevs.astype(int)
        if config.verbose >= 2:
            print(f"Seeding infection in {len(prev_indices)} nodes at {init_prev:.3f} prevalence.")

        # SIA schedule
        start_date = lp.date(f"{start_year}-01-01")
        historic = pd.read_csv(lp.root / "data/sia_historic_schedule.csv")
        future = pd.read_csv(lp.root / "data/sia_scenario_1.csv")
        sia_schedule = lp.process_sia_schedule_polio(pd.concat([historic, future]), dot_names, start_date)

        # Demographics and risk
        df_comp = pd.read_csv(lp.root / "data/compiled_cbr_pop_ri_sia_underwt_africa.csv")
        df_comp = df_comp[df_comp["year"] == start_year]
        pop = df_comp.set_index("dot_name").loc[dot_names, "pop_total"].values * pop_scale
        cbr = df_comp.set_index("dot_name").loc[dot_names, "cbr"].values
        ri = df_comp.set_index("dot_name").loc[dot_names, "ri_eff"].values
        sia_re = df_comp.set_index("dot_name").loc[dot_names, "sia_random_effect"].values
        # reff_re = df_comp.set_index("dot_name").loc[dot_names, "reff_random_effect"].values
        # TODO Need to REDO random effect probs since they might've been based on the wrong data. Also, need to do the processing before filtering because of the centering & scaling
        sia_prob = lp.calc_sia_prob_from_rand_eff(sia_re)
        # r0_scalars = lp.calc_r0_scalars_from_rand_eff(reff_re)
        # Calcultate geographic R0 modifiers based on underweight data (one for each node)
        underwt = df_comp.set_index("dot_name").loc[dot_names, "prop_underwt"].values
        r0_scalars = (1 / (1 + np.exp(24 * (0.22 - underwt)))) + 0.2  # The 0.22 is the mean of Nigeria underwt
        # # Check Zamfara means
        # print(f"{underwt[-14:]}")
        # print(f"{r0_scalars[-14:]}")

        # Validate all arrays match
        assert all(len(arr) == len(dot_names) for arr in [shp, init_immun, node_lookup, init_prevs, pop, cbr, ri, sia_prob, r0_scalars])

        # Base parameters (can be overridden)
        base_pars = {
            "start_date": start_date,
            "dur": n_days,
            "n_ppl": pop,
            "age_pyramid_path": lp.root / "data/Nigeria_age_pyramid_2024.csv",
            "cbr": cbr,
            "init_immun": init_immun,
            "init_prev": init_prevs,
            "r0_scalars": r0_scalars,
            "distances": None,
            "shp": shp,
            "node_lookup": node_lookup,
            "vx_prob_ri": ri,
            "sia_schedule": sia_schedule,
            "vx_prob_sia": sia_prob,
            "verbose": self.config.verbose,
            "stop_if_no_cases": False,
        }

        # Dynamic values passed by user/CLI/Optuna
        pars = PropertySet({**base_pars, **param_set.values})

        # Print pars
        # TODO: make this optional
        # sc.pp(pars.to_dict())

        return pars

    def run_sim(self, pars, seed: int):
        sim = lp.SEIR_ABM(pars)
        components = [lp.VitalDynamics_ABM, lp.DiseaseState_ABM, lp.Transmission_ABM]
        if pars.vx_prob_ri is not None:
            components.append(lp.RI_ABM)
        if pars.vx_prob_sia is not None:
            components.append(lp.SIA_ABM)
        sim.components = components

        sim.run()

        # Save results
        # if save_plots:
        #    Path(results_path).mkdir(parents=True, exist_ok=True)
        #    sim.plot(save=True, results_path=results_path)
        # if save_data:
        #    Path(results_path).mkdir(parents=True, exist_ok=True)
        #    lp.save_results_to_csv(sim, filename=results_path / "simulation_results.csv")

        return sim

    @cb.model_output("timeseries")
    def extract_timeseries(self, sim, seed: int) -> pl.DataFrame:
        simulation_results = (
            pl.DataFrame(
                {
                    "timestep": np.tile(range(sim.nt), len(sim.nodes)),
                    "date": np.tile(sim.datevec, len(sim.nodes)).tolist(),
                    "node": np.repeat(sim.nodes, sim.nt),
                    "S": sim.results.S.flatten(order="F"),
                    "E": sim.results.E.flatten(order="F"),
                    "I": sim.results.I.flatten(order="F"),
                    "R": sim.results.R.flatten(order="F"),
                    "P": sim.results.paralyzed.flatten(order="F"),
                    "new_exposed": sim.results.new_exposed.flatten(order="F"),
                }
            )
            .with_columns(pl.lit(seed, dtype=pl.Int64).alias("seed"))
            .with_columns((pl.col("date").dt.month()).alias("month"))
        )
        return simulation_results

    @cb.model_output("total_infected")
    def extract_total_infected(self, sim, seed: int) -> pl.DataFrame:
        # 1. Total infected
        total_infected = pl.DataFrame({"total_infected": sim.results.I.flatten(order="F").sum()}).with_columns(
            pl.lit(seed, dtype=pl.Int64).alias("seed")
        )
        return total_infected

        # 2. Yearly cases

    @cb.model_output("monthly_cases")
    def extract_monthly_cases(self, sim, seed):
        # 3. Monthly cases
        simulation_results = self.extract_timeseries(sim, seed)

        # monthly_cases = pl.DataFrame(
        #    {"monthly_cases": simulation_results[["month", "I"]].group_by("month").sum().sort("month")}
        # ).with_columns(pl.lit(seed, dtype=pl.Int64).alias("seed"))

        monthly_cases = (
            simulation_results.with_columns((pl.col("timestep") // 30 + 1).alias("month"))
            .filter(pl.col("month") <= 12)  # Time is 0 to 365
            .group_by("month")
            .agg(pl.col("I").sum().alias("monthly_cases"))
            .sort("month")
            .select("monthly_cases")  # Select only the "monthly_cases" column
            .transpose(include_header=True, column_names=[str(i) for i in range(1, 13)])
        )

        return monthly_cases

    @cb.model_output("regional_cases")
    def extract_regional_cases(self, sim, seed):
        # 4. Regional group cases as a single array
        simulation_results = self.extract_timeseries(sim, seed)
        if "summary_config" in self.config.parameters:
            rc = {}
            region_groups = self.config.parameters["summary_config"].get("region_groups", {})
            for name in region_groups:
                node_list = region_groups[name]
                total = simulation_results.filter(pl.col("node").is_in(node_list))["I"].sum()
                rc[name] = total
            regional_cases = pl.DataFrame(rc).transpose(include_header=True).with_columns(pl.lit(seed, dtype=pl.Int64).alias("seed"))
        else:
            regional_cases = pl.DataFrame()

        return regional_cases


laser_polio_model_bundle = SimpleNamespace(
    name="laser_polio",
    Config=LaserPolioConfig,
    Model=LaserPolioModel,
)
