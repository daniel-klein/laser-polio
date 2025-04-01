import json
from pathlib import Path

import numpy as np
import pandas as pd

import laser_polio as lp

start_year = 2019
regions = ["NIGERIA"]


def inv_logit(x):
    return 1 / (1 + np.exp(-x))


# Find the dot_names matching the specified string(s)
dot_names = lp.find_matching_dot_names(regions, lp.root / "data/compiled_cbr_pop_ri_sia_underwt_africa.csv")

# Load the NEW curated random effects
df_re = pd.read_csv("data/curation_scripts/random_effects/random_effects_curated.csv")
# Filter to adm0_name that == "NIGERIA"
df_re = df_re[df_re["adm0_name"].isin(regions)]
# Extract the Reff values
reff_re = df_re["reff_random_effect"].values  # R random effects from regression model

# Load the OLD Nigeria R scalars from EMOD
json_path = Path("data/curation_scripts/random_effects/r0_NGA_mult_2025Feb.json")
with json_path.open("r") as f:
    emod_scalars_dict = json.load(f)
emod_scalars = np.array(list(emod_scalars_dict.values()))


# Set bounds on R0
R0 = 14
R_m = 5 / R0  # Min R0. Divide by R0 since we ultimately want scalars on R0 (e.g., beta_spatial)
R_M = 20 / R0  # Max R0. Divide by R0 since we ultimately want scalars on R0 (e.g., beta_spatial)

# Compute scale and center values for the old and new datasets
# The emod_scalars are scalars on R0, the reff_re are random effects from the regression model
old_scale = np.std(emod_scalars)  # sd from data file for Nigeria = 0.367
old_center = np.mean(emod_scalars)  # mean from data for Nigeria (or median) = 0.646
new_scale = np.std(reff_re)  # Nigeria's R0 new scale Kurt used (not from TSIR) = 0.589
new_center = np.mean(reff_re)  # Nigeria's R0 "center" value = 0.836

# Option 1:
# new_R0 = exp(new_scale * (random_effect - old_center) / old_scale + new_center)
# This has some flaws. First, the median < mean with something exponentiated so the mean after doing exp() could look strange. Second, as you suggest you technically have an unbounded R0.

# Option 2: you might consider limits to how low or high R0 can go, e.g. max(min(R0, 15), 5), or a logit instead of log that uses these (or other) bounds.  This is not a smooth function, but:
w = inv_logit(new_scale * (reff_re - old_center) / old_scale)
R_c = np.exp(new_center)  # Nigeria central R0 scalar = 2.31

new_reff_scalars = R_c + (R_M - R_c) * np.maximum(w - 0.5, 0) + (R_c - R_m) * np.minimum(w - 0.5, 0) * 2
# With the idea that at w = 0.5 we are at the Nigeria center, as w -> 1, we get to the bound R_M, and as w ->0, we go to R_m.
mean_r0 = np.mean(new_reff_scalars) * R0
min_r0 = np.min(new_reff_scalars) * R0
max_r0 = np.max(new_reff_scalars) * R0
print(f"R0: {mean_r0:.2f} ({min_r0:.2f}, {max_r0:.2f})")  # R0: 27.54 (18.01, 31.88)


# # Load the shapefile
# shp = gpd.read_file("data/shp_africa_low_res.gpkg", layer="adm2")


# print("Done.")


# # Initialize lists to store results
# mean_r0_spatial_values = []
# min_r0_spatial_values = []
# max_r0_spatial_values = []

# # Loop over m values
# for m in m_values:
#     # Calculate r0_spatial  for the current m
#     r0_spatial = np.exp(m * (reff_re - np.mean(reff_re)) / np.std(reff_re) + np.log(R0))

#     # Compute statistics
#     mean_r0_spatial_values.append(np.mean(r0_spatial))
#     min_r0_spatial_values.append(np.min(r0_spatial))
#     max_r0_spatial_values.append(np.max(r0_spatial))

# # Plot the results
# plt.figure(figsize=(10, 6))
# plt.plot(m_values, mean_r0_spatial_values, label="Mean Reff", color="blue")
# plt.plot(m_values, min_r0_spatial_values, label="Min Reff", color="green")
# plt.plot(m_values, max_r0_spatial_values, label="Max Reff", color="red")
# plt.xlabel("m")
# plt.ylabel("Reff")
# plt.title("Mean, Min, and Max of Reff Across m")
# plt.legend()
# plt.grid()
# plt.show()


# # Initialize lists to store results
# mean_r0_spatial_values = []
# min_r0_spatial_values = []
# max_r0_spatial_values = []

# # Loop over m values
# for m in m_values:
#     # Calculate r0_spatial  for the current m
#     r0_spatial = np.exp(m * (reff_re - np.mean(reff_re)) / np.std(reff_re) + np.log(R0))

#     # Compute statistics
#     mean_r0_spatial_values.append(np.mean(r0_spatial))
#     min_r0_spatial_values.append(np.min(r0_spatial))
#     max_r0_spatial_values.append(np.max(r0_spatial))

# # Plot the results
# plt.figure(figsize=(10, 6))
# plt.plot(m_values, mean_r0_spatial_values, label="Mean Reff", color="blue")
# plt.plot(m_values, min_r0_spatial_values, label="Min Reff", color="green")
# plt.plot(m_values, max_r0_spatial_values, label="Max Reff", color="red")
# plt.xlabel("m")
# plt.ylabel("Reff")
# plt.title("Mean, Min, and Max of Reff Across m")
# plt.legend()
# plt.grid()
# plt.show()


# # Load the shapefile
# shp = gpd.read_file("data/shp_africa_low_res.gpkg", layer="adm2")


# print("Done.")
