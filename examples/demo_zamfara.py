import sciris as sc

import laser_polio as lp

"""
This script contains a demo simulation of polio transmission in Nigeria.

The model uses the same data and setup as the EMOD model, except in the following instances:
- The model assumes everyone >15y is immune
- The total population counts are being estimated by scaling up u5 population counts based on their proportion of the population
- I'm using a sinusoidal seasonality function rather than a step function
- The nodes are not divided below the adm2 level (with no plans to do so)
- There is no scaling of transmission between N & S Nigeria (other than underweight fraction)
- We do not update the cbr, ri, sia, or underwt data over time
- Vaccines are not allowed to transmit
"""

###################################
######### USER PARAMETERS #########

regions = ["ZAMFARA"]
start_year = 2019
n_days = 365
pop_scale = 1 / 100
init_region = "ANKA"
init_prev = 0.001
r0 = 14
results_path = "results/demo_zamfara"

######### END OF USER PARS ########
###################################


sim = lp.setup_sim(
    regions=regions,
    start_year=start_year,
    n_days=n_days,
    pop_scale=pop_scale,
    init_region=init_region,
    init_prev=init_prev,
    r0=r0,
    results_path=results_path,
    verbose=2,
)

sc.printcyan("Done.")
