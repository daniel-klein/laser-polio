# Priorities
- Add vx efficacy by type
- Test SIA_abm
- Check the RI & SIA figures - they're plotting strange results
- Merge my interventions branch

- Get R0 input file from Hil
- Check transmission probability with real data. Why do we need R0 so high!?
- Test full models with real data
- Drop ABM term from components
- Watch for JB to merge branch to main
- Try running calibration by myself - see the docs
- Export pars as pkl
- Rename variables to distinguish between exposure and infection
- Enable vx transmission
- Set a random number seed
- Use KM's gravity model scaling approach
- Update the birth and death plot to summarize by country.
- Plot expected births? 
- Calibration
- Add step size to components (e.g., like vital dynamics)
- Save results & specify frequency
- Reactive SIAs

# Refinement

- Count number of exportations for calibration
- Enable different RI rates over time
- Do we need sub-adm2 resolution? And if so, how do we handle the distance matrix to minimize file size? Consider making values nan if over some threshold?
- Add EMOD style seasonality
- Fork polio-immunity-mapping repo
- Double check that I'm using the ri_eff and sia_prob values correctly - do I need to multiply sia_prob by vx_eff?
- Get total pop data, not just <5
- Investigate extra dot_names in the pop dataset
- Look into age-specific death rates
- Import/seed infections throughout the sim after OPV use?
- Write pars to disk
- Add partial susceptibility & paralysis protection
- Add distributions for duration of each state
- Add in default pars and allow user pars to overwrite them
- Add CBR by country-year
- Add age pyramid by country
- In post(?), resample I count to get a variety of paralysis counts
