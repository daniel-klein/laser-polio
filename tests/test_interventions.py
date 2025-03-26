import numpy as np
from laser_core.propertyset import PropertySet

import laser_polio as lp


# Fixture to set up the simulation environment
def setup_sim(dur=30, n_ppl=None, vx_prob_ri=0.5, cbr=None, r0=14):
    if n_ppl is None:
        n_ppl = [50000, 50000]
    if cbr is None:
        cbr = np.array([30, 25])
    pars = PropertySet(
        {
            "dur": dur,
            "n_ppl": n_ppl,
            "cbr": cbr,  # Birth rate per 1000/year
            "init_immun": 0.0,  # initially immune
            "init_prev": 0.0,  # initially infected from any age
            "r0": r0,  # Basic reproduction number
            "dur_exp": lp.constant(value=2),  # Duration of the exposed state
            "dur_inf": lp.constant(value=1),  # Duration of the infectious state
            "vx_prob_ri": vx_prob_ri,  # Routine immunization probability
            "sia_schedule": [{"date": "2020-01-10", "nodes": [0], "age_range": (180, 365), "coverage": 0.6}],
            "sia_eff": [0.6, 0.8],  # SIA effectiveness per node
        }
    )
    sim = lp.SEIR_ABM(pars)
    sim.components = [lp.VitalDynamics_ABM, lp.DiseaseState_ABM, lp.RI_ABM, lp.SIA_ABM, lp.Transmission_ABM]
    return sim


# --- RI_ABM Tests ---


def test_ri_initialization():
    """Ensure that RI_ABM initializes correctly."""
    sim = setup_sim()
    sim.run()
    assert hasattr(sim.people, "ri_timer")
    assert hasattr(sim.results, "ri_vaccinated")
    assert hasattr(sim.results, "ri_protected")


def test_ri_manually_seeded():
    """Ensure that routine immunization occurs when manually seeded."""
    n_vx = 1000
    dur = 28
    sim = setup_sim(dur=dur, vx_prob_ri=1.0)
    sim.people.ri_timer[:n_vx] = np.random.randint(0, dur, n_vx)  # Set timers to trigger vaccination
    sim.run()
    assert sim.results.ri_vaccinated.sum() >= n_vx, "The number of vaccinations was lower than the number manually seeded."
    assert sim.results.ri_protected.sum() >= n_vx, "The number of vaccinations was lower than the number manually seeded."


def test_ri_on_births():
    dur = 365
    cbr = np.array([300, 250])
    sim = setup_sim(dur=dur, cbr=cbr, vx_prob_ri=1.0)
    sim.run()
    assert np.sum(sim.results.ri_vaccinated) > 0, "No routine immunizations occurred on births."
    assert np.sum(sim.results.ri_protected) > 0, "No routine immunizations occurred on births."


def test_ri_zero():
    dur = 365

    # Test RI when there are no births (there can still be some RI in existing population)
    cbr = np.array([0, 0])
    vx_prob_ri = 1.0
    sim_no_births = setup_sim(dur=dur, cbr=cbr, vx_prob_ri=vx_prob_ri)
    sim_no_births.run()
    assert np.sum(sim_no_births.results.ri_vaccinated[(98 + 14) :]) == 0, (
        "No RI vaccinations should've occurred after initial cohort aged out of RI (oldest 98 days + time_step)."
    )
    assert np.sum(sim_no_births.results.ri_protected[(98 + 14) :]) == 0, (
        "No RI vaccinations should've occurred after initial cohort aged out of RI."
    )

    # Zero routine immunization probability
    cbr = np.array([300, 250])
    vx_prob_ri = 0.0
    sim_zero_ri_prob = setup_sim(dur=dur, cbr=cbr, vx_prob_ri=vx_prob_ri)
    sim_zero_ri_prob.run()
    assert np.sum(sim_zero_ri_prob.results.ri_vaccinated) == 0, "RI vaccinations occurred, but there should've been zero."
    assert np.sum(sim_zero_ri_prob.results.ri_protected) == 0, "RI vaccinations occurred, but there should've been zero."


def test_ri_vx_prob():
    """Ensure that the vaccination probability is respected when no births are scheduled."""
    n_ppl = np.array([20, 20])
    n_vx = np.sum(n_ppl)
    dur = 28
    vx_prob_ri = 0.65
    sim = setup_sim(n_ppl=n_ppl, dur=dur, vx_prob_ri=vx_prob_ri, cbr=np.array([0, 0]))
    sim.people.ri_timer[:n_vx] = np.random.randint(0, dur, n_vx)  # Set timers to trigger vaccination

    print(sim.people.disease_state)

    sim.run()

    n_exp = n_vx * vx_prob_ri
    n_vx = np.sum(sim.results.ri_vaccinated)
    n_protected = np.sum(sim.results.ri_protected)
    n_r = np.sum(sim.results.R[-1])

    print(sim.people.disease_state)

    assert np.isclose(n_exp, n_vx, atol=75), "Vaccination rate does not match probability."
    assert n_vx == n_protected == n_r, "Vaccinated, protected, and Recovered counts should be equal if vx efficacy is 100%"


def test_ri_no_effect_on_non_susceptibles():
    """Ensure RI does not affect infected or recovered individuals."""
    n_ppl = np.array([10, 10])
    r0 = 0
    vx_prob_ri = 1.0
    sim = setup_sim(n_ppl=n_ppl, r0=r0, vx_prob_ri=vx_prob_ri)
    sim.people.ri_timer[:20] = 0
    sim.people.disease_state[:5] = 1  # Exposed
    sim.people.disease_state[5:10] = 2  # Infected
    sim.people.disease_state[10:15] = 3  # Recovered
    sim.run()
    assert np.sum(sim.results.ri_vaccinated) == np.sum(sim.results.R[-1]) == 20, "All individuals should've been vaccinated."
    assert np.sum(sim.results.ri_protected) == 5, "Only the 5 susceptible individuals should've been protected."


# --- SIA_ABM Tests ---


def test_sia_initialization():
    """Ensure that SIA_ABM initializes correctly."""
    sim = setup_sim()
    sim.run()
    assert hasattr(sim.results, "sia_vx")


def test_sia_execution_on_scheduled_date():
    """Ensure that SIA occurs on the correct date."""
    sim = setup_sim()
    sim.run()
    sim.t = 9  # Set to one day before the scheduled date
    sim.step()
    assert np.all(sim.results.sia_vx[9, :] == 0), "SIA should not run before scheduled date."
    sim.step()  # Move to scheduled date
    assert np.any(sim.results.sia_vx[10, :] > 0), "SIA did not execute on the scheduled date."


def test_sia_age_based_vaccination():
    """Ensure that only eligible age groups receive SIA vaccination."""
    sim = setup_sim()
    sim.run()
    sim.people.date_of_birth[:10] = -300  # 300 days old (should be vaccinated)
    sim.people.date_of_birth[10:20] = -500  # 500 days old (should not be vaccinated)
    sim.people.disease_state[:20] = 0  # All susceptible
    sim.t = 10  # Move to scheduled date
    sim.step()
    vaccinated = np.sum(sim.people.disease_state[:10] == 3)
    not_vaccinated = np.sum(sim.people.disease_state[10:20] == 3)
    assert vaccinated > 0, "Eligible individuals were not vaccinated."
    assert not_vaccinated == 0, "Ineligible individuals were vaccinated."


def test_sia_node_based_targeting():
    """Ensure that only targeted nodes receive SIA vaccination."""
    sim = setup_sim()
    sim.run()
    sim.people.node_id[:10] = 0  # Targeted node
    sim.people.node_id[10:20] = 1  # Untargeted node
    sim.people.date_of_birth[:20] = -300  # All eligible age-wise
    sim.people.disease_state[:20] = 0  # All susceptible
    sim.t = 10  # Move to scheduled date
    sim.step()
    vaccinated_targeted = np.sum(sim.people.disease_state[:10] == 3)
    vaccinated_untargeted = np.sum(sim.people.disease_state[10:20] == 3)
    assert vaccinated_targeted > 0, "Targeted nodes did not receive vaccination."
    assert vaccinated_untargeted == 0, "Untargeted nodes received vaccination."


def test_sia_coverage_probability():
    """Ensure SIA vaccination occurs at the expected probability."""
    sim = setup_sim()
    sim.run()
    sim.people.node_id[:100] = 0  # All in the targeted node
    sim.people.date_of_birth[:100] = -300  # All eligible age-wise
    sim.people.disease_state[:100] = 0  # All susceptible
    sim.t = 10  # Move to scheduled date
    sim.step()
    vaccinated = np.sum(sim.people.disease_state[:100] == 3)
    expected_vx_rate = sim.pars.sia_eff[0]
    assert np.isclose(vaccinated / 100, expected_vx_rate, atol=0.1), "SIA coverage rate does not match expected probability."


if __name__ == "__main__":
    # test_ri_initialization()
    # test_ri_manually_seeded()
    # test_ri_on_births()
    # test_ri_zero()
    test_ri_vx_prob()
    test_ri_no_effect_on_non_susceptibles()
    # test_sia_initialization()
    # test_sia_execution_on_scheduled_date()
    # test_sia_age_based_vaccination()
    # test_sia_node_based_targeting()
    # test_sia_coverage_probability()

    print("All initialization tests passed.")
