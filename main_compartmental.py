from compartmental.Simulation import Simulation

if __name__ == "__main__":
    duration_simulation = 24  # in hours
    time_step = 1 / 60  # in hours

    a_sim = Simulation(duration_simulation, time_step)

    dimers_degradation = d0 = 0.05
    monomers_degradation_in_nucleus = d1 = 0.3
    dimers_production = lamdA = 15
    coefs_dimer_formation_cytoplasm = a1, e1 = 0.001, 20
    coefs_migration_cytoplasm_to_crown = a2, b2, n2, e2 = 400, 0.4, 15, 20
    coefs_migration_crown_to_nucleus = a3, b3, n3, e3, f3 = 8, 0.5, 5, 0.5, 1
    coefs_complex_formation = a4, e4 = 0.0003, 20
    coefs_dimer_formation_crown = a5, b5, n5, e5 = 0.00012, 1, 1, 20
    e6 = 0.5
    coefs_fragmentation = cs, e0 = 0.0002, 20

    initial_compartments_no_crown = (300, 0, 0, 0, 200, 0, 0)
    initial_compartments_crown_formed = (300, 0, 0, 0, 0, 200, 300)

    a_sim.setup_compartmental_system(d0, d1, lamdA, coefs_dimer_formation_cytoplasm, coefs_migration_cytoplasm_to_crown,
                                     coefs_migration_crown_to_nucleus, coefs_complex_formation,
                                     coefs_dimer_formation_crown, e6, coefs_fragmentation,
                                     initial_compartments_no_crown)

    times_antioxidant = ()
    dose_antioxidant = 0.15
    times_irradiation = ()
    dose_irradiation = 15
    times_statin = ()
    dose_statin = 5

    a_sim.setup_experimental_conditions(times_antioxidant, dose_antioxidant, times_irradiation, dose_irradiation,
                                        times_statin, dose_statin)

    a_sim.simulate()

    a_sim.plot_all_compartments()

    # TODO why compartments not changing ?
