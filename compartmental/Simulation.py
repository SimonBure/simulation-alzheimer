import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
from spatial.oneD.OneDimSpace import TimeSpace
from compartmental.System import CompartmentalSystem
from spatial.oneD.Experiment import Antioxidant, Irradiation, Statin
from compartmental.Rate import (ConstantRate, DimerFormationRateCrown, DimerFormationRateCytoplasm,
                                MonomerDispersionRate, ComplexesFormationRate, MigrationRateCytoplasmToPc,
                                MigrationRatePcToNucleus, FragmentationRate)
from compartmental.Dose import AntioxidantDose, StatinDose, IrradiationDose


class Simulation:
    time: float
    time_index: int
    time_space: TimeSpace

    compartmental_system: CompartmentalSystem

    experiments: tuple[Antioxidant, Irradiation, Statin]

    crown_formation_speed: ndarray

    def __init__(self, maximum_time: float, time_step: float):
        self.time = 0
        self.time_index = 0
        self.time_space = TimeSpace(maximum_time, int(maximum_time / time_step))

        self.compartmental_system = CompartmentalSystem(self.time_space)

        self.crown_formation_speed = np.zeros(self.time_space.nb_points)

    def setup_compartmental_system(self, dimers_degradation: float, monomers_degradation_nucleus: float,
                                   dimers_production: float, coefs_dimer_formation_cyto: tuple[float, float],
                                   coefs_migration_cyto_pc: tuple[float, float, float, float],
                                   coefs_migration_pc_nucleus: tuple[float, float, float, float, float],
                                   coefs_complex_formation: tuple[float, float],
                                   coefs_dimer_formation_crown: tuple[float, float, float, float], e6: float,
                                   coefs_fragmentation: tuple[float, float],
                                   initial_conditions: tuple[float, float, float, float, float, float, float]):
        a1, e1 = coefs_dimer_formation_cyto
        a2, b2, n2, e2 = coefs_migration_cyto_pc
        a3, b3, n3, e3, f3 = coefs_migration_pc_nucleus
        a4, e4 = coefs_complex_formation
        a5, b5, n5, e5 = coefs_dimer_formation_crown
        cs, e0 = coefs_fragmentation

        dc0, mc0, ma0, mn0, a0, ca0, da0 = initial_conditions

        d0 = ConstantRate("d_0", dimers_degradation)
        d1 = ConstantRate("d1", monomers_degradation_nucleus)
        lam = ConstantRate("λ", dimers_production)
        k1 = DimerFormationRateCytoplasm("k_1", a1, e1)
        k2 = MigrationRateCytoplasmToPc("k_2", a2, b2, n2, e2)
        k3 = MigrationRatePcToNucleus("k_2", a3, b3, n3, e3, f3)
        k4 = ComplexesFormationRate("k_4", a4, e4)
        k5 = DimerFormationRateCrown("k_5", a5, b5, n5, e5)
        k6 = MonomerDispersionRate("k_6", e6)
        s = FragmentationRate("s", cs, e0)

        self.compartmental_system.setup_rates(lam, d0, d1, k1, k2, k3, k4, k5, k6, s)
        self.compartmental_system.setup_initial_conditions(dc0, mc0, ma0, mn0, a0, ca0, da0)

    def setup_experimental_conditions(self, antioxidant_start_and_ending_times: tuple | tuple[float, float] | tuple[tuple[float, float], ...],
                                      dose_antioxidant: float,
                                      irradiation_start_and_ending_times: tuple | tuple[float, float] | tuple[tuple[float, float], ...],
                                      dose_irradiation: float,
                                      statin_start_and_ending_times: tuple | tuple[float, float] | tuple[tuple[float, float], ...],
                                      dose_statin: float):
        antioxidant_exp = Antioxidant(antioxidant_start_and_ending_times)
        irradiation_exp = Irradiation(irradiation_start_and_ending_times)
        statin_exp = Statin(statin_start_and_ending_times)
        self.experiments = (antioxidant_exp, irradiation_exp, statin_exp)

        antioxidant_dose = AntioxidantDose(dose_antioxidant, self.time_space, antioxidant_exp)
        antioxidant_dose.setup_dose_over_time(self.time_space, antioxidant_exp)

        irradiation_dose = IrradiationDose(dose_irradiation, self.time_space, irradiation_exp)
        irradiation_dose.setup_dose_over_time(self.time_space, irradiation_exp)

        statin_dose = StatinDose(dose_statin, self.time_space, statin_exp)
        statin_dose.setup_dose_over_time(self.time_space, statin_exp)

        self.compartmental_system.setup_doses(antioxidant_dose, irradiation_dose, statin_dose)

    def simulate(self):
        while self.time < self.time_space.end:
            self.compartmental_system.update_rates(self.time)

            self.compartmental_system.set_next_compartments_values()

            self.fill_crown_speed_formation()

            self.compartmental_system.update_compartments()

            self.compartmental_system.fill_time_values(self.time_index)

            self.time += self.time_space.step
            self.time_index += 1

    def fill_crown_speed_formation(self):
        self.crown_formation_speed[self.time_index] = (self.compartmental_system.dimers_crown.next_value
                                                       - self.compartmental_system.dimers_crown.actual_value)

    def plot_all_compartments(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        print(type(ax1))

        self.plot_compartments_cytoplasm_and_nucleus(ax1)
        self.plot_compartments_crown(ax2)

        plt.subplots_adjust(wspace=0.4)  # setting horizontal space between the two plots

        plt.show()

    def plot_compartments_cytoplasm_and_nucleus(self, ax: plt.Axes):
        dimers_cyto, monomers_cyto, monomers_nucleus = (
            self.compartmental_system.get_compartments_cytoplasm_and_nucleus_time_values())

        ax.plot(self.time_space.space, dimers_cyto, label="$D_C$", color="blue")
        ax.plot(self.time_space.space, monomers_cyto, label='$M_C$', color='green')
        ax.plot(self.time_space.space, monomers_nucleus, label='$M_N$', color='red')
        print(ax)
        ax.set_title("Populations evolution in the cytoplasm and nucleus")

        self.label_axis_compartments(ax)

        self.highlight_time_zones_with_experiment(ax)

        self.caption_plot(ax)

    @staticmethod
    def label_axis_compartments(ax: plt.Axes):
        Simulation.label_time_axis(ax)
        ax.set_ylabel('Populations', fontsize=12)

    @staticmethod
    def label_time_axis(ax: plt.Axes):
        ax.set_xlabel('Time ($h$)', fontsize=12)

    def highlight_time_zones_with_experiment(self, ax: plt.Axes):
        for aox_starting_ending_times in self.experiments[1].time_experiments:
            ax.axvspan(aox_starting_ending_times[0], aox_starting_ending_times[1], color='green', alpha=0.3,
                       label='Antioxidant')
        for irr_starting_ending_times in self.experiments[1].time_experiments:
            ax.axvspan(irr_starting_ending_times[0], irr_starting_ending_times[1], color='red', alpha=0.3,
                       label='Irradiation')
        for statin_starting_ending_times in self.experiments[2].time_experiments:
            ax.axvspan(statin_starting_ending_times[0], statin_starting_ending_times[1], color='yellow', alpha=0.3,
                       label='Statin')

    @staticmethod
    def caption_plot(ax: plt.Axes):
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

    def plot_compartments_crown(self, ax: plt.Axes):
        monomers, apoe, complexes, dimers = self.compartmental_system.get_compartments_crown_time_values()

        ax.plot(self.time_space.space, monomers, label='$M_A$', color='dodgerblue')
        ax.plot(self.time_space.space, apoe, label='$A$', color='magenta')
        ax.plot(self.time_space.space, complexes, label='$C_A$', color='darkorange')
        ax.plot(self.time_space.space, dimers, label='$D_A$', color='crimson')

        # Titles & labels
        ax.set_title("Populations evolution in the perinuclear crown")

        self.label_axis_compartments(ax)

        self.highlight_time_zones_with_experiment(ax)

        self.caption_plot(ax)

    def plot_crown_formation_speed_along_time(self):
        fig, ax = plt.subplots()
        ax.plot(self.time_space.space, self.crown_formation_speed, color='blue')

        plt.title("Evolution of crown formation speed along time")
        self.label_time_axis(ax)
        self.label_crown_formation_speed_axis(ax)

    def plot_crown_formation_speed_along_rate(self, rate_values_over_time: ndarray, rate_label: str):
        fig, ax = plt.subplots()
        ax.plot(rate_values_over_time, self.crown_formation_speed, color='blue')

        ax.xlabel(rate_label + "($h^{-1}$)")
        self.label_crown_formation_speed_axis(ax)

    @staticmethod
    def label_crown_formation_speed_axis(ax: plt.Axes):
        ax.ylabel("Crown formation speed ($h^{-1}$)")
