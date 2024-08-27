import numpy as np
from numpy import ndarray
import scipy.sparse as sparse
import math
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from scipy.sparse.linalg import factorized
import gc
from spatial.oneD.OneDimSpace import SpatialSpace, TimeSpace
from spatial.oneD.Experiment import Antioxidant, Irradiation, Statin
from spatial.oneD.Parameter import DiffusionParameter, TransportParameter, FragmentationParameter, PermeabilityParameter
from spatial.oneD.System import ReactionDiffusionAtmApoeSystem


class Simulation1D:
    time: float
    time_index: int

    spatial_space: SpatialSpace
    time_space: TimeSpace

    atm_apoe_system: ReactionDiffusionAtmApoeSystem
    crown_density_over_time: ndarray

    masses_atm: ndarray

    atm_flux_nucleus: ndarray

    experiments: tuple[Antioxidant, Irradiation, Statin]

    def __init__(self, space_length: float, nb_space_points: int, maximum_time: float, time_step: float):
        self.time = 0.
        self.time_index = 0

        self.spatial_space = SpatialSpace(space_length, nb_space_points)

        nb_time_points = math.ceil(maximum_time / time_step)
        self.time_space = TimeSpace(maximum_time, nb_time_points)

        self.crown_density_over_time = np.zeros(nb_time_points)
        self.masses_atm = np.zeros(nb_time_points)
        self.atm_flux_nucleus = np.zeros(nb_time_points)

        self.atm_apoe_system = ReactionDiffusionAtmApoeSystem()
        self.atm_apoe_system.setup_spaces(self.spatial_space, self.time_space)

    def setup_atm_apoe_system(self, k: float, ka: float, diffusion_coefs: tuple[float, float],
                              transport_coefs: tuple[float, float], fragmentation_coefs: tuple[float, float, float],
                              permeability_coefs: tuple[float, float, float], ratio_fragm_dimer_complexes: float,
                              transport_space_constant: float):
        diffusion_param = DiffusionParameter(diffusion_coefs[0], diffusion_coefs[1])
        transport_param = TransportParameter(transport_coefs[0], transport_coefs[1])
        fragmentation_param = FragmentationParameter(fragmentation_coefs[0], fragmentation_coefs[1],
                                                     fragmentation_coefs[2])
        permeability_param = PermeabilityParameter(permeability_coefs[0], permeability_coefs[1], permeability_coefs[2])

        self.atm_apoe_system.setup_parameters(k, ka, diffusion_param, transport_param, fragmentation_param,
                                              permeability_param, ratio_fragm_dimer_complexes, transport_space_constant)

    def setup_system_initial_conditions(self, monomers_initial: ndarray, dimers_initial: ndarray,
                                        apoe_initial: ndarray, complexes_initial: ndarray):
        self.atm_apoe_system.setup_populations(monomers_initial, dimers_initial, apoe_initial,
                                               complexes_initial)

    def setup_experimental_conditions(self, antioxidant_start_and_ending_time: tuple | tuple[float, float] | tuple[tuple[float, float], ...],
                                      irradiation_start_and_ending_time: tuple | tuple[float, float] | tuple[tuple[float, float], ...],
                                      statin_start_and_ending_time: tuple | tuple[float, float] | tuple[tuple[float, float], ...]):
        antioxidant = Antioxidant(antioxidant_start_and_ending_time)
        irradiation = Irradiation(irradiation_start_and_ending_time)
        statin = Statin(statin_start_and_ending_time)
        self.experiments = antioxidant, irradiation, statin
        self.atm_apoe_system.setup_experiments_impact_on_parameters(antioxidant, irradiation, statin)

    def simulate(self):
        (solver_natural, solver_during_antioxidant, solver_during_irradiation,
         solver_during_antioxidant_and_irradiation) = self.create_all_solvers()

        while self.time_index < self.time_space.nb_points:
            self.atm_apoe_system.update_for_next_step()

            dimers_next_density = self.atm_apoe_system.compute_dimers_next_density(self.time_index)
            apoe_next_density = self.atm_apoe_system.compute_apoe_next_density(self.time_index)
            complexes_next_density = self.atm_apoe_system.compute_complexes_next_density(self.time_index)

            flux_in_nucleus = self.compute_monomers_flux_in_nucleus(dimers_next_density, apoe_next_density,
                                                                    complexes_next_density)
            self.fill_flux_nucleus(flux_in_nucleus)

            if self.is_migration_inside_nucleus_possible(flux_in_nucleus):
                right_solver = self.create_monomers_flux_solver(flux_in_nucleus)
            else:
                if self.is_antioxidant_now():
                    if self.is_irradiation_now():
                        right_solver = solver_during_antioxidant_and_irradiation
                    else:
                        right_solver = solver_during_antioxidant
                else:
                    if self.is_irradiation_now():
                        right_solver = solver_during_irradiation
                    else:
                        right_solver = solver_natural

            monomers_next_density = self.compute_monomers_next_density(right_solver)

            atm_crown = self.atm_apoe_system.compute_perinuclear_crown(dimers_next_density, apoe_next_density,
                                                                       complexes_next_density)
            self.fill_crown_density(atm_crown)

            mass_atm = self.atm_apoe_system.compute_atm_mass()
            self.fill_atm_mass(mass_atm)

            mass_atm_lost = self.compute_atm_loss()
            dimers_next_density += self.produce_new_dimers(mass_atm_lost)

            self.atm_apoe_system.set_next_values(monomers_next_density, dimers_next_density, apoe_next_density,
                                                 complexes_next_density)

            self.atm_apoe_system.fill_every_time_values(self.time_index)

            self.time_index += 1
            self.time += self.time_space.step

    def create_all_solvers(self) -> tuple[factorized, factorized, factorized, factorized]:
        solver_natural = self.create_natural_system_solver()
        solver_during_antioxidant = self.create_antioxidant_system_solver()
        solver_during_irradiation = self.create_irradiation_system_solver()
        solver_during_antioxidant_and_irradiation = self.create_antioxidant_irradiation_system_solver()
        return (solver_natural, solver_during_antioxidant, solver_during_irradiation,
                solver_during_antioxidant_and_irradiation)

    def create_natural_system_solver(self) -> factorized:
        diagonals = self.create_diagonals(self.atm_apoe_system.diffusion_parameter.natural_value,
                                          self.atm_apoe_system.transport_parameter.natural_value, 0)
        return self.create_solver_from_diagonals(diagonals)

    def create_antioxidant_system_solver(self) -> factorized:
        diagonals = self.create_diagonals(self.atm_apoe_system.diffusion_parameter.antioxidant_value,
                                          self.atm_apoe_system.transport_parameter.antioxidant_value, 0)
        return self.create_solver_from_diagonals(diagonals)

    def create_irradiation_system_solver(self) -> factorized:
        diagonals = self.create_diagonals(self.atm_apoe_system.diffusion_parameter.natural_value,
                                          self.atm_apoe_system.transport_parameter.irradiation_value, 0)
        return self.create_solver_from_diagonals(diagonals)

    def create_antioxidant_irradiation_system_solver(self) -> factorized:
        diagonals = self.create_diagonals(self.atm_apoe_system.diffusion_parameter.antioxidant_value,
                                          self.atm_apoe_system.transport_parameter.irradiation_value, 0)
        return self.create_solver_from_diagonals(diagonals)

    def create_monomers_flux_solver(self, flux_in_nucleus: float) -> factorized:
        diffusion_coefficient = self.atm_apoe_system.diffusion_parameter.get_value_at_specific_time(self.time_index)
        transport_coefficient = self.atm_apoe_system.transport_parameter.get_value_at_specific_time(self.time_index)
        diagonals = self.create_diagonals(diffusion_coefficient, transport_coefficient, flux_in_nucleus)
        return self.create_solver_from_diagonals(diagonals)

    def create_solver_from_diagonals(self, diagonals: tuple[ndarray, ndarray, ndarray]) -> factorized:
        sparse_system_matrix = self.create_sparse_system_matrix_from_diagonals(diagonals)
        return factorized(sparse_system_matrix)

    def create_diagonals(self, diffusion: float, transport: float, flux_in_nucleus) -> tuple[ndarray, ndarray, ndarray]:
        diagonal, lower, upper = self.setup_diagonals()

        f = self.compute_fourier_number()
        p = self.compute_peclet_number()

        transport_over_space = transport * self.atm_apoe_system.transport_parameter.over_space_values

        diagonal[:] = 1 + 2 * diffusion * f + transport_over_space * p
        lower[:] = - f * diffusion
        upper[:] = - f * diffusion - transport_over_space[1:] * p

        # Robin-Neumann boundary conditions
        diagonal[0] = 1 + self.spatial_space.step * flux_in_nucleus / diffusion
        upper[0] = - (1 + self.spatial_space.step * transport / diffusion)
        diagonal[-1] = -1
        lower[-1] = (1 - self.spatial_space.step * transport_over_space[-1] / diffusion)

        return diagonal, lower, upper

    def setup_diagonals(self) -> tuple[ndarray, ndarray, ndarray]:
        diagonal = np.zeros(self.spatial_space.nb_points)
        lower = np.zeros(self.spatial_space.nb_points - 1)
        upper = np.zeros(self.spatial_space.nb_points - 1)
        return diagonal, lower, upper

    def compute_fourier_number(self) -> float:
        return self.time_space.step / self.spatial_space.step ** 2

    def compute_peclet_number(self) -> float:
        return self.time_space.step / self.spatial_space.step

    def compute_monomers_flux_in_nucleus(self, dimers_next: ndarray, apoe_next: ndarray,
                                         complexes_next: ndarray) -> float:
        return float(self.atm_apoe_system.monomers.actual_values[0] *
                     self.atm_apoe_system.compute_nucleus_permeability(self.time_index, dimers_next, apoe_next,
                                                                       complexes_next)
                     )

    def is_simulation_started(self) -> bool:
        return self.time > 0

    def create_sparse_system_matrix_from_diagonals(self, diagonals: tuple[ndarray, ndarray, ndarray]) -> (
            sparse.csc_array):
        system_matrix = sparse.diags_array(diagonals, offsets=[0, -1, 1],
                                           shape=(self.spatial_space.nb_points, self.spatial_space.nb_points),
                                           format='csc')
        return system_matrix

    def is_antioxidant_now(self) -> bool:
        return self.is_experiment_now(0)

    def is_irradiation_now(self) -> bool:
        return self.is_experiment_now(1)

    def is_statin_now(self) -> bool:
        return self.is_experiment_now(2)

    def is_experiment_now(self, index_experiment: int) -> bool:
        starts, ends = self.experiments[index_experiment].get_starting_and_ending_times()
        for s, e in zip(starts, ends):
            if s < self.time < e:
                return True
        return False

    @staticmethod
    def is_migration_inside_nucleus_possible(flux: float) -> bool:
        return flux > 0

    def compute_monomers_next_density(self, solver: factorized) -> ndarray:
        monomers_reaction = self.atm_apoe_system.create_monomers_reaction_array(self.time_index)
        next_values = solver(self.atm_apoe_system.monomers.actual_values + self.time_space.step * monomers_reaction)
        return next_values

    def compute_atm_loss(self) -> float:
        return float(self.masses_atm[self.time_index - 1] - self.masses_atm[self.time_index])

    def produce_new_dimers(self, mass_atm_lost: float) -> float:
        return self.spatial_space.step * mass_atm_lost / self.spatial_space.end

    def fill_crown_density(self, crown_density: float):
        self.crown_density_over_time[self.time_index] = crown_density

    def fill_flux_nucleus(self, flux_in_nucleus: float):
        self.atm_flux_nucleus[self.time_index] = flux_in_nucleus

    def fill_atm_mass(self, atm_mass: float):
        self.masses_atm[self.time_index] = atm_mass

    def plot_all_densities(self):
        fig, ax = plt.subplots()

        self.plot_monomers_density_over_space(ax)
        self.plot_dimers_density_over_space(ax)
        self.plot_apoe_density_over_space(ax)
        self.plot_complexes_density_over_space(ax)

        self.label_x_space_axis(ax)
        ax.set_ylabel("Densities", fontsize=12)

        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

        plt.show()

    def plot_monomers_over_space(self):
        fig, ax = plt.subplots()

        self.plot_monomers_density_over_space(ax)

        self.label_x_space_axis(ax)
        self.label_y_monomers_density(ax)

        plt.show()

    def plot_atm_dimers_over_space(self):
        fig, ax = plt.subplots()

        self.plot_dimers_density_over_space(ax)

        self.label_x_space_axis(ax)
        self.label_y_atm_dimers_density(ax)

        plt.show()

    def plot_apoe_over_space(self):
        fig, ax = plt.subplots()

        self.plot_apoe_density_over_space(ax)

        self.label_x_space_axis(ax)
        self.label_y_apoe_density(ax)

        plt.show()

    def plot_complexes_over_space(self):
        fig, ax = plt.subplots()

        self.plot_complexes_density_over_space(ax)

        self.label_x_space_axis(ax)
        self.label_y_complexes_density(ax)

        plt.show()

    def plot_monomers_density_over_space(self, ax: plt.Axes):
        ax.plot(self.spatial_space.space, self.atm_apoe_system.get_monomers(), color='dodgerblue', label="$M$")

    def plot_dimers_density_over_space(self, ax: plt.Axes):
        ax.plot(self.spatial_space.space, self.atm_apoe_system.get_dimers(), color='crimson', label="$D$")

    def plot_apoe_density_over_space(self, ax: plt.Axes):
        ax.plot(self.spatial_space.space, self.atm_apoe_system.get_apoe(), color='magenta', label="$A$")

    def plot_complexes_density_over_space(self, ax: plt.Axes):
        ax.plot(self.spatial_space.space, self.atm_apoe_system.get_complexes(), color='darkorange', label="$C$")

    @staticmethod
    def label_y_monomers_density(ax: plt.Axes):
        ax.set_ylabel("ATM monomers density", fontsize=12)

    @staticmethod
    def label_y_atm_dimers_density(ax: plt.Axes):
        ax.set_ylabel("ATM dimers density", fontsize=12)
        ax.set_ylim([0, None])

    @staticmethod
    def label_y_apoe_density(ax: plt.Axes):
        ax.set_ylabel("ApoE protein density", fontsize=12)

    @staticmethod
    def label_y_complexes_density(ax: plt.Axes):
        ax.set_ylabel("ApoE-ATM Complexes density", fontsize=12)

    @staticmethod
    def label_x_space_axis(ax: plt.Axes):
        ax.set_xlabel("Space", fontsize=12)

    @staticmethod
    def label_x_time_axis(ax: plt.Axes):
        ax.set_xlabel("Time", fontsize=12)

    def plot_crown_density_over_time(self):
        fig, ax = plt.subplots()

        ax.plot(self.time_space.space, self.crown_density_over_time, color='red')

        self.label_x_time_axis(ax)

        ax.set_ylabel("Perinuclear crown density", fontsize=12)

        plt.show()

    def plot_system_mass_over_time(self):
        fig, ax = plt.subplots()

        ax.plot(self.time_space.space, self.masses_atm, color="black")

        self.label_x_time_axis(ax)
        ax.set_ylabel("Masse of ATM proteins in the cell (kDa)", fontsize=12)

        plt.show()

    def plot_flux_nucleus_over_time(self):
        fig, ax = plt.subplots()

        ax.plot(self.time_space.space, self.atm_flux_nucleus)

        self.label_x_time_axis(ax)
        ax.set_ylabel("ATM flux in the nucleus", fontsize=12)

        plt.show()

    def plot_atm_dimers_over_space_and_time(self):
        fig, ax = plt.subplots()
        plot, = ax.plot(self.spatial_space.space, self.atm_apoe_system.dimers.every_time_values[0], color='crimson')
        ax.set_ylim(0, self.atm_apoe_system.dimers.every_time_values[-1].max() + 1)
        self.label_x_space_axis(ax)
        self.label_y_atm_dimers_density(ax)

        def animate(frame):
            plot.set_ydata(self.atm_apoe_system.dimers.every_time_values[frame])
            return plot,

        animation = anim.FuncAnimation(fig, animate, frames=self.time_space.nb_points, blit=True, interval=50)

        return animation
