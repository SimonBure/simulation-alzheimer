import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
from numpy import ndarray
import scipy.sparse as sparse
from scipy.sparse.linalg import factorized
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

    experiments: tuple[Antioxidant, Irradiation, Statin]

    def __init__(self, space_length: float, nb_space_points: int, maximum_time: float, time_step: float):
        self.time = 0.
        self.time_index = 0

        self.spatial_space = SpatialSpace(space_length, nb_space_points)

        nb_time_points = int(maximum_time / time_step)
        self.time_space = TimeSpace(maximum_time, nb_time_points)

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

    def create_all_solvers(self) -> tuple[factorized, factorized, factorized, factorized]:
        solver_natural = self.create_natural_system_solver()
        solver_during_antioxidant = self.create_antioxidant_system_solver()
        solver_during_irradiation = self.create_irradiation_system_solver()
        solver_during_antioxidant_and_irradiation = self.create_antioxidant_irradiation_system_solver()
        return (solver_natural, solver_during_antioxidant, solver_during_irradiation,
                solver_during_antioxidant_and_irradiation)

    def create_natural_system_solver(self) -> factorized:
        diagonals = self.create_diagonals(self.atm_apoe_system.diffusion_parameter.natural_value,
                                          self.atm_apoe_system.transport_parameter.natural_value)
        return self.create_solver_from_diagonals(diagonals)

    def create_antioxidant_system_solver(self) -> factorized:
        diagonals = self.create_diagonals(self.atm_apoe_system.diffusion_parameter.antioxidant_value,
                                          self.atm_apoe_system.transport_parameter.antioxidant_value)
        return self.create_solver_from_diagonals(diagonals)

    def create_irradiation_system_solver(self) -> factorized:
        diagonals = self.create_diagonals(self.atm_apoe_system.diffusion_parameter.natural_value,
                                          self.atm_apoe_system.transport_parameter.irradiation_value)
        return self.create_solver_from_diagonals(diagonals)

    def create_antioxidant_irradiation_system_solver(self) -> factorized:
        diagonals = self.create_diagonals(self.atm_apoe_system.diffusion_parameter.antioxidant_value,
                                          self.atm_apoe_system.transport_parameter.irradiation_value)
        return self.create_solver_from_diagonals(diagonals)

    def create_monomers_flux_solver(self) -> factorized:
        diffusion_coefficient = self.atm_apoe_system.diffusion_parameter.get_value_at_specific_time(self.time_index)
        transport_coefficient = self.atm_apoe_system.transport_parameter.get_value_at_specific_time(self.time_index)
        diagonals = self.create_diagonals(diffusion_coefficient, transport_coefficient)
        return self.create_solver_from_diagonals(diagonals)

    def create_solver_from_diagonals(self, diagonals: tuple[ndarray, ndarray, ndarray]) -> factorized:
        sparse_system_matrix = self.create_sparse_system_matrix_from_diagonals(diagonals)
        return factorized(sparse_system_matrix)

    def create_diagonals(self, diffusion: float, transport: float) -> tuple[ndarray, ndarray, ndarray]:
        diagonal, lower, upper = self.setup_diagonals()

        f = self.compute_fourier_number()
        p = self.compute_peclet_number()
        nucleus_permeability = self.compute_nucleus_permeability()

        diagonal[:] = 1 + 2 * diffusion * f + self.atm_apoe_system.transport_parameter.over_space_values * p
        lower[:] = - f * diffusion
        upper[:] = - f * diffusion - self.atm_apoe_system.transport_parameter.over_space_values[1:] * p

        # Robin-Neumann boundary conditions
        diagonal[0] = 1 + self.spatial_space.step * nucleus_permeability / diffusion
        diagonal[-1] = -1
        lower[-1] = (1 - self.spatial_space.step *
                     self.atm_apoe_system.transport_parameter.over_space_values[-1] / diffusion)
        upper[0] = - (1 + self.spatial_space.step * transport / diffusion)

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

    def compute_nucleus_permeability(self) -> float:
        if self.is_simulation_started():
            bulk_on_nucleus = self.atm_apoe_system.compute_bulk_over_nucleus()
            return self.atm_apoe_system.permeability_parameter.get_permeability_depending_on_bulk(bulk_on_nucleus,
                                                                                                  self.time_index)
        else:
            return 0

    def is_simulation_started(self) -> bool:
        return self.time == 0

    def create_sparse_system_matrix_from_diagonals(self, diagonals: tuple[ndarray, ndarray, ndarray]) -> (
            sparse.csc_array):
        system_matrix = sparse.diags_array(diagonals, offsets=[0, -1, 1],
                                           shape=(self.spatial_space.nb_points, self.spatial_space.nb_points),
                                           format='csc')
        return system_matrix

    def compute_monomers_next_density(self, solver: factorized) -> ndarray:
        monomers_reaction = self.atm_apoe_system.create_monomers_reaction_array(self.time_index)
        next_values = solver(self.atm_apoe_system.monomers.actual_values + self.time_space.step * monomers_reaction)
        return next_values

    def simulate(self):
        (solver_natural, solver_during_antioxidant, solver_during_irradiation,
         solver_during_antioxidant_and_irradiation) = self.create_all_solvers()

        while self.time < self.time_space.end:
            dimers_next_density = self.atm_apoe_system.compute_dimers_next_density(self.time_index)
            apoe_next_density = self.atm_apoe_system.compute_apoe_next_density(self.time_index)
            complexes_next_density = self.atm_apoe_system.compute_complexes_next_density(self.time_index)

            if self.is_migration_inside_nucleus_possible():
                right_solver = self.create_monomers_flux_solver()
            else:
                if self.is_antioxidant_now():
                    if self.is_irradiation_now():
                        right_solver = solver_during_antioxidant_and_irradiation
                    else:
                        right_solver = solver_during_antioxidant
                else:
                    if self.is_antioxidant_now():
                        right_solver = solver_during_irradiation
                    else:
                        right_solver = solver_natural

            monomers_next_density = self.compute_monomers_next_density(right_solver)

            self.atm_apoe_system.set_next_values(monomers_next_density, dimers_next_density, apoe_next_density,
                                                 complexes_next_density)
            self.atm_apoe_system.fill_every_time_values(self.time_index)
            self.atm_apoe_system.update_for_next_step()

            self.time_index += 1
            self.time += self.time_space.step

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

    def is_migration_inside_nucleus_possible(self) -> bool:
        return self.compute_nucleus_permeability() > 0

    def plot_atm_dimers_over_space(self):
        fig, ax = plt.subplots()

        ax.plot(self.spatial_space.space, self.atm_apoe_system.get_dimers(), color='crimson')

        self.label_x_space_axis(ax)
        self.label_y_atm_dimers_density(ax)

    @staticmethod
    def label_x_space_axis(ax: plt.Axes):
        ax.set_xlabel("Space")

    @staticmethod
    def label_y_atm_dimers_density(ax: plt.Axes):
        ax.set_ylabel("ATM dimers density")

    def plot_atm_dimers_over_space_and_time(self):
        fig, ax = plt.subplots()
        plot = ax.plot(self.spatial_space.space, self.atm_apoe_system.get_dimers(), color='crimson')

    def animate(self, frame: int):
        pass
