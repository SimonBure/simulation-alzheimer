import numpy as np
from numpy import ndarray
import scipy.sparse as sparse
from scipy.sparse.linalg import factorized
from Space1D import SpatialSpace, TimeSpace
from Experiment import Antioxidant, Irradiation, Statin
from Parameter import DiffusionParameter, TransportParameter, FragmentationParameter, PermeabilityParameter
from System import ReactionDiffusionAtmApoeSystem


class Simulation1D:
    time: float
    time_index: int

    spatial_space: SpatialSpace
    time_space: TimeSpace

    atm_apoe_system: ReactionDiffusionAtmApoeSystem

    def __init__(self, space_length: float, nb_space_points: int, maximum_time: float, nb_time_points: int):
        self.time = 0.
        self.time_index = 0

        self.spatial_space = SpatialSpace(space_length, nb_space_points)
        self.time_space = TimeSpace(maximum_time, nb_time_points)

        self.atm_apoe_system = ReactionDiffusionAtmApoeSystem()
        self.atm_apoe_system.setup_spaces(self.spatial_space, self.time_space)

    def setup_atm_apoe_system(self, k: float, ka: float, diffusion_coefs: tuple[float, float],
                              transport_coefs: tuple[float, float], fragmentation_coefs: tuple[float, float, float],
                              permeability_coefs: tuple[float, float, float], ratio_fragm_dimer_complexes: float):
        diffusion_param = DiffusionParameter(diffusion_coefs[0], diffusion_coefs[1])
        transport_param = TransportParameter(transport_coefs[0], transport_coefs[1])
        fragmentation_param = FragmentationParameter(fragmentation_coefs[0], fragmentation_coefs[1],
                                                     fragmentation_coefs[2])
        permeability_param = PermeabilityParameter(permeability_coefs[0], permeability_coefs[1], permeability_coefs[2])

        self.atm_apoe_system.setup_parameters(k, ka, diffusion_param, transport_param, fragmentation_param,
                                              permeability_param, ratio_fragm_dimer_complexes)

    def setup_experimental_conditions(self, antioxidant_start_and_ending_time: tuple[float, float],
                                      irradiation_start_and_ending_time: tuple[float, float],
                                      statin_start_and_ending_time: tuple[float, float]):
        antioxidant = Antioxidant(antioxidant_start_and_ending_time[0], antioxidant_start_and_ending_time[1])
        irradiation = Irradiation(irradiation_start_and_ending_time[0], irradiation_start_and_ending_time[1])
        statin = Statin(statin_start_and_ending_time[0], statin_start_and_ending_time[1])
        self.atm_apoe_system.setup_experiments_impact_on_parameters(antioxidant, irradiation, statin)

    def setup_system_initial_conditions(self, monomers_initial: ndarray, dimers_initial: ndarray,
                                        apoe_initial: ndarray, complexes_initial: ndarray):
        self.atm_apoe_system.setup_initial_population_conditions(monomers_initial, dimers_initial, apoe_initial,
                                                                 complexes_initial)

    def create_system_solver(self) -> factorized:
        diagonals = self.create_system_matrix_diagonals()
        sparse_system_matrix = self.create_sparse_system_matrix_from_diagonals(diagonals)
        return factorized(sparse_system_matrix)

    def create_system_matrix_diagonals(self) -> tuple[ndarray, ndarray, ndarray]:
        diffusion_coefficient = float(self.atm_apoe_system.diffusion_parameter.over_time_values[self.time_index])
        transport_coefficient = float(self.atm_apoe_system.transport_parameter.over_time_values[self.time_index])

        return self.create_diagonals(diffusion_coefficient, transport_coefficient)

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
        bulk_on_nucleus = self.atm_apoe_system.compute_bulk_over_nucleus()
        return self.atm_apoe_system.permeability_parameter.get_permeability_depending_on_bulk(bulk_on_nucleus,
                                                                                              self.time_index)

    def create_sparse_system_matrix_from_diagonals(self, diagonals: tuple[ndarray, ndarray, ndarray]):
        system_matrix = sparse.diags(diagonals=diagonals, offsets=[0, -1, 1],
                                     shape=(self.spatial_space.nb_points, self.spatial_space.nb_points), format='csr')
        # Conversion to Compressed Sparse Column format for efficiency
        system_matrix = system_matrix.tocsc()
        return system_matrix

    def compute_monomers_next_density(self) -> ndarray:
        monomers_reaction = self.atm_apoe_system.create_monomers_reaction_array(self.time_index)
        # TODO finish this method use the solver from create_system_solver to compute monomers.next_values



