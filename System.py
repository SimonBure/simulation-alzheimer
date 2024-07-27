import numpy as np
from numpy import ndarray
from DensityOverSpace import AtmMonomers, AtmDimers, ApoeProteins, ApoeAtmComplexes
from Parameter import DiffusionParameter, TransportParameter, FragmentationParameter, PermeabilityParameter
from Space1D import TimeSpace, SpatialSpace
from Experiment import Antioxidant, Irradiation, Statin


class ReactionDiffusionAtmApoeSystem:
    monomers: AtmMonomers
    dimers: AtmDimers
    apoe_proteins: ApoeProteins
    complexes: ApoeAtmComplexes
    populations: [AtmMonomers, AtmDimers, ApoeProteins, ApoeAtmComplexes]

    k: float
    ka: float
    diffusion_parameter: DiffusionParameter
    transport_parameter: TransportParameter
    fragmentation_parameter: FragmentationParameter
    permeability_parameter: PermeabilityParameter
    variable_parameters: [DiffusionParameter, TransportParameter, FragmentationParameter, PermeabilityParameter]

    ratio_fragmentation_dimers_complexes: float

    spatial_space: SpatialSpace
    time_space: TimeSpace

    def setup_spaces(self, spatial_space: SpatialSpace, time_space: TimeSpace):
        self.spatial_space = spatial_space
        self.time_space = time_space

    def setup_parameters(self, k: float, ka: float, diffusion: DiffusionParameter, transport: TransportParameter,
                         fragmentation: FragmentationParameter, permeability: PermeabilityParameter,
                         ratio_fragmentation_dimers_complexes: float):
        self.k = k
        self.ka = ka

        self.diffusion_parameter = diffusion
        self.transport_parameter = transport
        self.fragmentation_parameter = fragmentation
        self.permeability_parameter = permeability

        self.variable_parameters = [self.diffusion_parameter, self.transport_parameter, self.fragmentation_parameter,
                                    self.permeability_parameter]
        self.setup_parameters_values_over_time()

        self.ratio_fragmentation_dimers_complexes = ratio_fragmentation_dimers_complexes

    def setup_parameters_values_over_time(self):
        for parameter in self.variable_parameters:
            parameter.setup_values_over_time(self.time_space)

    def setup_experiments_impact_on_parameters(self, antioxidant: Antioxidant, irradiation: Irradiation,
                                               statin: Statin):
        for parameter in self.variable_parameters[:-1]:
            parameter.impact_antioxidant(antioxidant, self.time_space)

        for parameter in self.variable_parameters[:-2]:
            parameter.impact_irradiation(irradiation, self.time_space)

        self.fragmentation_parameter.impact_antioxidant_and_irradiation_combined(antioxidant, irradiation,
                                                                                 self.time_space)
        self.permeability_parameter.impact_statin(statin, self.time_space)

    def setup_initial_population_conditions(self, monomers_initial: ndarray, dimers_initial: ndarray,
                                            apoe_initial: ndarray, complexes_initial: ndarray):
        self.monomers = AtmMonomers(monomers_initial)
        self.dimers = AtmDimers(dimers_initial)
        self.apoe_proteins = ApoeProteins(apoe_initial)
        self.complexes = ApoeAtmComplexes(complexes_initial)
        self.populations = [self.monomers, self.dimers, self.apoe_proteins, self.complexes]

    def setup_populations_every_time_values(self):
        for pop in self.populations:
            pop.setup_every_time_values(self.time_space)

    def compute_dimers_next_density(self, time_simulation_index: int) -> ndarray:
        fragmentation_rate_dimers = (self.ratio_fragmentation_dimers_complexes
                                     * self.fragmentation_parameter.over_time_values[time_simulation_index])
        return (self.dimers.actual_values + self.time_space.step * (0.5 * self.k * self.monomers.actual_values
                                                                    - fragmentation_rate_dimers
                                                                    * self.dimers.actual_values))

    def compute_apoe_next_density(self, time_simulation_index: int) -> ndarray:
        fragmentation_rate_complexes = self.fragmentation_parameter.over_time_values[time_simulation_index]
        return (self.apoe_proteins.actual_values + self.time_space.step * (- self.ka * self.monomers.actual_values
                                                                           * self.apoe_proteins.actual_values
                                                                           + fragmentation_rate_complexes
                                                                           * self.complexes.actual_values))

    def compute_complexes_next_density(self, time_simulation_index: int) -> ndarray:
        fragmentation_rate_complexes = self.fragmentation_parameter.over_time_values[time_simulation_index]
        return (self.complexes.actual_values + self.time_space.step * (self.ka* self.monomers.actual_values
                                                                       * self.apoe_proteins.actual_values
                                                                       - fragmentation_rate_complexes
                                                                       * self.complexes.actual_values))

    def compute_bulk_over_nucleus(self) -> float:
        protein_bulk = float(3.66 * self.monomers.actual_values[0] + 6.98 * self.dimers.actual_values[0]
                             + self.apoe_proteins.actual_values[0] + 4.66 * self.complexes.actual_values[0])
        return protein_bulk

    def create_monomers_reaction_array(self, time_simulation_index: int) -> ndarray:
        fragmentation_rate = self.fragmentation_parameter.over_time_values[time_simulation_index]
        reaction_array = np.zeros_like(self.monomers.actual_values)

        reaction_array[1:-1] = (- self.k * self.monomers.actual_values[1:-1] * self.monomers.actual_values[1:-1]
                                - self.ka * self.monomers.actual_values[1:-1] * self.apoe_proteins.actual_values[1:-1]
                                + fragmentation_rate * self.complexes.actual_values[1:-1]
                                + self.ratio_fragmentation_dimers_complexes * fragmentation_rate
                                * 2 * self.dimers.actual_values[1:-1])

        reaction_array[0] = -self.monomers.actual_values[0]
        reaction_array[-1] = -self.monomers.actual_values[-1]

        return reaction_array

    # TODO method to assign the next_values of the different populations of the system
