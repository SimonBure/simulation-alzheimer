from numpy import ndarray
from DensityOverSpace import AtmMonomers, AtmDimers, ApoeProteins, ApoeAtmComplexes
from Parameter import DiffusionParameter, TransportParameter, FragmentationParameter, PermeabilityParameter
from Space1D import TimeSpace, SpatialSpace


class ReactionDiffusionAtmApoeSystem:
    monomers: AtmMonomers
    dimers: AtmDimers
    apoe_proteins: ApoeProteins
    complexes: ApoeAtmComplexes

    k: float
    ka: float
    diffusion_parameter: DiffusionParameter
    transport_parameter: TransportParameter
    fragmentation_parameter: FragmentationParameter
    permeability_parameter: PermeabilityParameter

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
        self.ratio_fragmentation_dimers_complexes = ratio_fragmentation_dimers_complexes

    def setup_initial_population_conditions(self, monomers_initial: ndarray, dimers_initial: ndarray,
                                            apoe_initial: ndarray, complexes_initial: ndarray):
        self.monomers = AtmMonomers(monomers_initial)
        self.dimers = AtmDimers(dimers_initial)
        self.apoe_proteins = ApoeProteins(apoe_initial)
        self.complexes = ApoeAtmComplexes(complexes_initial)

    def compute_dimers_next_density(self, time_simulation_index: int):
        fragmentation_rate_dimers = (self.ratio_fragmentation_dimers_complexes *
                                     self.fragmentation_parameter.over_time_values[time_simulation_index])
        self.dimers.next_values = (self.dimers.actual_values + self.time_space.step *
                                   (0.5 * self.k * self.monomers.actual_values -
                                    fragmentation_rate_dimers * self.dimers.actual_values))

    def compute_apoe_next_density(self, time_simulation_index: int):
        fragmentation_rate_complexes = self.fragmentation_parameter.over_time_values[time_simulation_index]
        self.apoe_proteins.next_values = (self.apoe_proteins.actual_values + self.time_space.step *
                                          (- self.ka * self.monomers.actual_values * self.apoe_proteins.actual_values +
                                           fragmentation_rate_complexes * self.complexes.actual_values))

    def compute_complexes_next_density(self, time_simulation_index: int):
        fragmentation_rate_complexes = self.fragmentation_parameter.over_time_values[time_simulation_index]
        self.complexes.next_values = (self.complexes.actual_values + self.time_space.step *
                                      (self.ka * self.monomers.actual_values * self.apoe_proteins.actual_values -
                                       fragmentation_rate_complexes * self.complexes.actual_values))
