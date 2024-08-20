import numpy as np
from numpy import ndarray
from scipy import integrate
from spatial.oneD.DensityOverSpace import DensityOverSpace, AtmMonomers, AtmDimers, ApoeProteins, ApoeAtmComplexes
from spatial.oneD.Parameter import DiffusionParameter, TransportParameter, FragmentationParameter, PermeabilityParameter
from spatial.oneD.OneDimSpace import TimeSpace, SpatialSpace
from spatial.oneD.Experiment import Antioxidant, Irradiation, Statin


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

    def __init__(self):
        pass

    def setup_spaces(self, spatial_space: SpatialSpace, time_space: TimeSpace):
        self.spatial_space = spatial_space
        self.time_space = time_space

    def setup_parameters(self, k: float, ka: float, diffusion: DiffusionParameter, transport: TransportParameter,
                         fragmentation: FragmentationParameter, permeability: PermeabilityParameter,
                         ratio_fragmentation_dimers_complexes: float, transport_constant: float):
        self.k = k
        self.ka = ka

        self.diffusion_parameter = diffusion
        self.transport_parameter = transport
        self.fragmentation_parameter = fragmentation
        self.permeability_parameter = permeability

        self.variable_parameters = [self.transport_parameter, self.fragmentation_parameter, self.diffusion_parameter,
                                    self.permeability_parameter]
        self.setup_parameters_values_over_time()
        self.transport_parameter.setup_values_over_space(self.spatial_space, transport_constant)

        self.ratio_fragmentation_dimers_complexes = ratio_fragmentation_dimers_complexes

    def setup_parameters_values_over_time(self):
        for parameter in self.variable_parameters:
            parameter.setup_values_over_time(self.time_space)

    def setup_experiments_impact_on_parameters(self, antioxidant: Antioxidant, irradiation: Irradiation,
                                               statin: Statin):
        # Don't go through the permeability parameter
        for parameter in self.variable_parameters[:-1]:
            parameter.impact_antioxidant(antioxidant, self.time_space)
        # Don't go through the permeability parameter and the diffusion parameter
        for parameter in self.variable_parameters[:-2]:
            parameter.impact_irradiation(irradiation, self.time_space)

        self.fragmentation_parameter.impact_antioxidant_and_irradiation_combined(antioxidant, irradiation,
                                                                                 self.time_space)
        self.permeability_parameter.impact_statin(statin, self.time_space)

    def setup_populations(self, monomers_initial: ndarray, dimers_initial: ndarray,
                          apoe_initial: ndarray, complexes_initial: ndarray):
        self.monomers = AtmMonomers(monomers_initial)
        self.dimers = AtmDimers(dimers_initial)
        self.apoe_proteins = ApoeProteins(apoe_initial)
        self.complexes = ApoeAtmComplexes(complexes_initial)
        self.populations = [self.monomers, self.dimers, self.apoe_proteins, self.complexes]
        self.setup_populations_every_time_values()

    def setup_populations_every_time_values(self):
        for pop in self.populations:
            pop.setup_every_time_values(self.time_space)

    def get_dimers(self) -> ndarray:
        return self.dimers.actual_values

    def get_monomers(self) -> ndarray:
        return self.monomers.actual_values

    def get_apoe(self) -> ndarray:
        return self.apoe_proteins.actual_values

    def get_complexes(self) -> ndarray:
        return self.complexes.actual_values

    def compute_dimers_next_density(self, time_simulation_index: int) -> ndarray:
        fragmentation_rate_dimers = (self.ratio_fragmentation_dimers_complexes
                                     * self.fragmentation_parameter.over_time_values[time_simulation_index])
        return (self.dimers.actual_values + self.time_space.step * (0.5 * self.k * self.monomers.actual_values ** 2
                                                                    - fragmentation_rate_dimers *
                                                                    self.dimers.actual_values))

    def compute_apoe_next_density(self, time_simulation_index: int) -> ndarray:
        fragmentation_rate_complexes = self.fragmentation_parameter.over_time_values[time_simulation_index]
        return (self.apoe_proteins.actual_values + self.time_space.step * (- self.ka * self.monomers.actual_values
                                                                           * self.apoe_proteins.actual_values
                                                                           + fragmentation_rate_complexes
                                                                           * self.complexes.actual_values))

    def compute_complexes_next_density(self, time_simulation_index: int) -> ndarray:
        fragmentation_rate_complexes = self.fragmentation_parameter.over_time_values[time_simulation_index]
        return (self.complexes.actual_values + self.time_space.step * (self.ka * self.monomers.actual_values
                                                                       * self.apoe_proteins.actual_values
                                                                       - fragmentation_rate_complexes
                                                                       * self.complexes.actual_values))

    def compute_nucleus_permeability(self, time_index: int, dimers_next: ndarray, apoe_next: ndarray,
                                     complexes_next: ndarray) -> float:
        bulk = self.compute_bulk_on_nucleus(dimers_next, apoe_next, complexes_next)
        return self.permeability_parameter.get_permeability_depending_on_bulk(bulk, time_index)

    def compute_perinuclear_crown(self, dimers_next: ndarray, apoe_next: ndarray, complexes_next: ndarray) -> float:
        bulk_nucleus = self.compute_bulk_on_nucleus(dimers_next, apoe_next, complexes_next)
        # To get the crown, we normalize the bulk according to the bulk at the initial conditions
        initial_bulk_nucleus = float(3.66 * self.monomers.every_time_values[0][0]
                                     + 6.98 * self.dimers.every_time_values[0][0]
                                     + self.apoe_proteins.every_time_values[0][0]
                                     + 4.66 * self.complexes.every_time_values[0][0])
        return bulk_nucleus - initial_bulk_nucleus

    @staticmethod
    def compute_bulk_on_nucleus(dimers_next: ndarray, apoe_next: ndarray, complexes_next: ndarray) -> float:
        return float(2 * dimers_next[0] + apoe_next[0] + 2 * complexes_next[0])

    def create_monomers_reaction_array(self, time_simulation_index: int) -> ndarray:
        fragmentation_rate = self.fragmentation_parameter.over_time_values[time_simulation_index]
        reaction_array = np.zeros_like(self.monomers.actual_values)

        reaction_array[1:-1] = (- self.k * self.monomers.actual_values[1:-1] * self.monomers.actual_values[1:-1]
                                - self.ka * self.monomers.actual_values[1:-1] * self.apoe_proteins.actual_values[1:-1]
                                + fragmentation_rate * self.complexes.actual_values[1:-1]
                                + self.ratio_fragmentation_dimers_complexes * fragmentation_rate
                                * 2 * self.dimers.actual_values[1:-1])

        reaction_array[0] = -self.monomers.actual_values[0] / self.time_space.step
        reaction_array[-1] = -self.monomers.actual_values[-1] / self.time_space.step

        return reaction_array

    def compute_system_mass(self) -> float:
        return self.compute_atm_mass() + self.compute_apoe_mass()

    def compute_atm_mass(self) -> float:
        return 350 * (self.compute_mass_from_density(self.monomers) + 2 * self.compute_mass_from_density(self.dimers)
                      + self.compute_mass_from_density(self.complexes))  # kDa

    def compute_apoe_mass(self) -> float:
        return 34 * (self.compute_mass_from_density(self.apoe_proteins) +
                     self.compute_mass_from_density(self.complexes))  # kDa

    def compute_mass_from_density(self, density: DensityOverSpace) -> float:
        return float(integrate.trapezoid(density.actual_values, self.spatial_space.space, self.spatial_space.step))

    def set_next_values(self, monomers_next_values: ndarray, dimers_next_values: ndarray, apoe_next_values: ndarray,
                        complexes_next_values: ndarray):
        next_values = monomers_next_values, dimers_next_values, apoe_next_values, complexes_next_values
        for next_val, pop in zip(next_values, self.populations):
            pop.set_next_values(next_val)

    def update_for_next_step(self):
        for pop in self.populations:
            pop.update_values_for_next_step()

    def fill_every_time_values(self, time_index: int):
        for pop in self.populations:
            pop.fill_every_time_values(time_index)
