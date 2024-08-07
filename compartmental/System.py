from numpy import ndarray
from compartmental.Compartment import Compartment
from compartmental.Dose import AntioxidantDose, IrradiationDose, StatinDose
from compartmental.DynamicRate import (DimerFormationRateCrown, DimerFormationRateCytoplasm, MonomerDispersionRate,
                                       ComplexesFormationRate, MigrationRateCytoplasmToPc, MigrationRatePcToNucleus,
                                       FragmentationRate)
from spatial.oneD.OneDimSpace import TimeSpace


class CompartmentalSystem:
    dimers_cytoplasm: Compartment
    monomers_cytoplasm: Compartment
    monomers_crown: Compartment
    monomers_nucleus: Compartment
    apoe_proteins: Compartment
    complexes: Compartment
    dimers_crown: Compartment
    compartments: tuple[Compartment, Compartment, Compartment, Compartment, Compartment, Compartment, Compartment]

    degradation_rate_dimers: float
    degradation_rate_monomers_nucleus: float
    k1: DimerFormationRateCytoplasm
    k2: MigrationRateCytoplasmToPc
    k3: MigrationRatePcToNucleus
    k4: ComplexesFormationRate
    k5: DimerFormationRateCrown
    k6: MonomerDispersionRate
    fragmentation_rate: FragmentationRate
    production_rate_dimers: float

    antioxidant: AntioxidantDose
    irradiation: IrradiationDose
    statin: StatinDose

    time_space: TimeSpace

    def __init__(self, time_space: TimeSpace):
        self.time_space = time_space

    def setup_initial_conditions(self, initial_dimers_cytoplasm: float, initial_monomers_cytoplasm: float,
                                 initial_monomers_crown: float, initial_monomers_nucleus: float, initial_apoe: float,
                                 initial_complexes: float, initial_dimers_crown: float):
        self.dimers_cytoplasm = Compartment(initial_dimers_cytoplasm, self.time_space)
        self.monomers_cytoplasm = Compartment(initial_monomers_cytoplasm, self.time_space)
        self.monomers_crown = Compartment(initial_monomers_crown, self.time_space)
        self.monomers_nucleus = Compartment(initial_monomers_nucleus, self.time_space)
        self.apoe_proteins = Compartment(initial_apoe, self.time_space)
        self.complexes = Compartment(initial_complexes, self.time_space)
        self.dimers_crown = Compartment(initial_dimers_crown, self.time_space)
        self.compartments = (self.dimers_cytoplasm, self.monomers_cytoplasm, self.monomers_crown, self.monomers_nucleus,
                             self.apoe_proteins, self.complexes, self.dimers_crown)

    def setup_rates(self, production_rate_dimers: float, d0: float, d1: float, k1: DimerFormationRateCytoplasm,
                    k2: MigrationRateCytoplasmToPc, k3: MigrationRatePcToNucleus, k4: ComplexesFormationRate,
                    k5: DimerFormationRateCrown, k6: MonomerDispersionRate, s: FragmentationRate):
        self.production_rate_dimers = production_rate_dimers
        self.degradation_rate_dimers = d0
        self.degradation_rate_monomers_nucleus = d1
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.k5 = k5
        self.k6 = k6
        self.fragmentation_rate = s

    def setup_doses(self, antioxidant: AntioxidantDose, irradiation: IrradiationDose, statin: StatinDose):
        self.antioxidant = antioxidant
        self.irradiation = irradiation
        self.statin = statin

    def get_compartments_time_values(self) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
        return (self.dimers_cytoplasm.time_values, self.monomers_cytoplasm.time_values, self.monomers_crown.time_values,
                self.monomers_nucleus.time_values, self.apoe_proteins.time_values, self.complexes.time_values,
                self.dimers_crown.time_values
                )

    def get_compartments_cytoplasm_and_nucleus_time_values(self) -> tuple[ndarray, ndarray, ndarray]:
        return self.dimers_cytoplasm.time_values, self.monomers_cytoplasm.time_values, self.monomers_nucleus.time_values

    def get_compartments_crown_time_values(self) -> tuple[ndarray, ndarray, ndarray, ndarray]:
        return (self.monomers_crown.time_values, self.apoe_proteins.time_values, self.complexes.time_values,
                self.dimers_crown.time_values
                )

    def compute_next_value_dimers_cytoplasm(self) -> float:
        return (self.dimers_cytoplasm + self.time_space.step * (self.production_rate_dimers
                                                                - self.dimers_cytoplasm * self.degradation_rate_dimers
                                                                + 0.5 * self.k1 * self.monomers_cytoplasm ** 2
                                                                - self.fragmentation_rate * self.dimers_cytoplasm)
                )

    def compute_next_value_monomers_cytoplasm(self) -> float:
        return (self.monomers_cytoplasm + self.time_space.step * (- self.k1 * self.monomers_cytoplasm ** 2
                                                                  - self.k2 * self.monomers_crown
                                                                  + self.k6 * self.monomers_crown
                                                                  + 2 * self.fragmentation_rate * self.dimers_cytoplasm)
                )

    def compute_next_value_monomers_crown(self) -> float:
        return (self.monomers_crown + self.time_space.step * (self.k2 * self.monomers_cytoplasm
                                                              - self.k3 * self.monomers_crown
                                                              - self.k4 * self.apoe_proteins * self.monomers_crown
                                                              - self.k5 * self.monomers_crown ** 2
                                                              - self.k6 * self.monomers_crown
                                                              + 2 * self.fragmentation_rate * self.dimers_crown
                                                              + self.fragmentation_rate * self.complexes)
                )

    def compute_next_value_monomers_nucleus(self) -> float:
        return self.monomers_nucleus + self.time_space.step * (self.k3 * self.monomers_crown -
                                                               self.monomers_nucleus *
                                                               self.degradation_rate_monomers_nucleus)

    def compute_next_value_apoe_proteins(self) -> float:
        return self.apoe_proteins + self.time_space.step * (- self.k4 * self.monomers_crown * self.apoe_proteins
                                                            + self.fragmentation_rate * self.complexes)

    def compute_next_value_complexes(self) -> float:
        return self.complexes + self.time_space.step * (self.k4 * self.monomers_crown * self.apoe_proteins -
                                                        self.fragmentation_rate * self.complexes)

    def compute_next_value_dimers_crown(self) -> float:
        return self.dimers_crown + self.time_space.step * (0.5 * self.k5 * self.monomers_crown ** 2 -
                                                           self.fragmentation_rate * self.dimers_crown)

    def update_rates(self, time: float):
        dose_aox = self.antioxidant.get_dose(time, self.time_space)
        dose_irr = self.irradiation.get_dose(time, self.time_space)
        dose_statin = self.statin.get_dose(time, self.time_space)
        self.k1.update(self.k1.compute_next_value(dose_aox))
        self.k2.update(self.k2.compute_next_value(dose_aox, self.dimers_crown.actual_value))
        self.k3.update(self.k3.compute_next_value(dose_aox, self.dimers_crown.actual_value, dose_statin))
        self.k4.update(self.k4.compute_next_value(dose_aox))
        self.k5.update(self.k5.compute_next_value(dose_aox, self.complexes.actual_value))
        self.k6.update(self.k6.compute_next_value(dose_aox))
        self.fragmentation_rate.update(self.fragmentation_rate.compute_next_value(dose_aox, dose_irr))

    def set_next_compartments_values(self):
        self.dimers_cytoplasm.set_next_value(self.compute_next_value_dimers_cytoplasm())
        self.monomers_cytoplasm.set_next_value(self.compute_next_value_monomers_cytoplasm())
        self.monomers_crown.set_next_value(self.compute_next_value_monomers_crown())
        self.monomers_nucleus.set_next_value(self.compute_next_value_monomers_nucleus())
        self.apoe_proteins.set_next_value(self.compute_next_value_apoe_proteins())
        self.complexes.set_next_value(self.compute_next_value_complexes())
        self.dimers_crown.set_next_value(self.compute_next_value_dimers_crown())

    def update_compartments(self):
        for c in self.compartments:
            c.update_value_for_next_step()

    def fill_time_values(self, time_index: int):
        for c in self.compartments:
            c.fill_time_values(time_index)
