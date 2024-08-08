from spatial.oneD.OneDimSpace import TimeSpace
from compartmental.System import CompartmentalSystem
from compartmental.Rate import (ConstantRate, DimerFormationRateCrown, DimerFormationRateCytoplasm,
                                MonomerDispersionRate, ComplexesFormationRate, MigrationRateCytoplasmToPc,
                                MigrationRatePcToNucleus, FragmentationRate)
from compartmental.Dose import AntioxidantDose, IrradiationDose, StatinDose
from spatial.oneD.Experiment import Antioxidant, Irradiation, Statin


def test_setup_initial_conditions() -> bool:
    a_time_space = TimeSpace(1, 5)
    a_system = CompartmentalSystem(a_time_space)
    a_system.setup_initial_conditions(1, 1, 1, 1, 1, 1, 1)
    return True


def test_update_rates_and_compartments_without_experiments() -> bool:
    a_time_space = TimeSpace(1, 5000)
    a_system = CompartmentalSystem(a_time_space)
    k1 = 5
    print(str(k1))
    a_system.setup_initial_conditions(300, 1, 1, 1, 200, 1, 1)
    a_system.setup_rates(ConstantRate("lambda", 1), ConstantRate("d0", 0.01), ConstantRate("d1", 0.01),
                         DimerFormationRateCytoplasm("a", 1, 1), MigrationRateCytoplasmToPc("b", 1, 1, 1, 1),
                         MigrationRatePcToNucleus("c", 1, 1, 1, 1, 1, ), ComplexesFormationRate("d", 1, 1),
                         DimerFormationRateCrown("e", 1, 1, 1, 1), MonomerDispersionRate("f", 1),
                         FragmentationRate("g", 1, 1)
                         )
    a_system.setup_doses(AntioxidantDose(1, a_time_space, Antioxidant()),
                         IrradiationDose(1, a_time_space, Irradiation()), StatinDose(1, a_time_space, Statin())
                         )
    print(a_system)
    a_system.update_rates(0.5)  # initialize the rates
    a_system.set_next_compartments_values()
    print(a_system)
    a_system.update_compartments()
    a_system.update_rates(0.6)
    a_system.set_next_compartments_values()
    print(a_system)
    a_system.update_compartments()
    a_system.update_rates(0.7)
    a_system.set_next_compartments_values()
    print(a_system)
    return True


if __name__ == "__main__":
    assert test_setup_initial_conditions()
    assert test_update_rates_and_compartments_without_experiments()
    