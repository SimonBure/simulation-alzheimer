from spatial.oneD.OneDimSpace import TimeSpace
from compartmental.Compartment import Compartment
from compartmental.DynamicRate import DimerFormationRateCytoplasm


def test_multiplication() -> bool:
    a_time_space = TimeSpace(1, 5)

    a_compartment = Compartment(1.5, a_time_space)
    another_compartment = Compartment(5, a_time_space)

    a_rate = DimerFormationRateCytoplasm(1, 1)
    a_rate.actual_value = 5

    _ = a_compartment * another_compartment
    __ = another_compartment * a_compartment
    ___ = 5 * a_compartment
    ____ = a_compartment * 5
    _____ = a_rate * a_compartment
    ______ = a_compartment * a_rate
    return _ == __ == ___ == ____ == _____ == ______ == 7.5


if __name__ == "__main__":
    assert test_multiplication()
