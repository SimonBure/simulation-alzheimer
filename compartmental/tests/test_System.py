from spatial.oneD.OneDimSpace import TimeSpace
from compartmental.System import CompartmentalSystem


def test_setup_initial_conditions() -> bool:
    a_time_space = TimeSpace(1, 5)
    a_system = CompartmentalSystem(a_time_space)
    a_system.setup_initial_conditions(1, 1, 1, 1, 1, 1, 1)
    return True


if __name__ == "__main__":
    assert test_setup_initial_conditions()
    