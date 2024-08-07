from compartmental.DynamicRate import DimerFormationRateCytoplasm, MigrationRateCytoplasmToPc, MigrationRatePcToNucleus


def test_all_kind_of_constructors() -> bool:
    DimerFormationRateCytoplasm(1, 1)
    MigrationRateCytoplasmToPc(1, 1, 1, 1)
    MigrationRatePcToNucleus(1, 1, 1, 1, 1)
    return True


def test_multiplication() -> bool:
    a_rate = DimerFormationRateCytoplasm(1, 1)
    a_rate.actual_value = 4

    another_rate = DimerFormationRateCytoplasm(1, 1)
    another_rate.actual_value = 5

    _ = 5 * a_rate
    __ = a_rate * 5
    ___ = a_rate * another_rate
    ____ = another_rate * a_rate
    return _ == __ == ___ == ____ == 20


if __name__ == "__main__":
    assert test_all_kind_of_constructors()
    assert test_multiplication()
