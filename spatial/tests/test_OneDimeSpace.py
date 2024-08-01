from spatial.oneD.OneDimSpace import OneDimeSpace


def test_step_value() -> bool:
    a_space = OneDimeSpace(1, 5)
    expected_step = 0.25
    return a_space.step == expected_step


def test_map_value_to_index() -> bool:
    a_space = OneDimeSpace(1, 5)
    a_value = 0.4
    expected_index = 1
    return a_space.map_value_to_index(a_value) == expected_index


if __name__ == "__main__":
    assert test_step_value()
    assert test_map_value_to_index()
