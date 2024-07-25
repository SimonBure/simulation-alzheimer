import numpy as np
from Space1D import Space1D


def test_map_value_to_index() -> bool:
    a_space = Space1D(1, 5, 0.2)
    a_value = 0.4
    expected_index = 2
    return a_space.map_value_to_index(a_value) == expected_index


if __name__ == "__main__":
    assert test_map_value_to_index()
