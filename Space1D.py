import numpy as np


class Space1D:
    end: float
    nb_points: int
    step: float
    space: np.ndarray

    def __init__(self, end: float, nb_points: int, step: float):
        self.end = end
        self.nb_points = nb_points
        self.step = step

        self.space = np.linspace(0, end, nb_points)

    def map_value_to_index(self, value: float) -> int:
        return int(value / self.step)


# Shallow class for code clarity
class SpatialSpace(Space1D):
    pass


# Shallow class for code clarity
class TimeSpace(Space1D):
    pass
