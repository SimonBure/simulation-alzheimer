import math


class Distance:
    cartesian: float
    x: float
    y: float

    def __init__(self, obj1, obj2):
        self.cartesian = math.sqrt((obj2.x - obj1.x) ** 2 + (obj2.y - obj1.y) ** 2)
        self.x = math.fabs(obj2.x - obj1.x)
        self.y = math.fabs(obj2.y - obj1.y)
