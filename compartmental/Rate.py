import abc


class Rate(abc.ABC):
    name: str
    actual_value: float

    def __init__(self, name: str, value: float):
        self.name = name
        self.actual_value = value

    def __str__(self) -> str:
        return f"{self.name:>6}{self.actual_value:>14.2f}"

    def __add__(self, other) -> float:
        return self.actual_value + other

    def __sub__(self, other) -> float:
        return self.actual_value - other

    def __neg__(self):
        return - self.actual_value

    def __mul__(self, other) -> float:
        return self.actual_value * other

    def __rmul__(self, other) -> float:
        return other * self.actual_value

    def __pow__(self, power, modulo=None):
        return self.actual_value ** power

    def update(self, value: float):
        self.actual_value = value


class ConstantRate(Rate):
    def __init__(self, name: str, value: float):
        super().__init__(name, value)


class DynamicRate(Rate):
    name: str
    actual_value: float

    def __init__(self, name: str, value: float):
        super().__init__(name, value)

    @abc.abstractmethod
    def compute_next_value(self, *args):
        raise NotImplemented


class AntioxidantDependingRate(DynamicRate):
    def __init__(self, name: str, value: float):
        super().__init__(name, value)

    @abc.abstractmethod
    def compute_next_value(self, *args: float) -> float:
        raise NotImplemented


class DimerFormationRateCytoplasm(AntioxidantDependingRate):
    a: float
    e: float

    def __init__(self, name: str, a: float, e: float):
        super().__init__(name, 0)
        self.a = a
        self.e = e

    def compute_next_value(self, antioxidant: float) -> float:
        return self.a / (1 + self.e * antioxidant)


class ComplexesFormationRate(DimerFormationRateCytoplasm):
    def __init__(self, name: str, a: float, e: float):
        super().__init__(name, a, e)


class MonomerDispersionRate(AntioxidantDependingRate):
    e: float

    def __init__(self, name: str, e: float):
        super().__init__(name, 0)
        self.e = e

    def compute_next_value(self, antioxidant: float) -> float:
        return self.e * antioxidant


class AntioxidantCompartmentDependingRate(DynamicRate):
    def __init__(self, name: str, value: float):
        super().__init__(name, value)

    @abc.abstractmethod
    def compute_next_value(self, antioxidant: float, compartment: float) -> float:
        raise NotImplemented


class MigrationRateCytoplasmToPc(AntioxidantCompartmentDependingRate):
    a: float
    b: float
    n: float
    e: float

    def __init__(self, name: str, a: float, b: float, n: float, e: float):
        super().__init__(name, 0)
        self.a = a
        self.b = b
        self.n = n
        self.e = e

    def compute_next_value(self, antioxidant: float, compartment: float) -> float:
        return (self.a ** self.n) / ((self.a ** self.n + compartment ** self.n) * (1 + self.e * antioxidant))


class DimerFormationRateCrown(AntioxidantCompartmentDependingRate):
    a: float
    b: float
    n: float
    e: float

    def __init__(self, name: str, a: float, b: float, n: float, e: float):
        super().__init__(name, 0)
        self.a = a
        self.b = b
        self.n = n
        self.e = e

    def compute_next_value(self, antioxidant: float, compartment: float) -> float:
        return (self.a * compartment ** self.n) / ((self.b ** self.n + compartment ** self.n)
                                                   * (1 + self.e * antioxidant))


class AntioxidantStatinCompartmentDependingRate(DynamicRate):
    def __init__(self, name: str, value: float):
        super().__init__(name, value)

    @abc.abstractmethod
    def compute_next_value(self, antioxidant: float, compartment: float, statin: float) -> float:
        raise NotImplemented


class MigrationRatePcToNucleus(AntioxidantStatinCompartmentDependingRate):
    a: float
    b: float
    n: float
    e: float
    f: float

    def __init__(self, name: str, a: float, b: float, n: float, e: float, f: float):
        super().__init__(name, 0)
        self.a = a
        self.b = b
        self.n = n
        self.e = e
        self.f = f

    def compute_next_value(self, antioxidant: float, compartment: float, statin: float) -> float:
        return (((self.b * self.a ** self.n) / (self.a ** self.n + compartment ** self.n)) * (1 + self.e * antioxidant)
                * (1 + self.f * statin))


class FragmentationRate(AntioxidantDependingRate):
    cs: float
    e: float

    def __init__(self, name: str, cs: float, e: float):
        super().__init__(name, 0)
        self.cs = cs
        self.e = e

    def compute_next_value(self, antioxidant: float, dose_irradiation: float) -> float:
        return (self.cs / (1 + self.e * antioxidant)) + dose_irradiation
