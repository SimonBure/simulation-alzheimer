import abc


class DynamicRate(abc.ABC):
    actual_value: float

    def __add__(self, other) -> float:
        return self.actual_value + other

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

    @abc.abstractmethod
    def compute_next_value(self, *args):
        raise NotImplemented


class AntioxidantDependingRate(DynamicRate):
    @abc.abstractmethod
    def compute_next_value(self, *args: float) -> float:
        raise NotImplemented


class DimerFormationRateCytoplasm(AntioxidantDependingRate):
    a: float
    e: float

    def __init__(self, a: float, e: float):
        self.a = a
        self.e = e
        super().__init__()

    def compute_next_value(self, antioxidant: float) -> float:
        return self.a / (1 + self.e * antioxidant)


class ComplexesFormationRate(DimerFormationRateCytoplasm):
    pass


class MonomerDispersionRate(AntioxidantDependingRate):
    e: float

    def __init__(self, e: float):
        self.e = e
        super().__init__()

    def compute_next_value(self, antioxidant: float) -> float:
        return self.e * antioxidant


class AntioxidantCompartmentDependingRate(DynamicRate):
    @abc.abstractmethod
    def compute_next_value(self, antioxidant: float, compartment: float) -> float:
        raise NotImplemented


class MigrationRateCytoplasmToPc(AntioxidantCompartmentDependingRate):
    a: float
    b: float
    n: float
    e: float

    def __init__(self, a: float, b: float, n: float, e: float):
        self.a = a
        self.b = b
        self.n = n
        self.e = e
        super().__init__()

    def compute_next_value(self, antioxidant: float, compartment: float) -> float:
        return (self.a ** self.n) / ((self.a ** self.n + compartment ** self.n) * (1 + self.e * antioxidant))


class DimerFormationRateCrown(AntioxidantCompartmentDependingRate):
    a: float
    b: float
    n: float
    e: float

    def __init__(self, a: float, b: float, n: float, e: float):
        self.a = a
        self.b = b
        self.n = n
        self.e = e
        super().__init__()

    def compute_next_value(self, antioxidant: float, compartment: float) -> float:
        return (self.a * compartment ** self.n) / ((self.b ** self.n + compartment ** self.n)
                                                   * (1 + self.e * antioxidant))


class AntioxidantStatinCompartmentDependingRate(DynamicRate):
    @abc.abstractmethod
    def compute_next_value(self, antioxidant: float, compartment: float, statin: float) -> float:
        raise NotImplemented


class MigrationRatePcToNucleus(AntioxidantStatinCompartmentDependingRate):
    a: float
    b: float
    n: float
    e: float
    f: float

    def __init__(self, a: float, b: float, n: float, e: float, f: float):
        self.a = a
        self.b = b
        self.n = n
        self.e = e
        self.f = f
        super().__init__()

    def compute_next_value(self, antioxidant: float, compartment: float, statin: float) -> float:
        return (self.b * self.a ** self.n) / ((self.a ** self.n + compartment ** self.n) * (1 + self.e * antioxidant) *
                                              (1 + self.f * statin))


class FragmentationRate(AntioxidantDependingRate):
    cs: float
    e: float

    def __init__(self, cs: float, e: float):
        self.cs = cs
        self.e = e
        super().__init__()

    def compute_next_value(self, antioxidant: float, dose_irradiation: float) -> float:
        return (self.cs / (1 + self.e * antioxidant)) + dose_irradiation
