
def hill_fct_migration(concentration: float, coefficients: tuple) -> float:
    a = coefficients[0]
    b = coefficients[1]
    n = coefficients[2]
    return (b * a ** n) / (a ** n + concentration ** n)


def hill_fct_dimer_formation(concentration: float, coefficients: tuple) -> float:
    a = coefficients[0]
    b = coefficients[1]
    n = coefficients[2]
    return (a * concentration ** n) / (b ** n + concentration ** n)


def time_flow():
    # Coefficients definition
    lam = 15
    d0 = 0.05
    d1 = 0.3
    coef2 = (400, 0.4, 15)
    coef3 = (80, 0.5, 5)
    k4 = 0.05
    coef5 = (0.4, 150, 15)

    # Initial conditions
    Dc0 = 300
    Mc0 = 0
    Ma0 = 0
    Mn0 = 0
    A0 = 200
    Da0 = 0
    Ca0 = 0


if __name__ == "__main__":
    # Unit test for the different Hill functions
    coef1 = (3, 1, 2)
    concentration1 = 2
    assert hill_fct_migration(concentration1, coef1) == 9 / 13
    assert hill_fct_migration(concentration1, coef1) - 12 / 5 < 1e-5
