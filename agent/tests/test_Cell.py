from agent.Cell import Cell
from agent.Particle import Particle


def test_remove_particle() -> bool:
    a_cell = Cell(0)
    one_particle = Particle(0, 0, 0, 0)
    another_particle = Particle(1, 1, 0, 0)
    a_cell.add_particle(one_particle, another_particle)
    a_cell.remove_particle(one_particle)

    return (one_particle not in a_cell.particles_inside) and (another_particle in a_cell.particles_inside)


if __name__ == "__main__":
    assert test_remove_particle()
