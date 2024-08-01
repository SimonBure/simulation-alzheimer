from agent.Grid import Grid
from agent.Particle import Particle


def test_map_particle_to_cell() -> bool:
    particles = [Particle(0.88, 0.88, 0, 0)]
    a_grid = Grid(1, 1, 4, 4)
    a_grid.setup_grid()
    actual_cell = a_grid.map_particle_to_cell(particles[0])
    expected_cell_id = 16
    return actual_cell.id == expected_cell_id


if __name__ == "__main__":
    assert test_map_particle_to_cell()
