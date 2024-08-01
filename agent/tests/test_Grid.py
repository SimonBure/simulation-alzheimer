from agent.Grid import Grid
from agent.Particle import Particle


def test_map_particle_to_cell() -> bool:
    a_grid = Grid(1, 1, 11, 11)
    a_grid.setup_grid()
    particles = [Particle(0.5, 0.5, 0, 0)]
    actual_cell = a_grid.map_particle_to_cell(particles[0])
    expected_cell_id = 1
    return actual_cell.id == expected_cell_id


if __name__ == "__main__":
    assert test_map_particle_to_cell()
