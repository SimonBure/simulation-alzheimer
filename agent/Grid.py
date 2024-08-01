import math
from agent.Cell import Cell, BorderCell
from agent.Particle import Particle


class Grid:
    width: float
    height: float
    nb_cells_width: int
    nb_cells_height: int

    cell_width: float
    cell_height: float

    cell_grid: list[list[Cell]]
    border_cells: list[BorderCell]

    def __init__(self, width: float, height: float, nb_cells_width: int, nb_cells_height: int):
        self.width = width
        self.height = height
        self.nb_cells_width = nb_cells_width
        self.nb_cells_height = nb_cells_height

        self.cell_width = self.width / self.nb_cells_width
        self.cell_height = self.height / self.nb_cells_height

        self.cell_grid = [[None for _ in range(self.nb_cells_width)] for _ in range(self.nb_cells_height)]
        self.border_cells = []

    def __str__(self) -> str:
        s = "--" * self.nb_cells_width + "\n"
        for line in self.cell_grid:
            s += "|"
            for cell in line:
                s += f"{cell.id} "
            s += "|\n"
        s += "--" * self.nb_cells_width
        return s

    def setup_grid(self):
        index = 1
        for i in range(self.nb_cells_height):
            for j in range(self.nb_cells_width):
                if i == 0 or j == 0 or i == self.nb_cells_height - 1 or j == self.nb_cells_width - 1:
                    border_cell = BorderCell(index)
                    self.cell_grid[i][j] = border_cell
                    self.border_cells.append(border_cell)
                else:
                    self.cell_grid[i][j] = Cell(index)
                index += 1

    def fill_cells_with_particles(self, particles: list[Particle]):
        for p in particles:
            cell_containing_p = self.map_particle_to_cell(p)
            cell_containing_p.add_particle(p)

    def clear_cells(self):
        for line in self.cell_grid:
            for cell in line:
                cell.clear_particle()

    def map_particle_to_cell(self, particle: Particle) -> Cell:
        index_x_axis = particle.x / self.cell_width
        index_y_axis = particle.y / self.cell_height
        x_in_cell_units = int(index_x_axis) if index_x_axis < self.nb_cells_width - 1 else self.nb_cells_width - 1
        y_in_cell_units = int(index_y_axis) if index_y_axis < self.nb_cells_height - 1 else self.nb_cells_height - 1
        return self.cell_grid[x_in_cell_units][y_in_cell_units]

    def collision_on_borders(self):
        for b_cell in self.border_cells:
            b_cell.collision_on_border(self.width, self.height)

    def collision_inside_grid(self):
        for line in self.cell_grid:
            for cell in line:
                cell.collisions_inside()
