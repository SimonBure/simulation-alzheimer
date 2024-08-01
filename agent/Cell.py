from agent.Particle import Particle


class Cell:
    id: int
    particles_inside: list[Particle]

    def __init__(self, identifier: int):
        self.id = identifier
        self.particles_inside = []

    def __str__(self):
        return f"Cell n°{self.id}"

    def print_particles_inside(self):
        for p in self.particles_inside:
            print(p)

    def add_particle(self, *particles: Particle):
        for p in particles:
            self.particles_inside.append(p)

    def remove_particle(self, particle: Particle):
        self.particles_inside.remove(particle)

    def clear_particle(self):
        self.particles_inside.clear()

    def collisions_inside(self):
        for p1 in self.particles_inside:
            for p2 in self.particles_inside:
                if p1 is not p2:
                    if p1.is_collision(p2):
                        p1.x = 0


class BorderCell(Cell):

    def __init__(self, identifier: int):
        super().__init__(identifier)

    def __str__(self):
        return "Border" + super().__str__()

    def collision_on_border(self, width: float, height: float):
        for p in self.particles_inside:
            if p.is_collision_on_left_border():
                p.collision_x()
                p.move_after_collision_with_left_border()
            elif p.is_collision_on_right_border(width):
                p.collision_x()
                p.move_after_collision_with_right_border(width)
            if p.is_collision_on_bottom_border():
                p.collision_y()
                p.move_after_collision_with_bottom_border()
            elif p.is_collision_on_top_border(height):
                p.collision_y()
                p.move_after_collision_with_top_border(height)
