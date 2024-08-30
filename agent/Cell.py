import random
from agent.Particle import Particle, AtmProtein, ApoeProtein
from agent.Dimer import Dimer, AtmDimer, AtmApoeComplex


class Cell:
    id: int
    particles_inside: list[Particle | Dimer]

    def __init__(self, identifier: int):
        self.id = identifier
        self.particles_inside = []

    def __str__(self):
        return f"Cell n°{self.id}"

    def print_particles_inside(self):
        for p in self.particles_inside:
            print(p)

    def add_particle(self, *particles: Particle | Dimer):
        for p in particles:
            self.particles_inside.append(p)

    def remove_particle(self, particle: Particle | Dimer):
        self.particles_inside.remove(particle)

    def clear_particle(self):
        self.particles_inside.clear()

    def collisions_inside(self, complexes: list[Dimer], atms: list[AtmProtein], apoes: list[ApoeProtein],
                          dimer_formation_proba: float, complex_formation_proba: float):
        for p1 in self.particles_inside:
            for p2 in self.particles_inside:
                if p1 is not p2:
                    if p1.is_collision(p2):
                        if isinstance(p1, AtmProtein) and isinstance(p2, AtmProtein):
                            if random.random() < dimer_formation_proba:
                                formed_dimer = AtmDimer(p1, p2)
                                self.remove_particle(p1)
                                self.remove_particle(p2)
                                self.add_particle(formed_dimer)
                                complexes.append(formed_dimer)
                                atms.remove(p1)
                                atms.remove(p2)
                            else:
                                p1.move_after_collision_with_particle(p2)
                                p2.move_after_collision_with_particle(p1)
                                if p1.is_collision_along_x_axis(p2):
                                    p1.collision_x()
                                    p2.collision_x()
                                else:
                                    p1.collision_y()
                                    p2.collision_y()
                        elif isinstance(p1, AtmProtein) and isinstance(p2, ApoeProtein):
                            if random.random() < complex_formation_proba:
                                formed_complex = AtmApoeComplex(p1, p2)
                                self.remove_particle(p1)
                                self.remove_particle(p2)
                                self.add_particle(formed_complex)
                                complexes.append(formed_complex)
                                atms.remove(p1)
                                apoes.remove(p2)
                            else:
                                p1.move_after_collision_with_particle(p2)
                                p2.move_after_collision_with_particle(p1)
                                if p1.is_collision_along_x_axis(p2):
                                    p1.collision_x()
                                    p2.collision_x()
                                else:
                                    p1.collision_y()
                                    p2.collision_y()
                        elif isinstance(p1, ApoeProtein) and isinstance(p2, AtmProtein):
                            if random.random() < complex_formation_proba:
                                formed_complex = AtmApoeComplex(p2, p1)
                                self.remove_particle(p1)
                                self.remove_particle(p2)
                                self.add_particle(formed_complex)
                                complexes.append(formed_complex)
                                apoes.remove(p1)
                                atms.remove(p2)
                            else:
                                p1.move_after_collision_with_particle(p2)
                                p2.move_after_collision_with_particle(p1)
                                if p1.is_collision_along_x_axis(p2):
                                    p1.collision_x()
                                    p2.collision_x()
                                else:
                                    p1.collision_y()
                                    p2.collision_y()
                        else:
                            p1.move_after_collision_with_particle(p2)
                            p2.move_after_collision_with_particle(p1)
                            if p1.is_collision_along_x_axis(p2):
                                p1.collision_x()
                                p2.collision_x()
                            else:
                                p1.collision_y()
                                p2.collision_y()


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
