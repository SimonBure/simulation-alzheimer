import random

import pygame
import abc
import math


class Particle(abc.ABC):
    x: float
    y: float
    radius: float
    vx: float
    vy: float
    color: pygame.color.Color

    def __init__(self, x: float, y: float, vx: float, vy: float):
        self.x = x
        self.y = y
        self.radius = 5.
        self.vx = vx
        self.vy = vy
        self.color = pygame.color.Color("black")

    def __str__(self):
        return f"Particle in ({self.x}, {self.y})"

    def move(self):
        self.x += self.vx
        self.y += self.vy

    def brownian_motion(self):
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-1, 1)

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

    def compute_distance(self, particle: 'Particle') -> float:
        return math.sqrt((particle.x - self.x) ** 2 + (particle.y - self.y) ** 2)

    def compute_distance_x_axis(self, particle: 'Particle') -> float:
        return math.fabs(particle.x - self.x)

    def compute_distance_y_axis(self, particle: 'Particle') -> float:
        return math.fabs(particle.y - self.y)

    def compute_angle(self, particle: 'Particle') -> float:
        return math.acos(self.compute_distance_x_axis(particle) / self.compute_distance(particle))

    def is_collision(self, particle: 'Particle') -> bool:
        return self.compute_distance(particle) < self.radius + particle.radius

    def is_collision_along_x_axis(self, particle: 'Particle') -> bool:
        angle = self.compute_angle(particle)
        return (0. <= angle <= math.pi / 4) or (3 * math.pi / 4 <= angle <= math.pi)

    def compute_collision_distance(self, particle: 'Particle') -> float:
        return self.radius + particle.radius - self.compute_distance(particle)

    def compute_collision_distance_x_axis(self, particle: 'Particle') -> float:
        return self.radius + particle.radius - self.compute_distance_x_axis(particle)

    def compute_collision_distance_y_axis(self, particle: 'Particle') -> float:
        return self.radius + particle.radius - self.compute_distance_y_axis(particle)

    def move_after_collision_with_particle(self, particle: 'Particle'):
        collision_dist_x = self.compute_collision_distance_x_axis(particle)
        collision_dist_y = self.compute_collision_distance_y_axis(particle)
        dist_x = particle.x - self.x
        dist_y = particle.y - self.y
        if dist_x > 0:
            self.x -= collision_dist_x / 2
        else:
            self.x += collision_dist_x / 2
        if dist_y > 0:
            self.y -= collision_dist_y / 2
        else:
            self.y += collision_dist_y / 2

    def is_collision_on_the_x_axis(self, particle: 'Particle') -> bool:
        return particle.x - self.x > 0

    def is_collision_above(self, particle: 'Particle') -> bool:
        return particle.y - self.y > 0

    def is_collision_on_right_border(self, border_width: float) -> bool:
        return self.x + self.radius > border_width

    def is_collision_on_left_border(self) -> bool:
        return self.x - self.radius < 0

    def is_collision_on_bottom_border(self) -> bool:
        return self.y - self.radius < 0

    def is_collision_on_top_border(self, border_height: float) -> bool:
        return self.y + self.radius > border_height

    def collision_x(self):
        self.vx = - self.vx

    def collision_y(self):
        self.vy = - self.vy

    def move_after_collision_with_top_border(self, border_height: float):
        self.y = border_height - self.radius

    def move_after_collision_with_bottom_border(self):
        self.y = self.radius

    def move_after_collision_with_right_border(self, border_width: float):
        self.x = border_width - self.radius

    def move_after_collision_with_left_border(self):
        self.x = self.radius


class AtmProtein(Particle):
    def __init__(self, x: float, y: float, vx: float, vy: float):
        super().__init__(x, y, vx, vy)
        base_protein = ApoeProtein(x, y, vx, vy, 0)
        self.radius = base_protein.radius * 1.5322
        del base_protein
        self.color = pygame.color.Color("red")


class ApoeProtein(Particle):
    space_limit: float

    def __init__(self, x: float, y: float, vx: float, vy: float, space_limit: float):
        super().__init__(x, y, vx, vy)
        self.radius = 3.
        self.color = pygame.color.Color("blue")
        self.space_limit = space_limit

    def is_collision_on_left_border(self) -> bool:
        return self.x - self.radius < self.space_limit
