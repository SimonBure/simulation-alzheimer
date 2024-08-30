import pygame
from Particle import Particle


class Dimer(Particle):
    x_1: float
    y_1: float
    x_2: float
    y_2: float
    radius_1: float
    radius_2: float
    color: pygame.color.Color = pygame.color.Color("black")

    def __init__(self, part1: Particle, part2: Particle):
        self.x_1 = part1.x
        self.y_1 = part1.y
        self.radius_1 = part1.radius
        self.x_2 = part2.x
        self.y_2 = part2.y
        self.radius_2 = part2.radius
        super().__init__(self.x_1 + self.x_2 / 2, self.y_1 + self.y_2 / 2, 0, 0)

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x_1, self.y_1), self.radius_1)
        pygame.draw.circle(screen, self.color, (self.x_2, self.y_2), self.radius_2)


class AtmDimer(Dimer):
    color: pygame.color.Color = pygame.color.Color("forestgreen")


class AtmApoeComplex(Dimer):
    color: pygame.color.Color = pygame.color.Color("midnightblue")
