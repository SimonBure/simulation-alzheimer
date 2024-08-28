import random
import math
import pygame
import matplotlib.pyplot as plt
import numpy as np
from agent.Particle import AtmProtein, ApoeProtein
from agent.Dimer import Dimer
from Grid import Grid

# TODO fonction formation dimère ou complexe si collision 2 atm, ou 1 atm et 1 apoe
# TODO irradiation

def initialize_dimers(width_screen: int, height_screen: int, range_apoe: int, number: int) -> list[Dimer]:
    dimers = []
    for _ in range(number):
        random_x = random.randint(0, width_screen - range_apoe)
        random_y = random.randint(0, height_screen)
        atm1 = AtmProtein(random_x, random_y, 0, 0, (0, 0))
        random_theta = random.random() * 2 * math.pi
        r = 1.5 * atm1.radius
        x2 = random_x + r * math.cos(random_theta)
        y2 = random_y + r * math.sin(random_theta)
        atm2 = AtmProtein(x2, y2, 0, 0, (0, 0))
        dimers.append(Dimer(atm1, atm2))
    return dimers


def plot_flux_in_nucleus(nb_atm_nucleus: list[int], time_simu: float):
    time_space = np.linspace(0, time_simu / 1000, len(nb_atm_nucleus))  # in sec for more readability
    plt.plot(time_space, nb_atm_nucleus)
    plt.xlabel("Time ($s$)", fontsize=12)
    plt.ylabel("Number of ATM in the nucleus", fontsize=12)
    plt.show()


if __name__ == "__main__":
    pygame.init()

    width, height = 500, 500
    screen = pygame.display.set_mode((width, height), flags=pygame.SCALED)

    # Parameters
    initial_dimers_nb = 1000
    range_apoe = 50
    dimerisation_proba = 0.95
    complexation_proba = 0.95
    fragmentation_constant_stress_proba = 0.001
    fragmentation_irradiation_proba = 0.999
    # Parameters of the beta-function used for the bias toward the nucleus.
    param_transport_nucl = alpha, beta = 1.5, 1  # the stronger alpha, the stronger the transport
    # Initial conditions
    atms = [AtmProtein(random.randint(0, width), random.randint(0, height), random.uniform(-1, 1),
                       random.uniform(-1, 1), param_transport_nucl) for _ in range(1000)]
    # atms: list[AtmProtein] = []

    apoes = [ApoeProtein(random.randint(width - range_apoe, width), random.randint(0, height), 0, 0, width - 50)
             for _ in range(100)]

    # complexes = initialize_dimers(width, height, range_apoe, initial_dimers_nb)
    complexes = []

    nb_atm_in_nucleus_over_time = []

    clock = pygame.time.Clock()
    start_time = pygame.time.get_ticks()

    grid = Grid(width, height, 100, 100)
    grid.setup_grid()

    is_running = True

    while is_running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    is_running = False

        pygame.display.set_caption(f"Individual Based Model Simulation\tFPS: {int(clock.get_fps())}")

        screen.fill(pygame.color.Color("white"))

        grid.clear_cells()

        for c in complexes:
            c.draw(screen)

        nb_atm_in_nucleus = 0
        for a in atms:
            a.brownian_motion_with_transport()
            a.move()
            if a.is_collision_on_right_border(width):
                nb_atm_in_nucleus += 1
                atms.remove(a)
                del a
            else:
                a.draw(screen)
        nb_atm_in_nucleus_over_time.append(nb_atm_in_nucleus)

        for apoe in apoes:
            apoe.draw(screen)

        # grid.fill_cells_with_particles(particles)
        grid.fill_cells_with_particles(atms)
        grid.fill_cells_with_particles(apoes)

        grid.collision_inside_grid()

        grid.collision_on_borders()

        pygame.display.flip()

        clock.tick(60)  # limitation des fps à 60

    end_time = pygame.time.get_ticks()
    total_time = end_time - start_time
    plot_flux_in_nucleus(nb_atm_in_nucleus_over_time, total_time)
    pygame.quit()
