import random
import math
import pygame
import matplotlib.pyplot as plt
import numpy as np
from agent.Particle import AtmProtein, ApoeProtein
from agent.Dimer import Dimer, AtmDimer, AtmApoeComplex
from Grid import Grid


def initialize_dimers(width_screen: int, height_screen: int, range_apoe_generation: int, number: int) -> list[Dimer]:
    dimers_list = []
    for _ in range(number):
        random_x = random.randint(0, width_screen - range_apoe_generation)
        random_y = random.randint(0, height_screen)
        atm1 = AtmProtein(random_x, random_y, (0, 0))
        random_theta = random.random() * 2 * math.pi
        r = 1.5 * atm1.radius
        x2 = random_x + r * math.cos(random_theta)
        y2 = random_y + r * math.sin(random_theta)
        atm2 = AtmProtein(x2, y2, (0, 0))
        dimers_list.append(AtmDimer(atm1, atm2))
    return dimers_list


def is_dimer_fragmenting(frag_proba: float) -> bool:
    return random.random() < frag_proba


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
    initial_dimers_nb = 500
    range_apoe = 50
    dimerisation_proba = 0.95
    complexation_proba = 0.95
    fragmentation_constant_stress_proba = 0.0001
    fragmentation_irradiation_proba = 0.99
    # Parameters of the beta-function used for the bias toward the nucleus.
    param_transport_nucl = alpha, beta = 1.5, 1  # the stronger alpha, the stronger the transport
    start_irradiation, end_irradiation = 10, 11

    # Initial conditions
    # atms = [AtmProtein(random.randint(0, width - range_apoe), random.randint(0, height),
    #                    param_transport_nucl) for _ in range(1000)]
    atms: list[AtmProtein] = []

    apoes = [ApoeProtein(random.randint(width - range_apoe, width), random.randint(0, height), width - range_apoe)
             for _ in range(100)]

    dimers = initialize_dimers(width, height, range_apoe, initial_dimers_nb)
    # dimers = []

    nb_atm_in_nucleus_over_time = []

    clock = pygame.time.Clock()
    pygame_start_time = pygame.time.get_ticks()
    real_time = 0

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

        if start_irradiation < real_time < end_irradiation:
            screen.fill(pygame.color.Color("salmon"))
        else:
            screen.fill(pygame.color.Color("white"))

        grid.clear_cells()

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

        for d in dimers:
            if start_irradiation < real_time < end_irradiation:
                fragmentation_proba = fragmentation_irradiation_proba
            else:
                fragmentation_proba = fragmentation_constant_stress_proba

            if is_dimer_fragmenting(fragmentation_proba):
                if isinstance(d, AtmDimer):
                    atm1 = AtmProtein(d.x_1, d.y_1, param_transport_nucl)
                    atm2 = AtmProtein(d.x_2, d.y_2, param_transport_nucl)
                    atms.append(atm1)
                    atms.append(atm2)
                else:
                    atm = AtmProtein(d.x_1, d.y_1, param_transport_nucl)
                    apoe_prot = ApoeProtein(d.x_2, d.y_2, width - range_apoe)
                    atms.append(atm)
                    apoes.append(apoe_prot)

                dimers.remove(d)
            else:
                d.draw(screen)
        # print(len(dimers))

        # grid.fill_cells_with_particles(particles)
        grid.fill_cells_with_particles(atms)
        grid.fill_cells_with_particles(apoes)
        grid.fill_cells_with_particles(dimers)

        grid.collision_inside_grid(dimers, atms, apoes, dimerisation_proba, complexation_proba)

        grid.collision_on_borders()

        pygame.display.flip()

        clock.tick(60)  # limitation des fps à 60

        real_time = (pygame.time.get_ticks() - pygame_start_time) / 1000  # in seconds
        print(real_time)

    end_time = pygame.time.get_ticks()
    total_time = end_time - pygame_start_time
    plot_flux_in_nucleus(nb_atm_in_nucleus_over_time, total_time)
    pygame.quit()
