#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pygame
import sys
from boids import *


# set the positions of obstacles
def add_obstacles_circle(boids, radius, num_obstacles, screen_size):
    angles = np.linspace(0, 2 * np.pi, num_obstacles, endpoint=False)

    for angle in angles:
        if boids.num_obstacles < boids.max_num_obstacles:
            position = [np.cos(angle), np.sin(angle)]
            position = radius * np.array(position) + screen_size / 2
            boids.add_obstacle(position)


def main():
    # initialize pygame module
    pygame.init()

    pygame.display.set_caption("Boids")
    screen_size = 750
    screen = pygame.display.set_mode((screen_size, screen_size))

    # set the clock
    clock = pygame.time.Clock()
    clock.tick(60)  # FPS

    # set the initial parameters
    num_boids = 100
    neighborhood_size = 45
    w = 0.1
    boids = boids_system(num_boids, neighborhood_size, screen_size)

    # set the triangle points
    triangle_points = 5 * np.array([[2, 0], [-1, -1], [-1, 1]])

    # set the fonts
    font_size = 22
    font = pygame.font.get_default_font()
    font = pygame.font.SysFont(font, font_size)

    # add the obstacles
    add_obstacles_circle(boids, 250, 20, screen_size)

    while True:
        screen.fill((0, 0, 0))

        weights_str = "Cohesion: %.2f, Alignment: %.2f, Separation: %.2f" % (
            boids.cohesion_weight, boids.alignment_weight, boids.separation_weight)
        label = font.render(weights_str, True, (255, 255, 0))
        screen.blit(label, (5, 5))

        weights_str = "Obstacles repulsion: %.2f, Attractors cohesion: %.2f, Attractors repulsion %.2f" % (
            boids.obstacles_repulsion_weight, boids.attractors_cohesion_weight, boids.attractors_repulsion_weight)
        label = font.render(weights_str, True, (255, 255, 0))
        screen.blit(label, (5, 25))

        for i, position in enumerate(boids.positions):
            """
            rotation
            x'=rcos(α+β) = r(cosαcosβ-sinαsinβ) = xcosβ-ysinβ
            y'=rsin(α+β) = r(sinαcosβ-cosαsinβ) = xsinβ+ycosβ
            """
            R = np.array([[np.cos(boids.angles[i]), -np.sin(boids.angles[i])],
                          [np.sin(boids.angles[i]), np.cos(boids.angles[i])]])
            triangle = np.array([boids.positions[i] + R @ p for p in triangle_points])

            # draw triangles
            pygame.draw.polygon(screen, boids.display_colors[i], triangle)

        for position in boids.obstacles[:boids.num_obstacles]:
            pygame.draw.circle(screen, (255, 0, 0), position, 10)

        for position in boids.attractors[:boids.num_attractors]:
            pygame.draw.circle(screen, (0, 0, 255), position, 10)

        pygame.display.update()

        buttons = pygame.mouse.get_pressed()
        keys = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                position = pygame.mouse.get_pos()
                if buttons[0]:
                    boids.add_boid(position)
                if buttons[1]:
                    boids.add_attractor(position)
                if buttons[2]:
                    boids.add_obstacle(position)

            if event.type == pygame.KEYUP:
                if keys[pygame.K_q]:
                    boids.cohesion_weight -= w
                if keys[pygame.K_w]:
                    boids.cohesion_weight += w
                if keys[pygame.K_a]:
                    boids.alignment_weight -= w
                if keys[pygame.K_s]:
                    boids.alignment_weight += w
                if keys[pygame.K_z]:
                    boids.separation_weight -= w
                if keys[pygame.K_x]:
                    boids.separation_weight += w
                if keys[pygame.K_e]:
                    boids.obstacles_repulsion_weight -= 1
                if keys[pygame.K_r]:
                    boids.obstacles_repulsion_weight += 1
                if keys[pygame.K_d]:
                    boids.attractors_cohesion_weight -= w
                if keys[pygame.K_f]:
                    boids.attractors_cohesion_weight += w
                if keys[pygame.K_c]:
                    boids.attractors_repulsion_weight -= w
                if keys[pygame.K_v]:
                    boids.attractors_repulsion_weight += w

            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

        boids.update()


if __name__ == '__main__':
    main()
