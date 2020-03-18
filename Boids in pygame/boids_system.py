#!/usr/bin/env python
# -*-coding:utf-8 -*-
# author: Sympathyzzk  time:2020/3/18.

import numpy as np
from scipy.spatial import cKDTree as kd_tree

# the standard colors array
colors = np.array([(255, 0, 0),
                   (0, 255, 0),
                   (0, 0, 255),
                   (255, 255, 0),
                   (0, 255, 255),
                   (255, 0, 255),
                   (255, 255, 255)])


class boids_system(object):
    def __init__(self, num_boids, neighborhood_size, screen_size):
        """set the macroscopic size"""
        # set the number of birds
        self.num_boids = num_boids
        # set the observe area of boids' neighborhoods
        self.neighborhood_size = neighborhood_size
        # set the screen_size of foursquare,it's height is equal to width
        self.screen_size = screen_size

        """
        set the initial states of birds:
            positions,angles,max speed,velocities,individual colors and it's display colors
        """
        self.positions = np.random.uniform(10, screen_size - 10, (self.num_boids, 2))
        self.angles = np.random.uniform(0, 2 * np.pi, self.num_boids)
        self.max_speed = 2.0
        # get the random speed and it's component number in x and y axises
        r = np.random.uniform(1.0, 2.0, self.num_boids)
        x, y = r * np.cos(self.angles), r * np.sin(self.angles)
        # get the velocities array
        self.velocities = np.array(list(zip(x, y)))
        self.colors = colors[np.random.randint(0, len(colors), self.num_boids)]
        # they will never influence each other
        self.display_colors = np.copy(self.colors)

        """different weight of all kinds of force"""
        self.cohesion_weight = 1.0
        self.alignment_weight = 1.0
        self.separation_weight = 1.0
        self.obstacles_repulsion_weight = 5.0
        self.attractors_cohesion_weight = 0.4
        self.attractors_repulsion_weight = 0.3

        """obstacles"""
        self.num_obstacles = 0
        self.max_num_obstacles = 50
        self.obstacles = np.empty((self.max_num_obstacles, 2), dtype=np.int32)

        """attractors"""
        self.num_attractors = 0
        self.max_num_attractors = 50
        self.attractors = np.empty((self.max_num_attractors, 2), dtype=np.int32)

    def update(self):
        # update the positions,angles
        self.positions += self.velocities
        self.angles = np.arctan2(self.velocities[:, 1], self.velocities[:, 0])

        # get the location reactivities(remove itself) of birds
        tree = kd_tree(self.positions)
        self.close_pairs = tree.query_ball_tree(tree, r=self.neighborhood_size)
        for i in range(self.num_boids):
            self.close_pairs[i].remove(i)

        # get the location reactivities of obstacles
        if self.num_obstacles > 0:
            self.tree_obstacles = kd_tree(self.obstacles)
            self.close_pairs_obstacles = tree.query_ball_tree(self.tree_obstacles, r=20)

        # get the location reactivities of attractors
        if self.num_attractors > 0:
            self.tree_attractors = kd_tree(self.attractors)
            self.close_pairs_attractors = tree.query_ball_tree(self.tree_attractors, r=100)
            self.closest_pairs_attractors = tree.query_ball_tree(self.tree_attractors, r=50)

        # calculate the updated velocities
        cohesion = self.cohesion_weight * self.get_cohesion()
        alignment = self.alignment_weight * self.get_alignment()
        separation = self.separation_weight * self.get_separation()
        self.velocities += cohesion + alignment + separation

        if self.num_obstacles > 0:
            self.velocities += self.obstacles_repulsion_weight * self.get_obstacles_repulsion()

        if self.num_attractors > 0:
            attractors_cohesion = self.attractors_cohesion_weight * self.get_attractors_cohesion()
            attractors_repulsion = self.attractors_repulsion_weight * self.get_attractors_repulsion()
            self.velocities += attractors_cohesion + attractors_repulsion

        # limit the positions in screen and velocities
        self.positions = np.mod(self.positions, self.screen_size)
        self.limit_velocity()

        # update colors
        self.update_colors()

    def get_cohesion(self):
        cohesion = np.zeros((self.num_boids, 2))

        for i in range(self.num_boids):
            num_neighbors = len(self.close_pairs[i])

            if num_neighbors > 0:
                mass_center = np.mean(self.positions[self.close_pairs[i]], axis=0)
                cohesion[i] = mass_center - self.positions[i]
                norm = np.linalg.norm(cohesion[i])
                cohesion[i] /= norm

        return cohesion

    def get_alignment(self):
        alignment = np.zeros((self.num_boids, 2))

        for i in range(self.num_boids):
            num_neighbors = len(self.close_pairs[i])

            if num_neighbors > 0:
                alignment[i] = np.sum(self.velocities[self.close_pairs[i]], axis=0)
                norm = np.linalg.norm(alignment[i])
                alignment[i] /= norm

        return alignment

    def get_separation(self):
        separation = np.zeros((self.num_boids, 2))

        for i in range(self.num_boids):
            num_neighbors = len(self.close_pairs[i])

            if num_neighbors > 0:
                separation[i] = num_neighbors * self.positions[i] - \
                                np.sum(self.positions[self.close_pairs[i]], axis=0)
                norm = np.linalg.norm(separation[i])
                separation[i] /= norm

        return separation

    def update_colors(self):
        num_boids = self.num_boids
        display_colors = np.copy(self.colors[:num_boids])

        for k in range(5):
            for i in range(num_boids):
                num_neighbors = len(self.close_pairs[i])

                if num_neighbors > 0:
                    display_colors[i] = np.mean(display_colors[self.close_pairs[i]], axis=0)

        self.display_colors = display_colors

    def limit_velocity(self):
        for i, velocity in enumerate(self.velocities[:self.num_boids]):
            norm = np.linalg.norm(velocity)
            if norm > self.max_speed:
                velocity *= self.max_speed / norm
                self.velocities[i] = velocity

    def get_obstacles_repulsion(self):
        separation = np.zeros((self.num_boids, 2))

        for i in range(self.num_boids):
            num_close_obstacles = len(self.close_pairs_obstacles[i])

            if num_close_obstacles > 0:
                mass_center = np.mean(self.obstacles[self.close_pairs_obstacles[i]], axis=0)
                separation[i] = self.positions[i] - mass_center
                norm = np.linalg.norm(separation[i])
                separation[i] /= norm

        return separation

    def get_attractors_cohesion(self):
        cohesion = np.zeros((self.num_boids, 2))

        for i in range(self.num_boids):
            num_close_attractors = len(self.close_pairs_attractors[i])

            if num_close_attractors > 0:
                mass_center = np.mean(self.attractors[self.close_pairs_attractors[i]], axis=0)
                cohesion[i] = mass_center - self.positions[i]
                norm = np.linalg.norm(cohesion[i])
                cohesion[i] /= norm

        return cohesion

    def get_attractors_repulsion(self):
        repulsion = np.zeros((self.num_boids, 2))

        for i in range(self.num_boids):
            num_close_attractors = len(self.closest_pairs_attractors[i])

            if num_close_attractors > 0:
                mass_center = np.mean(self.attractors[self.closest_pairs_attractors[i]], axis=0)
                repulsion[i] = self.positions[i] - mass_center
                norm = np.linalg.norm(repulsion[i])
                repulsion[i] /= norm

        return repulsion

    def add_boid(self, position):
        self.positions = np.vstack((self.positions, position))

        theta = np.random.uniform(0, 2 * np.pi)

        # horizontal
        self.angles = np.hstack((self.angles, theta))

        r = np.random.uniform(0, 2.0)
        velocity = r * np.array([np.cos(theta), np.sin(theta)])
        self.velocities = np.vstack((self.velocities, velocity))

        color = colors[np.random.randint(0, len(colors))]
        self.colors = np.vstack((self.colors, color))

        self.num_boids += 1

    def add_attractor(self, position):
        if self.num_attractors == self.max_num_attractors:
            print("Limit reached")
            return

        self.attractors[self.num_attractors] = position
        self.num_attractors += 1

    def add_obstacle(self, position):
        if self.num_obstacles == self.max_num_obstacles:
            print("Limit reached")
            return

        self.obstacles[self.num_obstacles] = position
        self.num_obstacles += 1
