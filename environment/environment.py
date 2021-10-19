#!/usr/bin/env python3

import math

from pysim2d import pysim2d
from .environment_fitness import FitnessData
from .environment_node import Node

import action_mapper 

import numpy as np

class Environment:
    """
    Class as a wrapper for the Simulation2D.
    """

    def __init__(self, path_to_world, test_num):
        """
        Constructor to initialize the environment.
        :param path_to_world: The path to the world which should be selected.
        """
        self._env = pysim2d.pysim2d()
        self._fitness_data = FitnessData()
        self._cluster_size = 1
        self._observation_rotation_size = 64
        self._observation_rotation_use = False

        '''
        Adjacent matrix
        '''
        self._observation_adjacent_size = 100
        self._observation_adjacent_use = False

        self._test_num = test_num

        self._laser_min_distance = 1
        self._laser_min_distance_angle = 0

        self._laser_max_distance = 0
        self._laser_max_distance_angle = 0

        self._n_step = 0 
        self._n_episode = 0

        if not self._fitness_data.init(path_to_world + ".node"):
            print("Error: Load node file! -> " + path_to_world + ".node")
            exit(1)
        if not self._env.init(path_to_world + ".world"):
            print("Error: Load world file -> " + path_to_world + ".world")
            exit(1)

    def get_observation_rotation_size(self):
        return self._observation_rotation_size


    def set_observation_rotation_size(self, size):
        """
        Set the vector size for the rotation.
        :param size: size of the observation rotation vector.
        :return:
        """
        if size < 8:
            print("Warn: Observation rotation size is to low -> set to 8!")

            self._observation_rotation_size = 8
        else:
            self._observation_rotation_size = size

    def use_observation_rotation_size(self, use=True):
        """
        Flag for using the rotation observation vector. The vector is added to the laserscan observation. The rotation
        is decode in a vector like a compass. Dependent on the orientation of the robot to the target, the vector is
        filled with zero and a one by the orientation.

        Example:
        vector size:      8
        Target direction: 4
        vector values:    [0,0,0,0,1,0,0,0]

        1    2    3
             |
        0----+--->4
             |
        7    6    5

        :param use: True for using the rotation observation.
        :return:
        """
        self._observation_rotation_use = use

    def _get_observation(self):
        """
        Get the observation from the laserscan and plus the observation rotation when activated.
        :return: Observation vector
        """
        size = self._env.observation_size()
        observation = []

        self._laser_min_distance = 1
        self._laser_min_distance_angle = 0

        self._laser_max_distance = 0
        self._laser_max_distance_angle = 0

        for i in range(size):
            read = self._env.observation_at(i)

            if read < self._laser_min_distance:
                self._laser_min_distance = read
                self._laser_min_distance_angle = ((i*0.25)-135.25)

            if read > self._laser_max_distance:
                self._laser_max_distance = read
                self._laser_max_distance_angle = ((i*0.25)-135.25)

            observation.append(read)     

        #print (self._laser_min_distance)
        #print (self._laser_min_distance_angle)

        return observation

    def _get_observation_min_clustered(self):
        """
        Get the observation size of the vector for clustering.
        :return: Observation vector size.
        """
        size = self._env.observation_min_clustered_size(self._cluster_size)
        observation = []

        for i in range(size):
            observation.append(self._env.observation_min_clustered_at(i, self._cluster_size))

        return observation

    def set_cluster_size(self, size):
        """
        Set the clustering size for the laserscan vector. The cluster size is the number how many lasers are in a
        cluster.
        :param size: Size of laser in a cluster.
        :return:
        """
        self._cluster_size = size

    def observation_size(self):
        """
        Get the observation vector size.
        :return: Observation vector size.
        """
        if self._cluster_size < 2:
            size = self._env.observation_size()
        else:
            size = self._env.observation_min_clustered_size(self._cluster_size)

        if self._observation_rotation_use:
            size += self._observation_rotation_size

        '''
        Adjacent Matrix
        '''
        if self._observation_adjacent_use:
            size += self._observation_adjacent_size

        return size

    #def visualize(self):
    def render(self, mode='human', close=False):
        """
        Visualize the current state of the simulation with gnuplot.
        :return:
        """
        end_node = self._fitness_data.get_end_node()
        self._env.visualize(end_node.x(), end_node.y(), end_node.radius())

    #def step(self, linear_velocity: float, angular_velocity: float, skip_number: int = 1):
    def step(self, action):
        #action = np.argmax(action)
        #print(action)
        linear_velocity, angular_velocity = action_mapper.map_action(action)
        #linear_velocity = np.clip(float(action[0]), 0.3, -0.3)
        #angular_velocity = np.clip(float(action[1]), 0.2, -0.2)
        skip_number = 0

        """
        Execute a step in the simulation with the given angular and linear velocity. Return the observation, reward and
        done. If done the robot reach the goal or collided with an object.
        :param linear_velocity: Linear veloctiy of the robot.
        :param angular_velocity: Angular velocity of the robot.
        :param skip_number: Number of laserscan to skip until return.
        :return: observation, reward, done, message
        """
        self._env.step(linear_velocity, angular_velocity, skip_number)
        env_robot_x = self._env.get_robot_pose_x()
        env_robot_y = self._env.get_robot_pose_y()
        env_robot_orientation = self._env.get_robot_pose_orientation()
        env_done = self._env.done()

        reward, done = self._fitness_data.calculate_reward(env_robot_x,
                                                           env_robot_y,
                                                           env_robot_orientation,
                                                           env_done,
                                                           self._laser_min_distance,
                                                           self._laser_min_distance_angle,
                                                           self._laser_max_distance,
                                                           self._laser_max_distance_angle,
                                                           self._test_num,
                                                           self._n_step,
                                                           self._n_episode)
        self._n_step += 1

        if self._cluster_size < 2:
            observation = self._get_observation()
        else:
            observation = self._get_observation_min_clustered()
            
        if self._observation_rotation_use:
            not_set = True

            angle_target = self._fitness_data.angle_difference_from_robot_to_end(env_robot_x, env_robot_y, env_robot_orientation)
            angle_step_size = 2 * math.pi / self._observation_rotation_size
            angle_sum = - math.pi + angle_step_size

            for i in range(self._observation_rotation_size):
                if not_set and angle_target < angle_sum:
                    observation.append(1.0)
                    not_set = False
                else:
                    observation.append(0.0)

                angle_sum += angle_step_size

        if self._observation_adjacent_use:
            '''
            Adjacent Matrix
            '''
            weight = 2
            '''
            proto 2
            '''
            '''
            am = np.array([
                0,1,1,0,0,0,0,0,0,0,
                1,0,0,0,0,0,0,0,0,0,
                1,0,0,1,0,0,0,0,0,0,
                0,0,1,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,0,0
                ])
            '''

            

            am = np.array([
                0,1,1,0,0,0,0,0,0,0,
                1,0,0,0,0,0,0,0,0,0,
                1,0,0,1,1,0,0,0,0,0,
                0,0,1,0,0,1,0,0,0,0,
                0,0,1,0,0,0,0,0,0,0,
                0,0,0,1,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,0,0
                ])
            
            am = am * weight
            
            for j in range(self._observation_adjacent_size):
                observation.append(am[j])  

        return observation, reward, done, {}

    def _classify(self, observation):
        for i in range(len(observation)):
            if observation[i] < 0.1:
                observation[i] = 1
            elif observation[i] < 0.2:
                observation[i] = 2
            elif observation[i] < 0.3:
                observation[i] = 3
            elif observation[i] < 0.4:
                observation[i] = 4
            elif observation[i] < 0.5:
                observation[i] = 5
            elif observation[i] < 0.6:
                observation[i] = 6
            elif observation[i] < 0.7:
                observation[i] = 7
            elif observation[i] < 0.8:
                observation[i] = 8
            elif observation[i] < 0.9:
                observation[i] = 9
            else:
                observation[i] = 10

        return observation


    def reset(self):
        """
        Reset the simulation. Put the robot to the (new) start position and the the (new) target position depending on
        the selected mode.
        :return:
        """
        self._fitness_data.reset()
        x, y, orientation = self._fitness_data.get_robot_start()
        self._env.set_robot_pose(x, y, orientation)
        
        #A = np.array([0.0,0.0])
        #return self.step(A)
        self._n_step = 0
        self._n_episode += 1
        
        return self.step(11)
        
        #return self.step(0.0, 0.0)
        

    def set_mode(self, mode, terminate_at_end=True):
        """
        Set the mode for the simulation. The mode defines the selection of the start and end node. Nodes with the same
        id are in pairs.

        *** Modes ***
        ALL_COMBINATION: Take all possible combination from start and end node. Ignore the node id.
        ALL_RANDOM: Take randomly a start and end node. Ignore the node id.
        PAIR_ALL: Take all pair combination and select from the pair a randomly start and end node.
        PAIR_RANDOM: Take randomly a pair and select from the pair a randomly start and end node.
        CHECKPOINT: Take the first start node from the lowest id and the the target node from the next higher node id.
                    When reaching the target node select the next higher id until the highest id is reached.
        :param mode: Simulation mode.
        :param terminate_at_end: Done when the target node is reached.
        :return:
        """
        self._fitness_data.set_mode(mode, terminate_at_end)