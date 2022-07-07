#!/usr/bin/env python3

import math
import time
from .environment_node_data import NodeData, Mode
from math import sqrt
from collections import deque

import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy.ndimage import gaussian_filter

from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel, AiryDisk2DKernel


DEBUG = False

class FitnessData:
    """
    Class for calculating the fitness function reward from the current simulation state. Possible reward calculation
    are: travelled distance, distance to the target node, angle to the target node etc.
    """

    def __init__(self):
        """
        Constructor of the FitnessData class.
        """
        self._node_data = NodeData()

        self._terminate_at_end = True

        self._robot_x_last = 0.0
        self._robot_y_last = 0.0
        self._robot_orientation_last = 0.0

        self._x_max = 0
        self._y_max = 0

        self._x_last = 0
        self._y_last = 0

        self._distance_avg = 0.0
        self._distance_avg_last = 0.0

        self._reward_matrix = np.array( [ [] , [] ] )
        self._path_matrix = np.array( [ [] , [] ] )

        self.LOAD_MATRIX = False

        self._create_matrix = False
        self._create_path_matrix = False

        self._distance_end = 0
        self._gaussian_count = 0
        self._decaiment_count = 0
        self._final_reward = 2_000

        self.EXTEND_AREA = 2.0
        self._invert_plot = True

        self._grid_matrix = np.array( [ [] , [] ] )
        self._path = []

        #self.fig, (self.ax1, self.ax2) = plt.subplots(2)

    def get_end_node(self):
        """
        Get the current selected end Node.
        :return:
        """
        return self._node_data.get_node_end()

    def init(self, filename="") -> bool:
        """
        Read the node file and initialized the node data.
        :param filename:
        :return:
        """
        return self._node_data.read_node_file(filename)

    def set_mode(self, mode: Mode, terminate_at_end=True):
        """
        Set the mode for the simulation. More information in environment.py.
        :param mode: Simulation mode.
        :param terminate_at_end: Done when the target node is reached.
        :return:
        """
        self._node_data.set_mode(mode)
        self._terminate_at_end = terminate_at_end

    def _distance_start_to_end(self) -> float:
        """
        Calculate the distance between the start node and end node.
        :return: Distance between start and end node.
        """
        return self._distance(
            self._node_data.get_node_start().x(),
            self._node_data.get_node_start().y(),
            self._node_data.get_node_end().x(),
            self._node_data.get_node_end().y())

    def _distance_robot_to_end(self, robot_x: float, robot_y: float) -> float:
        """
        Calculate the distance between the robot position to the end node.
        :param robot_x: Robot position x.
        :param robot_y: Robot position y.
        :return: Distance between robot and end node.
        """
        return self._distance(
            robot_x,
            robot_y,
            self._node_data.get_node_end().x(),
            self._node_data.get_node_end().y())

    def _distance_between_last_step(self, robot_x: float, robot_y: float) -> float:
        """
        Calculate the distance from the last robot position to the current robot position.
        :param robot_x: Robot position x.
        :param robot_y: Robot position y.
        :return: Distance from last and current robot position.
        """
        return self._distance(
            robot_x,
            robot_y,
            self._robot_x_last,
            self._robot_y_last)

    def _orientation_robot_to_end(self, robot_x: float, robot_y: float) -> float:
        """
        Calculate the orientation from the robot position to the end node.
        :param robot_x: Robot position x.
        :param robot_y: Robot position y.
        :return: Orientation from robot position to end node.
        """
        x = self._node_data.get_node_end().x() - robot_x
        y = self._node_data.get_node_end().y() - robot_y
        return math.atan2(y, x)

    def angle_difference_from_robot_to_end(self, robot_x: float, robot_y: float, robot_orientation: float) -> float:
        """
        Calculate the angle difference from robot to end node.
        :param robot_x: Robot position x.
        :param robot_y: Robot position y.
        :return: Angle difference robot to end node.
        """
        return self._difference_two_angles(robot_orientation, self._orientation_robot_to_end(robot_x, robot_y))

    @staticmethod
    def _difference_two_angles(angle1: float, angle2: float) -> float:
        """
        Calculate the difference between to angle.
        :param angle1: First angle.
        :param angle2: Second angle.
        :return: Difference between to angle.
        """
        diff = (angle1 - angle2) % (math.pi * 2)
        if diff >= math.pi:
            diff -= math.pi * 2
        return diff

    @staticmethod
    def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
        """
        Calculate the euler distance from to point.
        :param x1: First point x.
        :param y1: First point y.
        :param x2: Second point x.
        :param y2: Second point y.
        :return: Euler distnace from to points.
        """
        x = x1 - x2
        y = y1 - y2
        return math.sqrt(x*x + y*y)

    def reset(self):
        """
        Reset call from the environment to select new node.
        :return:
        """
        self._create_path_matrix = False
        self._path = []
        self._final_reward = 2_000
        self._decaiment_count = 0

        self._invert_plot = True

        if(self.LOAD_MATRIX == True):
            self.load_matrix()

        self._node_data.new_node_selection()

    def calculate_reward(self, robot_x: float, robot_y: float, 
        robot_orientation: float, env_done: bool,
        observation: [],
        test_num: int,
        n_step: int,
        n_episode: int):
        """
        Calculate the reward from one step.
        :param robot_x: Robot position x.
        :param robot_y: Robot position y.
        :param robot_orientation: Robot orientation angle.
        :param env_done: Done when the robot collided with the robot.
        :return: Reward, done.
        """
        done = False
        reward = 0.0

        REWARD_MATRIX_START = 1
        REWARD_MATRIX_HALL = 0
        PATH_MATRIX_OK = 5
        xy_resolution = 0.05

        '''
        Verifica estado
        '''
        distance_robot_to_end = self._distance_robot_to_end(robot_x, robot_y)

        if env_done:
            reward = 0
            done = True

        elif n_step == 200:
            reward = 0
            done = True

        elif distance_robot_to_end < self._node_data.get_node_end().radius():
            print('Wir haben gewonnen')
            length = len(self._path)
            for i in range(length):
                if(i % 2 == 0):
                    if(self._grid_matrix.shape[0] > self._path[i] and self._grid_matrix.shape[1] > self._path[i+1]):
                        self._grid_matrix[self._path[i], self._path[i+1]] = PATH_MATRIX_OK
                else:
                    continue

            if(self.LOAD_MATRIX == False):
                np.save('matrices/_grid_matrix_'+str(test_num)+'.npy', self._grid_matrix)

            reward = 2_000 + self._final_reward

            done = self._handle_terminate_at_end()

        '''
        Carrega grid
        '''
        if(self.LOAD_MATRIX == True):
            self._grid_matrix = np.load('matrices/_grid_matrix_'+str(test_num)+'.npy', allow_pickle=True)
        '''
        Verifica distâncias angular e linear
        '''
        alpha = 2.0
        ef = self._distance(self._node_data.get_node_end().x(), self._node_data.get_node_end().y(), self._node_data.get_node_start().x(), self._node_data.get_node_start().y())
        r1 = (ef/(ef + distance_robot_to_end)) #- 0.5
        r1 = r1*alpha #[0 <-> alpha]


        theta = 1.5
        x = self._node_data.get_node_end().x() - robot_x
        y = self._node_data.get_node_end().y() - robot_y
        diff = (robot_orientation - math.atan2(y, x)) % (math.pi * 2) #[0 <-> 2pi]
        if diff >= math.pi:
            diff -= math.pi * 2
            diff = abs(diff)

        r3 = 1 - (diff/math.pi)   #[0 <-> 1]
        r3 = r3 * theta  #[0 <-> theta]

        '''
        Gera recompensa
        '''
        if(self._grid_matrix.shape[0] > self._x_last and self._grid_matrix.shape[1] > self._y_last):

            reward = (r1 * self._grid_matrix[self._x_last, self._y_last])
            reward += (r3 * self._grid_matrix[self._x_last, self._y_last])
            reward = reward - (self._decaiment_count * 0.1)

            if(reward >= 0):
                self._final_reward = self._final_reward - reward                    
            else:
                self._final_reward = self._final_reward + reward
                   
            self._decaiment_count = self._decaiment_count + 1

        self._robot_x_last = robot_x
        self._robot_y_last = robot_y
        self._robot_orientation_last = robot_orientation

        if(self.LOAD_MATRIX == False):
            '''
            Gerência Matriz
            '''
            ox = (np.sin(math.radians(90) - self._robot_orientation_last) * (observation[540]/0.05)) + self._robot_x_last
            oy = (np.cos(math.radians(90) - self._robot_orientation_last) * (observation[540]/0.05)) + self._robot_y_last

            xw  = int(round( ox / xy_resolution))
            yw  = int(round( oy / xy_resolution))

            self._x_last = int(round( self._robot_x_last / xy_resolution))
            self._y_last = int(round( self._robot_y_last / xy_resolution))

            if self._create_matrix == False:

                self._grid_matrix = np.full((xw+1, yw+1), REWARD_MATRIX_START, dtype=float)

                self._create_matrix = True

                self._x_max = xw
                self._y_max = yw

                print ('self._x_max: {}'.format(self._x_max))
                print ('self._y_max: {}'.format(self._y_max))

            else:

                if (xw > self._x_max):
                    x = xw
                    y = self._y_max + 1
                    _x = x - self._x_max

                    _matrix_temp = np.full((_x, y), REWARD_MATRIX_START, dtype=float)
                    self._grid_matrix = np.append( self._grid_matrix, _matrix_temp, axis=0)

                    self._x_max = xw

                if (yw > self._y_max):
                    x = self._x_max + 1 
                    y = yw
                    _y = y - self._y_max

                    _matrix_temp = np.full((x, _y), REWARD_MATRIX_START, dtype=float)
                    self._grid_matrix = np.append( self._grid_matrix, _matrix_temp, axis=1)

                    self._y_max = yw

            self._grid_matrix[xw-1, yw-1] = REWARD_MATRIX_HALL
            self._path.append(self._x_last)
            self._path.append(self._y_last)

            '''
            Convolução
            '''
            if self._gaussian_count == 25_000:
                gauss_kernel = AiryDisk2DKernel(1)
                smoothed_data_gauss = convolve(self._grid_matrix, gauss_kernel)
                self._grid_matrix = smoothed_data_gauss
                self._gaussian_count = 0
            else:
                self._gaussian_count += 1
        else:
            self._x_last = int(round( self._robot_x_last / xy_resolution))
            self._y_last = int(round( self._robot_y_last / xy_resolution))

        return reward, done

    def _distance_robot_to_start(self, robot_x: float, robot_y: float) -> float:
        """
        Calculate the distance between the robot position to the end node.
        :param robot_x: Robot position x.
        :param robot_y: Robot position y.
        :return: Distance between robot and end node.
        """
        return self._distance(
            robot_x,
            robot_y,
            self._node_data.get_node_start().x(),
            self._node_data.get_node_start().y())

    def get_robot_start(self):
        """
        Get the start node for the start position for the robot.
        :return: Start node.
        """
        return self._node_data.generate_robot_start_position()

    def _handle_terminate_at_end(self):
        """
        Select a new target node when the simulation souldn't terminate when reaching the target node.
        :return:
        """
        if not self._terminate_at_end:
            self._node_data.new_end_node()

        return self._terminate_at_end

