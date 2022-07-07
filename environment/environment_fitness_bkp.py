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
        self._final_reward = 20_000

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

    def load_matrix(self):
        #self._reward_matrix = np.load('matrices/reward_matrix_'+str(test_num)+'.npy', allow_pickle=True)
        self._reward_matrix = np.load('matrices/reward_matrix_9.npy', allow_pickle=True)
        _x, _y = self._reward_matrix.shape
        self._x_max = _x
        self._y_max = _y

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
        '''
        #Abstand zwinschen P(t-1) und P(t) --> #1
        distance_between_last_step = self._distance_between_last_step(robot_x, robot_y)

        #Abstand zwinschen P(t) und Endposition -->#2
        distance_robot_to_end = self._distance_robot_to_end(robot_x, robot_y)

        #Abstand zwinschen P(t-1) und Endposition -->#3
        distance_robot_to_end_last = self._distance_robot_to_end(self._robot_x_last, self._robot_y_last)

        # -->#4
        distance_robot_to_end_diff = distance_robot_to_end_last - distance_robot_to_end;
        # -->#5
        distance_robot_to_end_diff_abs = abs(distance_robot_to_end_diff)

        diff_rotation_to_end_last = self.angle_difference_from_robot_to_end(self._robot_x_last, self._robot_y_last,self._robot_orientation_last)

        diff_rotation_to_end = self.angle_difference_from_robot_to_end(robot_x, robot_y,robot_orientation)

        rotations_cos_sum =  math.cos(diff_rotation_to_end) #  [ -1 , 1]

        diff_rotations = math.fabs(math.fabs(diff_rotation_to_end_last) - math.fabs(diff_rotation_to_end))   # [0 , pi]

        if distance_between_last_step != 0:
            distance_robot_to_end_diff_abs = distance_robot_to_end_diff_abs/distance_between_last_step # Normalization to [0 , 1]
        else:
            distance_robot_to_end_diff_abs = 0

        if distance_robot_to_end > sqrt(distance_between_last_step**2 + distance_robot_to_end_last**2):
            distance_robot_to_end_diff_abs *= -6.0 # [-6,0]
        else:
            distance_robot_to_end_diff_abs *= 6.0 #[0, 6]

        if math.fabs(diff_rotation_to_end) > math.fabs(diff_rotation_to_end_last):

            diff_rotations *= -3.0 # [-3xpi,0]
        else:
            diff_rotations *=  2.0 # [0,2xpi]
        '''
        if(False):

            PATH_MATRIX_START = 0
            PATH_MATRIX_INCREASE = 1
            PATH_MATRIX_UP_LIMITE = 20
            REWARD_MATRIX_START = -1
            REWARD_MATRIX_CRITICAL_0 = 0
            REWARD_MATRIX_CRITICAL_1 = 1
            REWARD_MATRIX_CRITICAL_2 = 2
            REWARD_MATRIX_CRITICAL_3 = 3
            REWARD_MATRIX_CRITICAL_4 = 4
            REWARD_MATRIX_CRITICAL_5 = 5
            REWARD_MATRIX_SAFETY = 6

            if self._create_matrix == False:
                x = robot_x * 10
                y = robot_y * 10

                x = int(x)
                y = int(y)

                print ('x: {}'.format(x))
                print ('y: {}'.format(y))

                print ('robot_x: {}'.format(robot_x))
                print ('robot_y: {}'.format(robot_y))

                if(self.LOAD_MATRIX == False):
                    self._reward_matrix = np.full((x+1, y+1), REWARD_MATRIX_START, dtype=float)
                    self._x_max = x+1
                    self._y_max = y+1
                else:
                    self.load_matrix()

                print ('self._x_max: {}'.format(self._x_max))
                print ('self._y_max: {}'.format(self._y_max))

                self._create_matrix = True

                self._x_last = x
                self._y_last = y
            else:
                x = robot_x * 10
                y = robot_y * 10
                x = int(x)
                y = int(y)

                if (int(robot_x * 10)+1) > self._x_max:
                    x = int(robot_x * 10) + 1
                    y = int(self._y_max)

                    _x = x - int(self._x_max)

                    _reward_matrix_temp = np.full((_x, y), REWARD_MATRIX_START, dtype=float)
                    self._reward_matrix = np.append( self._reward_matrix, _reward_matrix_temp, axis=0)

                    _path_matrix_temp = np.full((_x, y), PATH_MATRIX_START, dtype=float)
                    self._path_matrix = np.append( self._path_matrix, _path_matrix_temp, axis=0)
                    
                    self._x_max = int(robot_x*10) + 1

                    #print('self._x_max: {}'.format(self._x_max))

                if (int(robot_y * 10)+1) > self._y_max:
                    x = int(self._x_max)
                    y = int(robot_y * 10) + 1

                    _y = y - int(self._y_max)

                    _reward_matrix_temp = np.full((x, _y), REWARD_MATRIX_START, dtype=float)
                    self._reward_matrix = np.append( self._reward_matrix, _reward_matrix_temp, axis=1)

                    _path_matrix_temp = np.full((x, _y), PATH_MATRIX_START, dtype=float)
                    self._path_matrix = np.append( self._path_matrix, _path_matrix_temp, axis=1)

                    self._y_max = int(robot_y*10) + 1

                    #print('self._y_max: {}'.format(self._y_max))

                if n_step == 0:
                    self._x_last = int(robot_x * 10)
                    self._y_last = int(robot_y * 10)

            if self._create_path_matrix == False:
                self._path_matrix = np.full((self._x_max, self._y_max), PATH_MATRIX_START, dtype=float)
                self._create_path_matrix = True


            if env_done:
                self._reward_matrix[self._x_last, self._y_last] = REWARD_MATRIX_CRITICAL_0
                reward = 0
                done = True
                #if(self.LOAD_MATRIX == False):
                #    np.save('matrices/reward_matrix_'+str(test_num)+'.npy', self._reward_matrix)
                #np.save('matrices/path_matrix_'+str(test_num)+'.npy', self._path_matrix)
                #print(" Reward: " + str(reward))

            elif n_step == 200:
                reward = 0
                done = True
                #print("Reward: " + str(reward))

            elif distance_robot_to_end < self._node_data.get_node_end().radius():
                reward = 20_000 + self._final_reward
                
                if(self.LOAD_MATRIX == True):
                    self.load_matrix()  
                
                self._reward_matrix = np.add(self._reward_matrix, self._path_matrix)

                print('Wir haben gewonnen')
                done = self._handle_terminate_at_end()
                
                if(self.LOAD_MATRIX == False):
                    np.save('matrices/reward_matrix_'+str(test_num)+'.npy', self._reward_matrix)
                else:
                    np.save('matrices/reward_matrix_9.npy', self._reward_matrix)
                np.save('matrices/path_matrix_'+str(test_num)+'.npy', self._path_matrix)


                #print("Reward: " + str(self._final_reward))
            else:
                if(self.LOAD_MATRIX == False):
                    if (self._x_last != int(robot_x*10) or self._y_last != int(robot_y*10)):
                        #reward matrix
                        if(self._reward_matrix[self._x_last, self._y_last] != REWARD_MATRIX_CRITICAL_5):
                            if(self._reward_matrix[self._x_last, self._y_last] != REWARD_MATRIX_CRITICAL_4):
                                if(self._reward_matrix[self._x_last, self._y_last] != REWARD_MATRIX_CRITICAL_3):
                                    if(self._reward_matrix[self._x_last, self._y_last] != REWARD_MATRIX_CRITICAL_2):
                                        if(self._reward_matrix[self._x_last, self._y_last] != REWARD_MATRIX_CRITICAL_1):
                                            if(self._reward_matrix[self._x_last, self._y_last] != REWARD_MATRIX_CRITICAL_0):
                                                self._reward_matrix[self._x_last, self._y_last] = REWARD_MATRIX_SAFETY
                                            else:
                                                self._reward_matrix[self._x_last, self._y_last] = REWARD_MATRIX_CRITICAL_1
                                        else:
                                            self._reward_matrix[self._x_last, self._y_last] = REWARD_MATRIX_CRITICAL_2
                                    else:
                                        self._reward_matrix[self._x_last, self._y_last] = REWARD_MATRIX_CRITICAL_3
                                else:
                                    self._reward_matrix[self._x_last, self._y_last] = REWARD_MATRIX_CRITICAL_4
                            else:
                                self._reward_matrix[self._x_last, self._y_last] = REWARD_MATRIX_CRITICAL_5
                    
                #if(LOAD_MATRIX == False):
                #    np.save('matrices/reward_matrix_'+str(test_num)+'.npy', self._reward_matrix)
                #np.save('matrices/path_matrix_'+str(test_num)+'.npy', self._path_matrix)

                #alpha = 1.5
                alpha = 5.0
                #Distance between end point and reference point
                ef = self._distance(self._node_data.get_node_end().x(), self._node_data.get_node_end().y(), self._node_data.get_node_start().x(), self._node_data.get_node_start().y())
                r1 = (ef/(ef + distance_robot_to_end)) #- 0.5
                #r1 = (ef/distance_robot_to_end) - 1
                r1 = r1*alpha #[0 <-> alpha]

                #r1 = self._distance_end - distance_robot_to_end

                theta = 5.0

                x = self._node_data.get_node_end().x() - robot_x
                y = self._node_data.get_node_end().y() - robot_y
                diff = (robot_orientation - math.atan2(y, x)) % (math.pi * 2) #[0 <-> 2pi]
                if diff >= math.pi:
                    diff -= math.pi * 2
                    diff = abs(diff)

                r3 = 1 - (diff/math.pi)   #[0 <-> 1]
                r3 = r3 * theta  #[0 <-> theta]

                #print("Matrix: ", self._reward_matrix[self._x_last, self._y_last])
                #print("_path_matrix: ", self._path_matrix[self._x_last, self._y_last])
                '''
                if (self._reward_matrix[self._x_last, self._y_last] >= REWARD_MATRIX_CRITICAL_0 and self._reward_matrix[self._x_last, self._y_last] <= REWARD_MATRIX_CRITICAL_1):
                    #reward = -10
                    reward = -20
                elif (self._reward_matrix[self._x_last, self._y_last] > REWARD_MATRIX_CRITICAL_1 and self._reward_matrix[self._x_last, self._y_last] <= REWARD_MATRIX_CRITICAL_2):
                    #reward = -8
                    reward = -18
                elif (self._reward_matrix[self._x_last, self._y_last] > REWARD_MATRIX_CRITICAL_2 and self._reward_matrix[self._x_last, self._y_last] <= REWARD_MATRIX_CRITICAL_3):
                    #reward = -7
                    reward = -17
                elif (self._reward_matrix[self._x_last, self._y_last] > REWARD_MATRIX_CRITICAL_3 and self._reward_matrix[self._x_last, self._y_last] <= REWARD_MATRIX_CRITICAL_4):
                    #reward = -6
                    reward = -16
                elif (self._reward_matrix[self._x_last, self._y_last] > REWARD_MATRIX_CRITICAL_4 and self._reward_matrix[self._x_last, self._y_last] <= REWARD_MATRIX_CRITICAL_5):
                    #reward = -5
                    reward = -15
                else:
                    reward = (r1 * self._reward_matrix[self._x_last, self._y_last])
                    reward += (r1 * self._path_matrix[self._x_last, self._y_last])
                    reward += (r3 * self._reward_matrix[self._x_last, self._y_last])
                    reward += (r3 * self._path_matrix[self._x_last, self._y_last])
                '''
                reward = (r1 * self._reward_matrix[self._x_last, self._y_last])
                #reward += (r1 * self._path_matrix[self._x_last, self._y_last])
                reward += (r3 * self._reward_matrix[self._x_last, self._y_last])
                #reward += (r3 * self._path_matrix[self._x_last, self._y_last])

                '''
                elif self._path_matrix[self._x_last, self._y_last] < 0:
                    reward = (r1 * self._reward_matrix[self._x_last, self._y_last])
                    reward += (r1 * self._path_matrix[self._x_last, self._y_last])
                    reward += (r3 * self._reward_matrix[self._x_last, self._y_last])
                    reward += (r3 * self._path_matrix[self._x_last, self._y_last])
                else:
                    reward = (r1 * self._reward_matrix[self._x_last, self._y_last] * self._path_matrix[self._x_last, self._y_last])
                    reward += (r3 * self._reward_matrix[self._x_last, self._y_last] * self._path_matrix[self._x_last, self._y_last])
                '''
                #pathmatrix

                if(self._path_matrix[self._x_last, self._y_last] + self._reward_matrix[self._x_last, self._y_last] < PATH_MATRIX_UP_LIMITE):
                    self._path_matrix[self._x_last, self._y_last] += PATH_MATRIX_INCREASE                

                reward = reward - (self._decaiment_count * 0.1)

                if(reward >= 0):
                    self._final_reward = self._final_reward - reward                    
                else:
                    self._final_reward = self._final_reward + reward
               

            self._x_last = int(robot_x*10)
            self._y_last = int(robot_y*10) 

            if self._gaussian_count == 50_000:
                gauss_kernel = AiryDisk2DKernel(1)
                smoothed_data_gauss = convolve(self._reward_matrix, gauss_kernel)
                self._reward_matrix = smoothed_data_gauss
                self._gaussian_count = 0
            else:
                self._gaussian_count += 1

            self._decaiment_count = self._decaiment_count + 1

            #print("Reward: " + str(reward))
            #print("self._final_reward: " + str(self._final_reward))
        else:

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
                            self._grid_matrix[self._path[i], self._path[i+1]] = 12
                    else:
                        continue

                np.save('matrices/_grid_matrix.npy', self._grid_matrix)
                done = self._handle_terminate_at_end()

        self._robot_x_last = robot_x
        self._robot_y_last = robot_y
        self._robot_orientation_last = robot_orientation
        #self.grid_map_test(observation)
        self.grid(observation)

        return reward, done

    def grid(self, observation):
        
        REWARD_MATRIX_START = 6
        PATH_MATRIX_START = 0
        xy_resolution = 0.1

        ox = (np.sin(math.radians(90) - self._robot_orientation_last) * (observation[540]/0.05)) + self._robot_x_last
        oy = (np.cos(math.radians(90) - self._robot_orientation_last) * (observation[540]/0.05)) + self._robot_y_last

        xw  = int(round( ox / xy_resolution))
        yw  = int(round( oy / xy_resolution))

        self._x_last = int(round( self._robot_x_last / xy_resolution))
        self._y_last = int(round( self._robot_y_last / xy_resolution))

        if self._create_matrix == False:

            self._grid_matrix = np.full((xw+1, yw+1), REWARD_MATRIX_START, dtype=float)
            #self._path_matrix = np.full((xw+1, yw+1), PATH_MATRIX_START, dtype=float)

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

                #_path_matrix_temp = np.full((_x, y), PATH_MATRIX_START, dtype=float)
                #self._path_matrix = np.append( self._path_matrix, _path_matrix_temp, axis=0)

                self._x_max = xw

            if (yw > self._y_max):
                x = self._x_max + 1 
                y = yw
                _y = y - self._y_max

                _matrix_temp = np.full((x, _y), REWARD_MATRIX_START, dtype=float)
                self._grid_matrix = np.append( self._grid_matrix, _matrix_temp, axis=1)

                #_path_matrix_temp = np.full((x, _y), PATH_MATRIX_START, dtype=float)
                #self._path_matrix = np.append( self._path_matrix, _path_matrix_temp, axis=1)

                self._y_max = yw

        self._grid_matrix[xw-1, yw-1] = 0

        self._path.append(self._x_last)
        self._path.append(self._y_last)

    def grid_map_test(self, observation):
        angles = []
        distances = []
        oy = []
        ox = []
        xy_resolution = 0.02

        state = list(map(lambda x: x, observation))
        for i in range(len(state)):

            _deg = float((0.25*i) - 135)
            if _deg < 0:
                _deg = 360 + _deg
            #print(math.radians(_deg))
            angles.append(float(self._robot_orientation_last + math.radians(_deg)))
            distances.append(float(state[i]))

        angles = np.array(angles)
        distances = np.array(distances)        

        #dat = np.array([angles, distances])
        #dat = dat.T
        #np.savetxt('m_angles.txt', dat, delimiter = ';')

        #print(*angles)
        #print(*distances)

        ox = (np.sin(angles) * (distances))
        oy = (np.cos(angles) * (distances))

        occupancy_map, min_x, max_x, min_y, max_y, xy_resolution = self.generate_ray_casting_grid_map(ox, oy, xy_resolution, True)

        #plt.imshow(occupancy_map, cmap="PiYG_r")        

        #xy_res = np.array(occupancy_map).shape
        plt.imshow(occupancy_map, cmap="PiYG_r")               
        
        top, bottom = plt.ylim()  # return the current y-lim
        if(top < bottom):
            plt.ylim((top, bottom))
        else:
            plt.ylim((bottom, top))

        plt.grid(True)
        plt.show()

    def generate_ray_casting_grid_map(self, ox, oy, xy_resolution, breshen=True):
        """
        The breshen boolean tells if it's computed with bresenham ray casting
        (True) or with flood fill (False)
        """
        min_x, min_y, max_x, max_y, x_w, y_w = self.calc_grid_map_config(ox, oy, xy_resolution)

        # default 0.5 -- [[0.5 for i in range(y_w)] for i in range(x_w)]
        occupancy_map = np.ones((x_w, y_w)) / 2
        center_x = int(round(-min_x / xy_resolution))  # center x coordinate of the grid map
        center_y = int(round(-min_y / xy_resolution))  # center y coordinate of the grid map

        #center_x = int(round(self._robot_x_last / xy_resolution))
        #center_y = int(round(self._robot_y_last / xy_resolution))

        #center_x = int(round(self._robot_x_last))
        #center_y = int(round(self._robot_y_last))

        # occupancy grid computed with bresenham ray casting
        if breshen:
            for (x, y) in zip(ox, oy):
                # x coordinate of the the occupied area
                ix = int(round((x - min_x) / xy_resolution))
                # y coordinate of the the occupied area
                iy = int(round((y - min_y) / xy_resolution))
                laser_beams = self.bresenham((center_x, center_y), (ix, iy))  # line form the lidar to the occupied points
                for laser_beam in laser_beams:
                    occupancy_map[laser_beam[0]][laser_beam[1]] = 0.0  # free area 0.0
                occupancy_map[ix][iy] = 1.0  # occupied area 1.0
                occupancy_map[ix + 1][iy] = 1.0  # extend the occupied area
                occupancy_map[ix][iy + 1] = 1.0  # extend the occupied area
                occupancy_map[ix + 1][iy + 1] = 1.0  # extend the occupied area
        # occupancy grid computed with with flood fill
        else:
            occupancy_map = self.init_flood_fill((center_x, center_y), (ox, oy),
                                            (x_w, y_w),
                                            (min_x, min_y), xy_resolution)
            self.flood_fill((center_x, center_y), occupancy_map)
            occupancy_map = np.array(occupancy_map, dtype=np.float)
            for (x, y) in zip(ox, oy):
                ix = int(round((x - min_x) / xy_resolution))
                iy = int(round((y - min_y) / xy_resolution))
                occupancy_map[ix][iy] = 1.0  # occupied area 1.0
                occupancy_map[ix + 1][iy] = 1.0  # extend the occupied area
                occupancy_map[ix][iy + 1] = 1.0  # extend the occupied area
                occupancy_map[ix + 1][iy + 1] = 1.0  # extend the occupied area
        return occupancy_map, min_x, max_x, min_y, max_y, xy_resolution

    def bresenham(self, start, end):
        """
        Implementation of Bresenham's line drawing algorithm
        See en.wikipedia.org/wiki/Bresenham's_line_algorithm
        Bresenham's Line Algorithm
        Produces a np.array from start and end (original from roguebasin.com)
        >>> points1 = bresenham((4, 4), (6, 10))
        >>> print(points1)
        np.array([[4,4], [4,5], [5,6], [5,7], [5,8], [6,9], [6,10]])
        """
        # setup initial conditions
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
        is_steep = abs(dy) > abs(dx)  # determine how steep the line is
        if is_steep:  # rotate line
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        # swap start and end points if necessary and store swap state
        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True
        dx = x2 - x1  # recalculate differentials
        dy = y2 - y1  # recalculate differentials
        error = int(dx / 2.0)  # calculate error
        y_step = 1 if y1 < y2 else -1
        # iterate over bounding box generating points between start and end
        y = y1
        points = []
        for x in range(x1, x2 + 1):
            coord = [y, x] if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += y_step
                error += dx
        if swapped:  # reverse the list if the coordinates were swapped
            points.reverse()
        points = np.array(points)
        return points

    def calc_grid_map_config(self, ox, oy, xy_resolution):
        """
        Calculates the size, and the maximum distances according to the the
        measurement center
        """
        min_x = round(min(ox) - self.EXTEND_AREA / 2.0)
        min_y = round(min(oy) - self.EXTEND_AREA / 2.0)
        max_x = round(max(ox) + self.EXTEND_AREA / 2.0)
        max_y = round(max(oy) + self.EXTEND_AREA / 2.0)

        xw = int(round((max_x - min_x) / xy_resolution))
        yw = int(round((max_y - min_y) / xy_resolution))
        print("The grid map is ", xw, "x", yw, ".")
        return min_x, min_y, max_x, max_y, xw, yw

    def atan_zero_to_twopi(y, x):
        angle = math.atan2(y, x)
        if angle < 0.0:
            angle += math.pi * 2.0
        return angle

    def init_flood_fill(self, center_point, obstacle_points, xy_points, min_coord,
                        xy_resolution):
        """
        center_point: center point
        obstacle_points: detected obstacles points (x,y)
        xy_points: (x,y) point pairs
        """
        center_x, center_y = center_point
        prev_ix, prev_iy = center_x - 1, center_y
        ox, oy = obstacle_points
        xw, yw = xy_points
        min_x, min_y = min_coord
        occupancy_map = (np.ones((xw, yw))) * 0.5
        for (x, y) in zip(ox, oy):
            # x coordinate of the the occupied area
            ix = int(round((x - min_x) / xy_resolution))
            # y coordinate of the the occupied area
            iy = int(round((y - min_y) / xy_resolution))
            free_area = self.bresenham((prev_ix, prev_iy), (ix, iy))
            for fa in free_area:
                occupancy_map[fa[0]][fa[1]] = 0  # free area 0.0
            prev_ix = ix
            prev_iy = iy
        return occupancy_map

    def flood_fill(self, center_point, occupancy_map):
        """
        center_point: starting point (x,y) of fill
        occupancy_map: occupancy map generated from Bresenham ray-tracing
        """
        # Fill empty areas with queue method
        sx, sy = occupancy_map.shape
        fringe = deque()
        fringe.appendleft(center_point)
        while fringe:
            n = fringe.pop()
            nx, ny = n
            # West
            if nx > 0:
                if occupancy_map[nx - 1, ny] == 0.5:
                    occupancy_map[nx - 1, ny] = 0.0
                    fringe.appendleft((nx - 1, ny))
            # East
            if nx < sx - 1:
                if occupancy_map[nx + 1, ny] == 0.5:
                    occupancy_map[nx + 1, ny] = 0.0
                    fringe.appendleft((nx + 1, ny))
            # North
            if ny > 0:
                if occupancy_map[nx, ny - 1] == 0.5:
                    occupancy_map[nx, ny - 1] = 0.0
                    fringe.appendleft((nx, ny - 1))
            # South
            if ny < sy - 1:
                if occupancy_map[nx, ny + 1] == 0.5:
                    occupancy_map[nx, ny + 1] = 0.0
                    fringe.appendleft((nx, ny + 1))

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

