# -*- coding: utf-8 -*-

import cv2
import numpy as np
from PIL import Image
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy, SoftmaxPolicy, EpsGreedyQPolicy, GreedyQPolicy
from rl.memory import SequentialMemory, RingBuffer, EpisodeParameterMemory
from rl.core import Processor
from rl.core import Env
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from environment.environment import Environment
from environment.environment_node_data import Mode

from keras.callbacks import Callback 
from keras.callbacks import CallbackList as KerasCallbackList

import math
import matplotlib.pyplot as plt
from collections import deque

'''
Mapeamento dos dados discretos da ação
'''
import action_mapper 

DEBUG = True

class ManualProc(Processor):
	def __init__(self):
		print ('**********__init__**********')

	def process_step(self, observation, reward, done, info):
		return observation, reward, done, info

	def process_observation(self, observation):
		obs = observation[0]
		return obs

	def process_reward(self, reward):
		return reward

	def process_info(self, info):
		return info

	def process_action(self, action):
		return action

	def process_state_batch(self, batch):
		return batch[:, 0, :]

env = Environment("Simulation2d/svg/ac5ac75376e2e43d9bb14460e41271d0_out", 1)
env.use_observation_rotation_size(True)
env.set_observation_rotation_size(128)
env.set_mode(Mode.ALL_RANDOM)

processor = ManualProc()
states = env.observation_size()
actions = action_mapper.ACTION_SIZE

if DEBUG:
	print('states: {0}'.format(states))
	print('actions: {0}'.format(actions))

state, reward, done, _ = env.reset()
env.render()

done = False

EXTEND_AREA = 1.0

def file_read(f):
	"""
	Reading LIDAR laser beams (angles and corresponding distance data)
	"""
	with open(f) as data:
		measures = [line.split(",") for line in data]
	angles = []
	distances = []
	for measure in measures:
		angles.append(float(measure[0]))
		distances.append(float(measure[1]))
	angles = np.array(angles)
	distances = np.array(distances)
	return angles, distances

def bresenham(start, end):
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

def calc_grid_map_config(ox, oy, xy_resolution):
	"""
	Calculates the size, and the maximum distances according to the the
	measurement center
	"""
	min_x = round(min(ox) - EXTEND_AREA / 2.0)
	min_y = round(min(oy) - EXTEND_AREA / 2.0)
	max_x = round(max(ox) + EXTEND_AREA / 2.0)
	max_y = round(max(oy) + EXTEND_AREA / 2.0)

	xw = int(round((max_x - min_x) / xy_resolution))
	yw = int(round((max_y - min_y) / xy_resolution))
	print("The grid map is ", xw, "x", yw, ".")
	return min_x, min_y, max_x, max_y, xw, yw

def atan_zero_to_twopi(y, x):
	angle = math.atan2(y, x)
	if angle < 0.0:
		angle += math.pi * 2.0
	return angle

def init_flood_fill(center_point, obstacle_points, xy_points, min_coord,
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
		free_area = bresenham((prev_ix, prev_iy), (ix, iy))
		for fa in free_area:
			occupancy_map[fa[0]][fa[1]] = 0  # free area 0.0
		prev_ix = ix
		prev_iy = iy
	return occupancy_map

def flood_fill(center_point, occupancy_map):
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

def generate_ray_casting_grid_map(ox, oy, xy_resolution, breshen=True):

	min_x, min_y, max_x, max_y, x_w, y_w = calc_grid_map_config(ox, oy, xy_resolution)

	occupancy_map = np.ones((x_w, y_w)) / 2
	center_x = int(round(-min_x / xy_resolution))
	center_y = int(round(-min_y / xy_resolution))

	if breshen:
		for (x, y) in zip(ox, oy):

			ix = int(round((x - min_x) / xy_resolution))

			iy = int(round((y - min_y) / xy_resolution))
			laser_beams = bresenham((center_x, center_y), (ix, iy))
			for laser_beam in laser_beams:
				occupancy_map[laser_beam[0]][laser_beam[1]] = 0.0
			occupancy_map[ix][iy] = 1.0

	'''
	else:
		occupancy_map = init_flood_fill((center_x, center_y), (ox, oy),
										(x_w, y_w),
										(min_x, min_y), xy_resolution)
		flood_fill((center_x, center_y), occupancy_map)
		occupancy_map = np.array(occupancy_map, dtype=np.float)
		for (x, y) in zip(ox, oy):
			ix = int(round((x - min_x) / xy_resolution))
			iy = int(round((y - min_y) / xy_resolution))
			occupancy_map[ix][iy] = 1.0  # occupied area 1.0
			occupancy_map[ix + 1][iy] = 1.0  # extend the occupied area
			occupancy_map[ix][iy + 1] = 1.0  # extend the occupied area
			occupancy_map[ix + 1][iy + 1] = 1.0  # extend the occupied area
	'''
	return occupancy_map, min_x, max_x, min_y, max_y, xy_resolution

fig = plt.figure()

while not done:
	plt.ion()
	value = input("Próxima ação: [0 - 6]: \n")
	action = int(value)
	state, reward, done, _ = env.step(action)

	print("Reward: {}", reward)
	
	env.render()

	
	#print(np.max(oy))
	#print(*oy)	

	'''
	xy_resolution = 0.2

	occupancy_map, min_x, max_x, min_y, max_y, xy_resolution = generate_ray_casting_grid_map(ox, oy, xy_resolution, True)

	#xy_res = np.array(occupancy_map).shape
	#plt.figure(1, figsize=(10, 4))
	#plt.subplot(122)
	plt.imshow(occupancy_map, cmap="PiYG_r")
	#plt.clim(-0.4, 1.4)
	#plt.gca().set_xticks(np.arange(-.5, xy_res[1], 1), minor=True)
	#plt.gca().set_yticks(np.arange(-.5, xy_res[0], 1), minor=True)
	#plt.grid(True, which="minor", color="w", linewidth=0.6, alpha=0.5)
	#plt.colorbar()
	#plt.subplot(121)
	#plt.plot([oy, np.zeros(np.size(oy))], [ox, np.zeros(np.size(oy))], "ro-")
	#plt.axis("equal")
	#plt.plot(0.0, 0.0, "ob")
	#plt.gca().set_aspect("equal", "box")
	#bottom, top = plt.ylim()  # return the current y-lim
	#plt.ylim((top, bottom))  # rescale y axis, to match the grid orientation
	plt.pause(0.1)
	#plt.show()
	'''