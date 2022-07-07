'''
python svg_to_world.py ac5ac75376e2e43d9bb14460e41271d0_out.svg 0.015

'''
import gym
import numpy as np
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, PReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import elu

import tensorflow as tf

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

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

'''
Mapeamento dos dados discretos da ação
'''
import action_mapper 

DEBUG = True
N_STEPS = 1_000_000

class DQNAgentProc(Processor):
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

env = Environment("Simulation2d/svg/ac5ac75376e2e43d9bb14460e41271d0_out", 4)
env.use_observation_rotation_size(True)
env.set_observation_rotation_size(128)
env.set_mode(Mode.ALL_COMBINATION)

processor = DQNAgentProc()
states = env.observation_size()
actions = action_mapper.ACTION_SIZE

if DEBUG:
	print('states: {0}'.format(states))
	print('actions: {0}'.format(actions))

def build_callbacks(env_name):
    weights_filename = 'results/'+ env_name +'{step}.h5f'
    log_filename = 'log/{}.json'.format(env_name)
    callbacks = [ModelIntervalCheckpoint(weights_filename, interval=100_000)]
    callbacks += [FileLogger(log_filename, interval=1_000)]
    return callbacks


def build_model(states, actions):
	init = tf.keras.initializers.HeUniform()
	model = Sequential()
	model.add(Dense(2048, input_dim=states, activation='relu', kernel_initializer=init))
	#model.add(Dense(512, activation='relu', kernel_initializer=init))
	#model.add(Dense(256, activation='relu', kernel_initializer=init))
	model.add(Dense(actions, activation='linear', kernel_initializer=init))
	model.summary()
	return model
'''
def build_model(states, actions):
	model = Sequential()
	model.add(Dense(2048, input_dim=states, activation='relu'))
	model.add(Dense(512, activation='elu'))
	model.add(Dropout(0.25))
	model.add(Dense(256, activation='relu'))
	model.add(Dense(128, activation='elu'))
	model.add(Dropout(0.25))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(32, activation='elu'))
	model.add(Dropout(0.25))
	model.add(Dense(actions, activation='linear'))
	model.summary()
	return model
'''

def build_agent(model, actions):
	policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), 
		attr='eps', value_max=1., 
		value_min=.05, value_test=.05, 
		nb_steps= 850_000)
	#policy = EpsGreedyQPolicy(eps=.1)
	#policy = GreedyQPolicy()

	#policy = BoltzmannQPolicy()
	memory = SequentialMemory(limit=30000, window_length=1)
	dqn = DQNAgent(model=model, memory=memory, policy=policy, processor=processor,
		nb_actions=actions, nb_steps_warmup=1_000, target_model_update=10_000,
		enable_double_dqn=True,
		enable_dueling_network=True, dueling_type='avg',
		batch_size=64, gamma=.99)
	return dqn

model = build_model(states, actions)
dqn = build_agent(model, actions)
callbacks = build_callbacks('ac5ac75376e2e43d9bb14460e41271d0_out_4')

#dqn.compile(Adam(lr=1e-3), metrics=['mse'])

#dqn.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss=tf.keras.losses.Huber(), metrics=['accuracy'])
dqn.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), metrics=['mse'])

dqn.fit(env, nb_steps=N_STEPS, visualize=False, verbose=1, log_interval=1_000, callbacks=callbacks)
#scores = dqn.test(env, nb_episodes=10, visualize=True)
#print (np.mean(scores.history['episode_reward']))

