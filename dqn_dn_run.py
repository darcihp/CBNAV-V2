import gym
import numpy as np
import sys

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
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

'''
Mapeamento dos dados discretos da ação
'''
import action_mapper 

DEBUG = True
N_STEPS = 1_000_000

class DQNAgentProc(Processor):
	def __init__(self):
		print ('**********__init__**********')
		self.gewonnen = 0

	def get_gewonnen(self):
		return self.gewonnen
	
	def set_gewonnen(self, gewonnen):
		self.gewonnen = gewonnen;
	

	def process_step(self, observation, reward, done, info):
		if reward == 180 or reward == 20:
			self.gewonnen += 1
			#print ('Wir haben gewonnen: {0} zeit'.format(self.gewonnen ))
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

def build_name(env_name):
    weights_filename = 'results/'+ env_name + str(N_STEPS) +'.h5f'
    return weights_filename

'''
def build_model(states, actions):
	model = Sequential()
	model.add(Dense(1209, input_dim=states, activation='relu'))
	model.add(Dense(512, activation='relu'))
	model.add(Dense(256, activation='relu'))
	model.add(Dense(actions, activation='linear'))
	model.summary()
	return model
'''

def build_model(states, actions):
	model = Sequential()
	model.add(Dense(2048, input_dim=states, activation='relu'))
	model.add(Dense(512, activation='relu'))
	model.add(Dense(256, activation='relu'))
	model.add(Dense(actions, activation='linear'))
	model.summary()
	return model

'''
def build_model(states, actions):
	model = Sequential()
	model.add(Dense(2048, input_dim=states, activation='relu'))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(256, activation='relu'))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(32, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(actions, activation='linear'))
	model.summary()
	return model
'''
def build_agent(model, actions, processor):
	policy = GreedyQPolicy()
	memory = SequentialMemory(limit=30000, window_length=1)
	dqn = DQNAgent(model=model, memory=memory, policy=policy, processor=processor,
		nb_actions=actions, nb_steps_warmup=500, target_model_update=1e-3,
		enable_double_dqn=True,
		enable_dueling_network=True, dueling_type='avg',
		batch_size=8, gamma=.95)
	return dqn

def main():

	print("python dqn_dq_run.py [Landkarte] [Keras model] [Keras model number]")
	print("python dqn_dq_run.py proto_1 t_1 1")

	#landkartes = ["proto_1", "proto_2", "proto_3", "proto_4", "proto_5"]

	#landkartes = ["d0aeed69cef4bb46a2cdbf7a7e13d6cc_out"]
	#landkartes = ["ac5ac75376e2e43d9bb14460e41271d0_out"]
	#landkartes = ["a24e5d6b82e244357f688ca1a57b5806_out"]
	#landkartes = ["7e80c5f4c9905bc273560854d8abe916_out"]
	#landkartes = ["658e5214673c7a4e25b458e56bdb6144_out"]
	#landkartes = ["524f0a38058d39c81d8c3a067cf43904_out"]
	#landkartes = ["2deaa98e9acd1293d5224e25715c1393_out"]
	landkartes = ["01e53c5618fd8d4d78c916b9dcc4ff92_out"]
	#landkartes = ["proto_2"]
	
	#kerasmodels = ["t_1", "t_2", "t_3", "t_4", "t_5", "t_6", "t_7", "t_8", "t_9", "t_10"]
	
	#kerasmodels = ["d0aeed69cef4bb46a2cdbf7a7e13d6cc_out"]
	#kerasmodels = ["ac5ac75376e2e43d9bb14460e41271d0_out"]
	#kerasmodels = ["a24e5d6b82e244357f688ca1a57b5806_out"]
	#kerasmodels = ["7e80c5f4c9905bc273560854d8abe916_out"]
	#kerasmodels = ["658e5214673c7a4e25b458e56bdb6144_out"]
	#kerasmodels = ["524f0a38058d39c81d8c3a067cf43904_out"]
	#kerasmodels = ["2deaa98e9acd1293d5224e25715c1393_out"]
	kerasmodels = ["01e53c5618fd8d4d78c916b9dcc4ff92_out"]
	
 
	for landkarte in landkartes:
		i = 1
		for kerasmodel in kerasmodels:
			#if len(sys.argv) == 4:
			#landkarte = str(sys.argv[1])
			#kerasmodel = str(sys.argv[2])
			#model_number = sys.argv[3]

			model_number = i

			if DEBUG:
				print ('Landkarte: {0}'.format(landkarte))
				print ('Keras Model: {0}'.format(kerasmodel))

			env = Environment("Simulation2d/svg/"+landkarte, int(model_number))
			env.use_observation_rotation_size(True)
			env.set_observation_rotation_size(128)
			env.set_mode(Mode.ALL_RANDOM)

			processor = DQNAgentProc()
			states = env.observation_size()
			actions = action_mapper.ACTION_SIZE

			if DEBUG:
				print('states: {0}'.format(states))
				print('actions: {0}'.format(actions))

			model = build_model(states, actions)
			dqn = build_agent(model, actions, processor)
			#name = build_name('dqn_dn_boltzmann_room')
			name = build_name(kerasmodel)

			dqn.compile(Adam(lr=1e-3), metrics=['mse'])
			dqn.load_weights(name)

			scores = dqn.test(env, nb_episodes=1000, visualize=True, verbose=1)
			#print (np.mean(scores.history['episode_reward']))
			f = open('results_'+landkarte+'.txt', 'a')
			f.write('{0};{1};{2}'.format(landkarte, kerasmodel, processor.get_gewonnen()))
			f.write('\n')
			f.close()
		    #else:
			#print ('Das ist nicht richtig. Bitte informiere Sie die Landkarte und die Keras model. ')
			#exit(0)

			i = i + 1


if __name__ == "__main__":
    main()
