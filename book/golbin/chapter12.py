import tensorflow as tf
import numpy as np
import random
import time

from game import Game
from model import DQN

tf.app.flags.DEFINE_boolean("train",False,"학습모드. 게임을 화면에 보여주지 않습니다.")
FLAGS = tf.app.flags.FLAGS

MAX_EPISODE = 10000 #최대로 학습할 게임 횟수
TARGET_UPDATE_INTERVAL = 1000 #학습을 일정 횟수만큼 진행할 때마다 한번씩 목표 신경망을 갱신
TRAIN_INTERVAL = 4 #4프레임마다 한 번씩 학습하라
OBSERVE = 100 #일정 수준의 학습 데이터가 쌓이기 전에는 학습하지 않고 지켜보기

NUM_ACTION = 3 #좌, 유지, 우
SCREEN_WIDTH = 6
SCREEN_HEIGHT = 10

def train():
	print('뇌세포 깨우는 중..')
	sess = tf.Session()
	
	game = Game(SCREEN_WIDTH, SCREEN_HEIGHT, show_game=False)
	brain = DQN(sess, SCREEN_WIDTH, SCREEN_HEIGHT, NUM_ACTION)

	rewards = tf.placeholder(tf.float32, [None])
	tf.summary.scalar('avg.reward/ep.', tf.reduce_mean(rewards))
	
	saver = tf.train.Saver()
	sess.run(tf.global_variables_initializer())
	
	writer = tf.summary.FileWriter('logs', sess.graph)
	summary_merged = tf.summary.merge_all()
	
	brain.update_target_network()
	
	epsilon = 1.0
	time_step = 0
	total_reward_list = []
	
	for episode in range(MAX_EPISODE):
		terminal = False
		total_reward = 0
		
		state = game.reset() #게임 상태 초기화
		brain.init_state(state)
		
		while not terminal:
			if np.random.rand() < epsilon:
				action = random.randrange(NUM_ACTION)
			else:
				action = brain.get_action()
			
			if episode > OBSERVE:
				epsilon -= 1/1000
			
			state,reward,terminal = game.step(action)
			total_reward += reward
			
			brain.remember(state,action,reward,terminal)
			
			if time_step > OBSERVE and time_step % TRAIN_INTERVAL == 0:
				brain.train()
			
			if time_step % TARGET_UPDATE_INTERVAL == 0:
				brain.update_target_network()
			
			time_step += 1
		
		print('게임횟수: %d 점수: %d' % (episode+1,total_reward))
		
		total_reward_list.append(total_reward)
		
		if episode % 10 == 0:
			summary = sess.run(summary_merged, feed_dict={reward:total_reward_list})
			writer.add_summary(summary,time_step)
			total_reward_list = []
		
		if episode % 100 == 0:
			saver.save(sess,'model/dqn.ckpt',global_step=time_step)

#학습결과는 실행하는 함수
def replay():
	print('뇌세포 깨우는 중..')
	sess = tf.Session()
	
	game = Game(SCREEN_WIDTH, SCREEN_HEIGHT, show_game=True)
	brain = DQN(sess, SCREEN_WIDTH, SCREEN_HEIGHT, NUM_ACTION)
	
	saver = tf.train.Saver()
	ckpt = tf.train.get_checkpoint_state('model')
	saver.restore(sess,ckpt.model_checkpoint_path)

	#게임진행
	for episode in range(MAX_EPISODE):
		terminal = False
		total_reward = 0
		
		state = game.reset()
		brain.init_state(state)
		
		while not terminal:
			action = brain.get_action()
			
			state,reward,terminal = game.step(action)
			total_reward += reward
			
			brain.remember(state,action,reward,terminal)
			
			time.sleep(0.3)
		
		print('게임횟수: %d 점수: %d' % (episode+1,total_reward))

def main(_):
	if FLAGS.train:
		train()
	else:
		replay()

if __name__ == '__main__':
	tf.app.run()

#

import tensorflow as tf
import numpy as np
import random
from collections import deque

class DQN:
	REPLAY_MEMORY = 10000 #학습에 사용할 플레이 결과를 얼마나 많이 저장해서 사용할지
	BATCH_SIZE = 32 #한번 학습할 때 몇개의 기억을 사용할지. 미니배치 크기
	GAMMA = 0.99 #오래된 상태의 가중치를 줄이기 위한 하이퍼파라미터
	STATE_LEN = 4 #한번에 볼 프레임 총 수

def __init__(self, session, width, height, n_action):
	self.session = session
	self.n_action = n_action
	self.width = width
	self.height = height
	self.memory = deque() #게임 플레이 결과를 저장할 메모리를 만드는 코드
	self.state = None

	self.input_X = tf.placeholder(tf.float32, [None,width,height,self.STATE_LEN]) #게임 상태를 입력받습니다.
	self.input_A = tf.placeholder(tf.int64,[None]) #각 상태를 만들어낸 액션의 값을 입력받습니다.
	self.input_Y = tf.placeholder(tf.float32, [None]) #손실값 걔산에 사용할 값을 입력받습니다.
	
	self.Q = self._build_network('main')
	self.cost, self.train_op = self._build_op()
	
	self.target_Q = self._build_network('target')
	
	def _build_network(self, name):
		with tf.variable_scope(name):
			model = tf.layers.conv2d(self.input_X,32,[4,4],padding='same',activation=tf.nn.relu)
			model = tf.layers.conv2d(model,64,[2,2],padding='same',activation=tf.nn.relu)
			model = tf.contrib.layers.flatten(model)
			model = tf.layers.dense(model,512,activation=tf.nn.relu)
			
			Q = tf.layers.dense(model,self.n_action,activation=None)
		
		return Q
	
	def _build_op(self):
		one_hot = tf.one_hot(self.input_A,self.n_action,1.0,0.0)
		Q_value = tf.reduce_sum(tf.multiply(self.Q,one_hot),axis=1)
		cost = tf.reduce_mean(tf.square(self.input_Y-Q_value))
		train_op = tf.train.AdamOptimizer(1e-6).minimize(cost)
		
		return cost, train_op
	
	def update_target_network(self):
		copy_op = []
		
		main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='main')
		target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='target')
		
		for main_var, target_var in zip(main_vars,target_vars):
			copy_op.append(target_var.assign(main_var.value()))
		
		self.session.run(copy_op)
	
	def get_action(self):
		Q_value = self.session.run(self.Q,feed_dict={self.input_X:[self.state]})
		
		action = np.argmax(Q_value[0])
		
		return action
	
	#학습
	def train(self):
		state,next_state,action,reward,terminal = self._sample_memory()
		
		target_Q_value = self.session.run(self.target_Q,feed_dict={self.input_X:next_state}
		
		Y = []
		for i in range(self.BATCH_SIZE):
			if terminal[i]:
				Y.append(reward[i])
			else:
				Y.append(reward[i] + self.GAMMA * np.max(target_Q_value[i]))
		
		self.session.run(self.train_op,feed_dict={self.input_X:state, self.input_A:action, self.input_Y:Y})
	
	def init_state(self, state):
        state = [state for _ in range(self.STATE_LEN)]
        self.state = np.stack(state, axis=2)
	
	def remember(self, state, action, reward, terminal):
        next_state = np.reshape(state, (self.width, self.height, 1))
        next_state = np.append(self.state[:, :, 1:], next_state, axis=2)

        self.memory.append((self.state, next_state, action, reward, terminal))

        if len(self.memory) > self.REPLAY_MEMORY:
            self.memory.popleft()

        self.state = next_state

    def _sample_memory(self):
        sample_memory = random.sample(self.memory, self.BATCH_SIZE)

        state = [memory[0] for memory in sample_memory]
        next_state = [memory[1] for memory in sample_memory]
        action = [memory[2] for memory in sample_memory]
        reward = [memory[3] for memory in sample_memory]
        terminal = [memory[4] for memory in sample_memory]

        return state, next_state, action, reward, terminal

#C:\>python agent.py --train
#C:\>python agent.py