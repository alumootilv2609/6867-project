# trying basic DQN
# much of this comes from Lawrence Li's DQN as well as the jiromiru(?) thing 
# uses keras

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _pickle as pickle 
import numpy as np
import tensorflow as tf
import gym
import os
import sys
import time
import random
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque
from keras.models import *
from keras.layers import *
from keras.optimizers import *

class Agent():

    def __init__(self):
        self.problem = 'Pong-v0'
        self.env = gym.make(self.problem)
        self.stateCnt = (84,84,4)
        self.actionCnt = self.env.action_space.n
        self.memory_capacity = 10000 # replay memory capacity
        self.batch_size = 32 # batch size
        self.train_time = 100000 # number total frames to train; maybe not necessary
        self.D = deque([], maxlen = self.memory_capacity)
        self.done = True
        self.m_steps = 0
        self.phi = np.zeros([1, 84, 84, 4]) # processed inputs 
        self.gamma = 0.90

        # epsilon things
        self.max_epsilon = 1
        self.min_epsilon = 0.01
        self.eps = self.max_epsilon # at least at start
        self.decay_rate = 0.001

        self.k = 4 # frame skipping
        self.save_dir = str(self.problem) +'-'+ '-ckpts.h5'

    def _create_model(self):
        self.m_steps = 0
        model = Sequential()
        model.add(Conv2D(filters=16, kernel_size=[8,8], strides=4, padding='same', activation='relu', input_shape=(84, 84, 1)))
        model.add(Conv2D(filters=32, kernel_size=[4,4], strides=2, padding = 'same', activation='relu'))
        model.add(Dense(self.actionCnt, activation='linear'))
        opt = RMSprop(lr=0.02000)
        model.compile(loss='mse', optimizer = opt)

        self.model = model
            

    def process_image(self, obs):
        return resize(rgb2gray(obs), (110, 84))[13:97, :]

    def update_phi(self, obs):
        self.phi = np.reshape(self.process_image(obs), (1, 84, 84, 1))
        # self.phi = np.concatenate([self.phi[:, :, :, 1:], obs], axis = 3)


    def _train(self):
        """
        train with experience replay
        """

        batch = random.sample(self.D, min(self.batch_size, len(self.D)))
        no_state = np.zeros(self.stateCnt)

        states = [ o[0] for o in batch]
        states_ = [ (no_state if o[3] is None else o[3]) for o in batch ]

        p = []
        p_ = []
        for ii in range(len(batch)):
            p.append(self._predict(states[ii][:,:,:]))
            p_.append(self._predict(states_[ii][:,:,:]))

        batchLen = len(batch)

        x = np.zeros((batchLen, 84, 84, 1))
        y =np.zeros((batchLen, 11,11,6))

        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]

            t = p[i][0,:,:,:]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + self.gamma* np.amax(p_[i])
            x[i] = s
            y[i] = t

        self.model.fit(x,y,nb_epoch=1,verbose=0)


    def _predict(self, s):
        return self.model.predict_on_batch(s)

    def _act(self): # your current state is self.phi
        if random.random() <= self.eps:
            return np.random.random_integers(0,self.actionCnt-1,size=4)
        else:
            self.pred = self.model.predict(self.phi)
            return keras.backend.argmax(self.pred)
            

    def _add_sample(self, sample): # in the (s, a, r, s_) format
        self.D.append(sample)
        if len(self.D) > self.memory_capacity:
            self.D.pop(0)

    def _sample_memory(self,n):
        n = min(n, len(self.D))
        return random.sample(self.D,n)

    def _observe(self, sample): # in (s, a, r, s_) format
        self._add_sample(sample)
        self.m_steps += 1
        self.eps = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate*self.m_steps)


    def _save_model_h5(self):
        self.model.save(self.save_dir)

    def _load_model_h5(self):
        self.model = load_model(self.save_dir)

    def dump_deque(self):
        max_bytes = 2**31 - 1
        bytes_out = pickle.dumps(self.D)
        n_bytes = sys.getsizeof(bytes_out)
        with open('Pong-v0-deque.p', 'wb') as f_out:
            for i in range(0, n_bytes, max_bytes):
                f_out.write(bytes_out[i:i + max_bytes])

    def load_deque(self):
        max_bytes = 2**31 - 1
        input_size = os.path.getsize('Pong-v0-deque.p')
        bytes_in = bytearray(0)
        with open('Pong-v0-deque.p', 'rb') as f_in: 
            for _ in range(0, input_size, max_bytes): 
                bytes_in += f_in.read(max_bytes)
        self.D = pickle.loads(bytes_in)



    def step_env(self):
        self.env.render()
        print('Step {}'.format(self.m_steps)) 

        if self.done:
            obs = self.env.reset()
            self.update_phi(obs)
            self.done=False
            reward = 0

        if np.random.rand() <= self.eps:
            action = random.randint(0,self.actionCnt-1)
        else:
            maxes = np.asarray([np.amax(self.model.predict(self.phi)[0,0,0,i]) for i in range(0,6)])
            action = np.argmax(maxes)

        for ii in range(self.k):
            obs , r, self.done, info = self.env.step(action)
            if self.done: break
            r = 1 if r > 0 else -1 if r < 0 else 0
            phi_bef = self.phi
            self.update_phi(obs)
            self._observe((phi_bef,action,r,self.phi))

        self._train()

    def run_env(self):
        if os.path.exists(self.save_dir):
            print('Loading from file')
            self._load_model_h5()

            # num = str(self.save_dir).split('-')[2]
            # self.m_steps = int(num)*10000
            print('fasdf')
            print(self.m_steps)
            self.eps = 0.1

            self.load_deque()
        else:
            self.m_steps = 2
            self._create_model()

        for t in range(self.m_steps + 1, self.train_time):
            self.step_env()
            print(self.m_steps)
            if self.m_steps % 10000 == 0:
                print("Saving {}....".format(self.m_steps))
                self._save_model_h5()
                self.dump_deque()


##################### MAIN ######################

def main():
    agent = Agent()
    agent.run_env()

main()
