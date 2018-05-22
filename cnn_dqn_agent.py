# -*- coding: utf-8 -*-

import six.moves.cPickle as pickle
import copy
import os
import numpy as np
from chainer import cuda

from cnn_feature_extractor import CnnFeatureExtractor
from q_net import QNet
from collections import deque


class CnnDqnAgent(object):
    policy_frozen = False
    epsilon_delta = 1.0 / 10 ** 4.4# print '%.10f' %(1.0 / 10 ** 4.4)
                                   # 0.0000398107170553496878006617676337697275812388397753238677978515625
    min_eps = 0.1

    actions = [0, 1, 2]

    cnn_feature_extractor = 'alexnet_feature_extractor.pickle'
    model = 'bvlc_alexnet.caffemodel'
    model_type = 'alexnet'
    image_feature_dim = 256 * 6 * 6
    image_feature_count = 1
    actions_evaluate = deque(maxlen=4) #----------------------------------------------------------

    def _observation_to_featurevec(self, observation):
        # TODO clean
        if self.image_feature_count == 1:
            #print observation["image"][0].shape, type(observation["image"][0])#会error因为不是np所以没有shape
            #print self.feature_extractor.feature(observation["image"][0]).shape#,\#返回的是1D的256*6*6
            #observation["depth"][0].shape
            return np.r_[self.feature_extractor.feature(observation["image"][0])]
                         #, observation["depth"][0]]
        elif self.image_feature_count == 4:
            return np.r_[self.feature_extractor.feature(observation["image"][0]),
                         self.feature_extractor.feature(observation["image"][1]),
                         self.feature_extractor.feature(observation["image"][2]),
                         self.feature_extractor.feature(observation["image"][3])]#
                         # observation["depth"][0],
                         # observation["depth"][1],
                         # observation["depth"][2],
                         # observation["depth"][3]]
        else:
            print("not supported: number of camera")

    def agent_init(self, **options):
        self.use_gpu = options['use_gpu']
        #self.depth_image_dim = options['depth_image_dim']
        self.q_net_input_dim = self.image_feature_dim * self.image_feature_count #+ self.depth_image_dim

        if os.path.exists(self.cnn_feature_extractor):
            print("loading... " + self.cnn_feature_extractor),
            self.feature_extractor = pickle.load(open(self.cnn_feature_extractor))
            print("done")
        else:
            self.feature_extractor = CnnFeatureExtractor(self.use_gpu, self.model, self.model_type, self.image_feature_dim)
            pickle.dump(self.feature_extractor, open(self.cnn_feature_extractor, 'w'))
            print("pickle.dump finished")

        self.time = 0
        self.epsilon = 1.0  # Initial exploratoin rate
        self.q_net = QNet(self.use_gpu, self.actions, self.q_net_input_dim)

    def agent_start(self, observation):
        obs_array = self._observation_to_featurevec(observation)#拿到前面去提取分析再r合并了

        # Initialize State
        self.state = np.zeros((self.q_net.hist_size, self.q_net_input_dim), dtype=np.uint8)
        self.state[0] = obs_array
        state_ = np.asanyarray(self.state.reshape(1, self.q_net.hist_size, self.q_net_input_dim), dtype=np.float32)
        if self.use_gpu >= 0:
            state_ = cuda.to_gpu(state_)

        # Generate an Action e-greedy
        action, q_now = self.q_net.e_greedy(state_, self.epsilon)#return return_action1
        return_action = action                                   #return return_action2
        print return_action, type(return_action)#------------------------------------------￥￥￥￥ Random 2 <type 'int'>
        # Update for next step
        self.last_action = copy.deepcopy(return_action)
        self.last_state = self.state.copy()#作为下个状态的开始
        self.last_observation = obs_array #：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：

        return return_action                                     #return return_action3
                                               # action, q_now = self.q_net.e_greedy(state_, self.epsilon)-75

    def agent_step(self, reward, observation):
        obs_array = self._observation_to_featurevec(observation)#拿到前面去提取分析再r＿合并了

        #obs_processed = np.maximum(obs_array, self.last_observation)  # Take maximum from two frames

        # Compose State : 4-step sequential observation
        if self.q_net.hist_size == 4:
            self.state = np.asanyarray([self.state[1], self.state[2], self.state[3], obs_array], dtype=np.uint8)
        elif self.q_net.hist_size == 2:
            self.state = np.asanyarray([self.state[1], obs_array], dtype=np.uint8)
        elif self.q_net.hist_size == 1:
            self.state = np.asanyarray([obs_array], dtype=np.uint8)
        else:
            print("self.DQN.hist_size err")

        state_ = np.asanyarray(self.state.reshape(1, self.q_net.hist_size, self.q_net_input_dim), dtype=np.float32)
        #state_从self.state = np.asanyarray([obs_array], dtype=np.uint8)的uint8去小数变成shape（1，1，256*6*6）的float32
        if self.use_gpu >= 0:
            state_ = cuda.to_gpu(state_)

        # Exploration decays along the time sequence
        if self.policy_frozen is False:  # Learning ON/OFF
            if self.q_net.initial_exploration < self.time:#10000<运行时间
                self.epsilon -= self.epsilon_delta        #那么开始渐渐减少eps
                if self.epsilon < self.min_eps:   #如果eps已经被减少的快没了比预定的最小值还要小，
                    self.epsilon = self.min_eps   #则等于min_eps =0.1
                eps = self.epsilon # self.epsilon = 1.0 ----61 理由是 if np.random.rand() < epsilon:q_net.py160行
            else:  # Initial Exploation Phase
                print("Initial Exploration : %d/%d steps" % (self.time, self.q_net.initial_exploration)),
                eps = 1.0 #---------------------1￥打印现在的step，需要学习的步子（例如 Initial Exploration : 173/1000 steps）
        else:  # Evaluation
            print("Policy is Frozen")
            eps = 0.05

        # Generate an Action by e-greedy action selection
        action, q_now = self.q_net.e_greedy(state_, eps)#-----------------------------------------3维度数组state_和1.0
        return action, eps, q_now, obs_array
        # server.py 120行 self.agent.agent_step_update(reward, action, eps, q_now, obs_array)

    def agent_step_update(self, reward, action, eps, q_now, obs_array):
        # Learning Phase
        if self.policy_frozen is False:  # Learning ON/OFF
            self.q_net.stock_experience(self.time, self.last_state, self.last_action, reward, self.state, False)
            print "-------------------  Real Index%d" % (self.q_net.data_index)#%int
            self.actions_evaluate.append(self.last_action)
            if self.actions_evaluate[-1] == self.actions.index(2) and reward >= 1.0 and len(self.actions_evaluate) == 4:
                if [self.actions_evaluate[i]for i in xrange(3)] == ([1, 0, 1]or[1, 0, 1]):
                    index = np.asanyarray(self.q_net.data_index, dtype=np.int8)
                    for i in xrange(1, len(self.actions_evaluate)+1):
                        self.q_net.d[2][index - i] -= 0.5
      #-----#   self.action_evaluate = deque()----------------------------------------------!!!!!!!!!!!!!!!!!!!!!!!!!!
            self.q_net.experience_replay(self.time)

        # Target model update
        if self.q_net.initial_exploration < self.time and np.mod(self.time, self.q_net.target_model_update_freq) == 0:
            print("Model Updated")
            self.q_net.target_model_update()

        # Simple text based visualization
        if self.use_gpu >= 0:
            q_max = np.max(q_now.get())
        else:
            q_max = np.max(q_now)

        print('Step:%d  Action:%d  Reward:%.1f  Epsilon:%.6f  Q_max:%3f' % (
            self.time, self.q_net.action_to_index(action), reward, eps, q_max))
        # ￥Step:92  Action:0  Reward:0.0  Epsilon:1.000000  Q_max:0.000000
        # Updates for next step
        self.last_observation = obs_array#：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：

        if self.policy_frozen is False:
            self.last_action = copy.deepcopy(action)
            self.last_state = self.state.copy()
            self.time += 1

    def agent_end(self, reward):  # Episode Terminated！！
        print('episode finished. Reward:%.1f / Epsilon:%.6f' % (reward, self.epsilon))

        # Learning Phase
        if self.policy_frozen is False:  # Learning ON/OFF
            self.q_net.stock_experience(self.time, self.last_state, self.last_action, reward, self.last_state,
                                        True)#----------------------------------------------------------------

            self.q_net.experience_replay(self.time)

        # Target model update
        if self.q_net.initial_exploration < self.time and np.mod(self.time, self.q_net.target_model_update_freq) == 0:
            print("Model Updated")
            self.q_net.target_model_update()

        # Time count
        if self.policy_frozen is False:
            self.time += 1
