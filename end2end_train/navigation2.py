import numpy as np
from itertools import count
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.nn import init
from torch.nn import DataParallel

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.animation
from IPython.display import HTML

import POMDPgame_bars
from POMDPgame_bars import*

import POMDPgame_basic
from POMDPgame_basic import*

import POMDPgame_holes
from POMDPgame_holes import*

import POMDPgame_scale
from POMDPgame_scale import*

import POMDPgame_scale_y
from POMDPgame_scale_y import*

import POMDPgame_scale_x
from POMDPgame_scale_x import*

import POMDPgame_implicit
from POMDPgame_implicit import*

import RNN 
from RNN import *

import torch
import torch.utils.data



def CreateGame(Game, holes = 0, implicit = False, noise = 0, task = 'basic', gpu = 0):

    class ValueMaxGame(Game):

        def __init__(self, e = noise, holes = 0, grid_size = 8, random_seed = 0, set_reward = 0, time_limit = 200, input_type = 0, lam = 0.5, discount = 0.99, alpha = 0.5, implicit = implicit, task = 'basic'):
            Game.__init__(self, discount=discount, grid_size=grid_size, time_limit=time_limit,
                             random_seed=random_seed, set_reward=set_reward, input_type=input_type)
            # need to have randomness
            self.e = e
            self.gpu = gpu
            self.net = RNN(9, 512, 4, k_action = 1).cuda(self.gpu)
            self.hidden = self.net.initHidden()
            self.action = self.net.initAction()
            self.Loss = 0
            self.lr = 0
            self.Life = []
            self.Succeed = []
            self.trace = []
            self.hiddens = []
            self.Hiddens = []
            self.Qs = []
            self.Pos = []
            self.time_limit = time_limit
            # running avearage rate
            self.alpha = alpha
            # backward ratio
            self.lam = lam
            self.succeed = 0
            self.life = 0
            self.y_mid = 0
            self.x_mid = 0
            self.holes = holes
            # control the map seed, if train == true then render seed between 0 , 1
            self.implicit = implicit
            self.task = task 

        def sample(self):
            # choose between 0, 1,2,3
            np.random.seed()
            return np.random.randint(0,4)

        def placefield(self, pos):
            pos_ = (pos[0] - self.y_mid, pos[1] - self.x_mid)
    #         print (pos_, self.set_reward)
    #         print (self.grid_size)
            field =np.zeros((2, 19))
    #         print (len(field))
            for k in range(2):
                for i in range(field.shape[1]):
                # distance generation
                    pos_relative = pos[k] * 19./(self.grid_size[1] + 4)
                    field[k, i] =  (i- pos_relative) ** 2
    #                 print (i - pos[k])e
            # gaussian density, but before exponential to help learning identity mapping input to output
            field = - 0.1 * torch.from_numpy(field).cuda(self.gpu).resize(1, 2 * 19).float()
            return field

        def maxplay(self, state):
            # Move according to action: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
            # value for four action
            action0 = self.action.clone()
            if self.implicit == True:
                self.values, self.hidden = self.net(state.cuda(self.gpu), self.hidden.cuda(self.gpu), self.action.cuda(self.gpu), self.placefield(self.pos_reward_).cuda(self.gpu))
            else:
                # print(self.action, self.hidden, self.placefield(self.pos_reward), state)
                self.values, self.hidden = self.net(state.cuda(self.gpu), self.hidden.cuda(self.gpu), self.action.cuda(self.gpu), self.placefield(self.pos_reward).cuda(self.gpu))
            action = self.values.data.cpu().numpy().argmax()
            # action to state
            if np.random.random()< self.e:
                action = self.sample()
            self.action = torch.eye(4)[action].resize(1, 4).cuda(self.gpu)
            return action, action0
    #  decode is binary
        def decode(self):
            pos = self.hidden.matmul(self.net.h2p_rls) + self.net.bp_rls
            return pos
        # test is for testing phase, decode is to train decoder
        def step(self, policy, epsilon = 'default', record = False, test = False, cross = False, train_hidden = False, decode = False):
            if epsilon != 'default':
                self.e = epsilon
            self.t += 1
            """enviroment dynamics Update state (grid and agent) based on an action"""
            # state to action
            state_t0 = self.visible_state
            pos0 = self.agent.pos
            # network dynamics and decision
            action, action0 = policy(torch.FloatTensor(state_t0).cuda(self.gpu).resize(1, 9))
            # action to state
            self.agent.act(action)
            state_t1 = self.visible_state
            pos1 = self.agent.pos

            def wall(pos1):
                y, x = pos0
                # wall detection
                if self.grid.grid[pos1] < 0:
                    self.agent.pos = pos0
                    pos_possible = [pos for pos in [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)] if
                                    self.grid.grid[pos] >= 0]
                    self.agent.pos = pos_possible[np.random.randint(len(pos_possible))]
                    pos1 = self.agent.pos

                    self.t += 1
                    return True
                else:
                    return False
            def rewarding(pos1):
                # punish collide
                collision = wall(pos1)
                # punishment by cold water or click
                if collision == True:
                    reward = -0.5
                else:
                    reward = -0.01
                if self.grid.grid[pos1] > 0:
                    reward = self.grid.grid[pos1]
                # death
                elif self.t >= self.time_limit:
                    reward = -1
                # Check if agent won (reached the goal) or lost (health reached 0)
                # attention! 需要括号， 否则reward会被更新
                done = (reward > 0 or self.t >= self.time_limit)
                return reward, done
            def rewarding_hole(pos1):
                # punish collide
                collision = wall(pos1)
                # punishment by cold water or click
#                 if collision == True:
#                     reward = -0.5
                # else:
                reward = -0.01
                if self.grid.grid[pos1] > 0:
                    reward = self.grid.grid[pos1]
                # death
                elif self.t >= self.time_limit:
                    reward = -1
                # Check if agent won (reached the goal) or lost (health reached 0)
                # attention! 需要括号， 否则reward会被更新
                done = (reward > 0 or self.t >= self.time_limit)
                return reward, done
            def rewarding_bar(pos1):
                # punish collide
                collision = wall(pos1)
                # punishment by cold water or click
#                 if collision == True:
#                     reward = -0.5
                # else:
                reward = -0.01
                reward = reward - 0.01 * np.int(action != torch.max(action0).cpu().data.numpy())
                if self.grid.grid[pos1] > 0:
                    reward = self.grid.grid[pos1]
                # death
                elif self.t >= self.time_limit:
                    reward = -1
                # Check if agent won (reached the goal) or lost (health reached 0)
                # attention! 需要括号， 否则reward会被更新
                done = (reward > 0 or self.t >= self.time_limit)
                return reward, done
                
            def rewarding_scale(pos1):
                # punish collide
                collision = wall(pos1)
                # punishment by cold water or click
#                 if collision == True:
#                     reward = -0.5
                # else:
                reward = -0.01
                if self.grid.grid[pos1] > 0:
                    reward = self.grid.grid[pos1]
                # death
                elif self.t >= self.time_limit:
                    reward = -1
                done = (reward > 0 or self.t >= self.time_limit)
                return reward, done
            # rewarding accdonig to tasks 
            if self.task == 'hole':
                reward, done = rewarding_hole(pos1)
            elif self.task == 'bar':
                reward, done = rewarding_bar(pos1)
            elif self.task == 'scale':
                reward, done = rewarding_scale(pos1)
            else:
                reward, done = rewarding(pos1)
            # update value function
            def TD(decode = False):
                if self.implicit == True:
                    realQ, _ = self.net(Variable(torch.FloatTensor(state_t1)).resize(1, 9).cuda(self.gpu), self.hidden, self.action,
                                        self.placefield(self.pos_reward_))
                else:
                    realQ,_  = self.net(Variable(torch.FloatTensor(state_t1)).resize(1,9).cuda(self.gpu), self.hidden, self.action, self.placefield(self.pos_reward))
                # target Q is for state before updated, it only needs to update the value assocate with action taken
                targetQ = self.values.clone().detach()
                # new Q attached with the new state
                # max of q for calculating td error
                Qmax = torch.max(realQ)
                if done != True:
                    delta  =  torch.FloatTensor([reward]).cuda(self.gpu) + self.discount*Qmax  - targetQ[0, action]
            # the virtual max action reaches terminal , there is no q max term assciate with next max value
                elif done == True:
                    delta  =  torch.FloatTensor([reward]).cuda(self.gpu)- targetQ[0, action]
                # eligilibty trace for updating all last states before because of the information about new state
                self.trace = [e * self.discount * self.lam for e in self.trace]
                # eligibility trace attach new state
                self.Qs.append(targetQ)
                self.trace.append(1)
                # corresponding features h
                # update all last action values with eligibility trace, the q will add a new updated value
                def f(e, delta, q):
        #             print (e, delta, q)
                    q[0, action] = q[0, action] + self.alpha * delta * e
                    return q
        #         print (self.trace)
                self.Qs = [f(e, delta, q) for e, q in zip(self.trace, self.Qs)]
        #         print (self.Qs)
                # record values
            if test == False:
                TD()
            # record position and hidden state
            self.Hiddens.append(self.hidden.clone())
            if train_hidden == True:
                self.Pos.append(self.placefield())
            elif decode == True:
                self.Pos.append(torch.FloatTensor([pos0[0], pos0[1]]).resize(1, 2)).cuda(self.gpu)
    #         print (pos1)
            return pos0, state_t0, reward, done

        def step_empty(self, stim=torch.zeros(1, 9), action_=0, epsilon='default', open_loop=True,
                       context=torch.zeros(1, 38)):
            self.t += 1
            # set e in any case
            if epsilon != 'default':
                self.e = epsilon
            state = stim
            pos_reward = self.Set_reward[0]
            self.values, self.hidden = self.net(torch.FloatTensor(state).resize(1, 9).cuda(self.gpu), self.hidden.cuda(self.gpu), self.action.cuda(self.gpu),
                                                context.cuda(self.gpu))
            action = self.values.data.cpu().numpy().argmax()
            if random.random() < self.e:
                action = self.sample()
            if open_loop == True:
                if action_ < 4:
                    self.action = torch.eye(4)[action_].resize(1, 4).cuda(self.gpu)
                else:
                    self.action = 0.25 * torch.ones(4).resize(1, 4).cuda(self.gpu)
            else:
                self.action = torch.eye(4)[action].resize(1, 4).cuda(self.gpu)
            return action

        def velocity(self, stim=torch.zeros(1, 9), hidden0=torch.randn(1, 512, requires_grad=True), action=4):
            # Move according to action: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
            # value for four action
            if action <= 3:
                self.action = torch.eye(4)[action].resize(1, 4)
            else:
                self.action = 0.25 * torch.ones(4).resize(1, 4)
            velocity = self.net.velocity(stim, hidden0, self.action, self.placefield(self.pos_reward))
            return velocity


        def episode(self, epochs=10, epsilon='default', reward_control=None, size_range=(10, 20), prob=5 * [0.2],
                        train_hidden = True, test=False, decode=False, batch_size = 8):
                if reward_control == None:
                    self.reset(reward_control=np.random.randint(len(self.set_reward)), size_range=size_range, prob=prob)
                else:
                    self.reset(reward_control=reward_control, size_range=size_range, prob=prob)
                done = False
                # train only hidden to output
                if epsilon != 'default':
                    self.e = epsilon
                k = 0
                self.Hiddens_batch = []
                self.Targets_batch = []
                self.Pos_batch = []

                def Done(decode=decode):
                    # data record
                    self.Hiddens_batch.extend(self.Hiddens)
                    self.Targets_batch.extend(self.Qs)
                    self.Pos_batch.extend(self.Pos)
                    self.trace = []
                    self.Qs = []
                    self.Hiddens = []
                    self.Pos = []
                    # reset
                    done = False
                    if reward_control == None:
                        self.reset(reward_control=np.random.randint(len(self.set_reward)), size=self.size)
                    else:
                        self.reset(reward_control=reward_control, size=self.size)

                        #         process = psutil.Process(os.getpid())

                #         print('start episode', process.memory_info().rss)
                for iters in range(batch_size):
                    for i in range(epochs):
                        k += 1
                        for t in range(self.size * 10):
                            _, state_t0, reward, done = self.step(self.maxplay, epsilon=epsilon, train_hidden=train_hidden,
                                                                  test=test, decode=decode)
                            if done == True:
                                Done()
                        Done()


        # if use random tensor , there is no memory leak,  if use hiddens and targets for beta but not really update weight, still leak,  so the error is in between
        def train_sgd(self, lr_rate=1e-5, batch_size=8):
                hiddens = torch.stack(self.Hiddens_batch).view(int(len(self.Hiddens_batch)/batch_size), batch_size, -1).cuda(self.gpu)
                targets = torch.stack(self.Targets_batch).squeeze().view(int(len(self.Hiddens_batch)/batch_size), batch_size, -1).cuda(self.gpu)
                # pair h and q
                data = [(h, p) for h, p in zip(hiddens, targets)]
                trainloader = torch.utils.data.DataLoader(data, batch_size=batch_size)
                optimizer = torch.optim.Adam(
                    [
                        {'params': self.net.i2h, 'lr': lr_rate, 'weight_decay': 0},
                        {'params': self.net.a2h, 'lr': lr_rate, 'weight_decay': 0},
                        {'params': self.net.h2h, 'lr': lr_rate, 'weight_decay': 0},
                        {'params': self.net.bh, 'lr': lr_rate, 'weight_decay': 0},
                        {'params': self.net.h2o, 'lr': 1e1 * lr_rate, 'weight_decay': 0},
                        {'params': self.net.bo, 'lr': 1e1 * lr_rate, 'weight_decay': 0},
                    ]
                )
                # for a number of iterations, do gradient descent, 10 here is set as a number which loss starts to stablize with 
                for k in range(10):
                    for i, data in enumerate(trainloader, 0):
                        hidden, labels = data
                        output = hidden.matmul(self.net.h2o) + self.net.bo
                        #                 print (predicts.size(), labels.size())
                        # cross entropy of mini batch
                        loss = torch.sum((output - labels) ** 2)
                        # print (output, labels)
                        self.Loss += loss
                    self.Loss.backward(retain_graph=True)
                    optimizer.step()
                    self.net.zero_grad()
                    print('loss', self.Loss)
                    self.Loss = 0

        def experiment(self, rls_sl, epochs=20, epsilon=0, reward_control=None, train_hidden=False, train_q=True,
                       decode=False, size_range=[10], test=False, lr_rate = 1e-5, batch_size = 8):
            # initialize, might take data during test
            self.trace = []
            self.Qs = []
            self.Hiddens = []
            self.Pos = []
            self.episode(epochs=epochs, epsilon=epsilon, reward_control=reward_control, size_range=size_range,
                         prob=np.ones(len(size_range)) / len(size_range), train_hidden=train_hidden, test=test,
                         decode=decode, batch_size = batch_size)
            if train_hidden == False and train_q == False:
                self.train_sl(rls_sl)
            elif train_hidden == False and train_q == True:
                self.train_sgd(lr_rate = lr_rate, batch_size = batch_size)
            self.Hiddens_batch = []
            self.Targets_batch = []
            self.Pos_batch = []
            process = psutil.Process(os.getpid())
            print('clear session data', process.memory_info().rss)

    game = ValueMaxGame(grid_size=(15, 15), holes=holes, random_seed=4, set_reward=[(0.5, 0.25), (0.5, 0.75)], input_type =0, task = task)
    return game



