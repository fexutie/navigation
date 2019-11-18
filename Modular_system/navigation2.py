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

import modules_net
from modules_net import * 


def CreateGame(Game, holes = 0, implicit = False, noise = 0, task = 'basic', weight1 = None, weight2 = None):

    class ValueMaxGame(Game):

        def __init__(self, e = noise, holes = 0, grid_size = 8, random_seed = 0, set_reward = 0, time_limit = 200, input_type = 0, lam = 0.5, discount = 0.99, alpha = 0.5, implicit = implicit, task = 'basic', weight1 = weight1, weight2 = weight2, gpu = 1):
            print ('task', Game)
            Game.__init__(self, discount=discount, grid_size=grid_size, time_limit=time_limit,
                             random_seed=random_seed, set_reward=set_reward, input_type=input_type)
            # need to have randomness
            self.e = e
            self.gpu = gpu
            self.net = Modules(9, 512, 4)
            if ((weight1 != None) and (weight2 != None)):
                self.net.loadweight(weight1, weight2)
            self.hidden1 = self.net.initHidden()
            self.hidden2 = self.net.initHidden()
            self.action = self.net.initAction()
            self.Loss = 0
            self.lr = 0
            self.Life = []
            self.Succeed = []
            self.trace1 = []
            self.trace2 = []
            self.hiddens = []
            self.Hiddens = []
            self.Selectors = []
            self.Qs1 = []
            self.Qs2 = []
            self.TargetQ_meta = []
            self.Pos = []
            self.time_limit = time_limit
            # running avearage rate
            self.alpha = alpha
            self.alpha2 = 0.5
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
            field = - 0.1 * torch.from_numpy(field).resize(1, 2 * 19).float()
            return field

        def maxplay(self, state):
            # Move according to action: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
            # value for four action
            action0 = self.action.clone()
            if self.implicit == True:
                self.values1, self.values2, self.hidden1, self.hidden2, self.Q_selector = self.net(state, self.hidden1, self.hidden2, self.action, self.placefield(self.pos_reward_))
            else:
                self.values1, self.values2, self.hidden1, self.hidden2, self.Q_selector = self.net(state, self.hidden1, self.hidden2, self.action, self.placefield(self.pos_reward))
            meta_action = self.Q_selector.data.numpy().argmax()
            if meta_action == 0:
                action = (self.values1).data.cpu().numpy().argmax()
            else:
                action = (self.values2).data.cpu().numpy().argmax()
            # action to state
            if np.random.random()< self.e:
                action = self.sample()
            self.action = torch.eye(4)[action].resize(1, 4)
            return action, action0, meta_action

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
            action, action0, meta_action = policy(Variable(torch.FloatTensor(state_t0)).resize(1, 9))
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
                reward = reward - 0.01 * np.int(action != torch.max(action0).data.numpy())  
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
            if self.implicit == True:
                realQ1, realQ2, _, _ , realQ_sel= self.net(Variable(torch.FloatTensor(state_t1)).resize(1, 9), self.hidden1,
                                                self.hidden2, self.action,
                                                self.placefield(self.pos_reward_))
            else:
                realQ1, realQ2, _, _,  realQ_sel = self.net(Variable(torch.FloatTensor(state_t1)).resize(1, 9), self.hidden1,
                                                self.hidden2, self.action, self.placefield(self.pos_reward))

            # td compute for each module
            def TD(realQ, Module = 0, decode = False):

                # target Q is for state before updated, it only needs to update the value assocate with action taken
                if Module == 0:
                    targetQ = self.values1.clone().detach()
                else:
                    targetQ = self.values2.clone().detach()
                # new Q attached with the new state
                # max of q for calculating td error
                Qmax = torch.max(realQ)
                if done != True:
                    delta  =  torch.FloatTensor([reward]) + self.discount*Qmax  - targetQ[0, action]
                # td error, the virtual max action reaches terminal , there is no q max term assciate with next max value
                elif done == True:
                    delta  =  torch.FloatTensor([reward]) - targetQ[0, action]
                # eligilibty trace for updating all last states before because of the information about new state
                def f(e, delta, q):
                    #             print (e, delta, q)
                    q[0, action] = q[0, action] + self.alpha2 * delta * e
                    return q
                # eligibility trace attach new state
                if Module == 0:
                    self.trace1 = [e * self.discount * self.lam for e in self.trace1]
                    self.Qs1.append(targetQ)
                    self.trace1.append(1)
                    self.Qs1 = [f(e, delta, q) for e, q in zip(self.trace1, self.Qs1)]
                else:
                    self.trace2 = [e * self.discount * self.lam for e in self.trace2]
                    self.Qs2.append(targetQ)
                    self.trace2.append(1)
                    self.Qs2 = [f(e, delta, q) for e, q in zip(self.trace2, self.Qs2)]
            if test == False:
                for i, realQ in enumerate([realQ1, realQ2]):
                    TD(realQ, Module = i)
            # else:
            #     print (meta_action)
            # update meta Q
            if test == False:
                if meta_action == 0:
                    reward_meta = torch.max(self.Qs1[-1])
                else:
                    reward_meta = torch.max(self.Qs2[-1])
                # meta Q update
                targetQ_meta = self.Q_selector.clone().detach()
                Qmax_meta = torch.max(realQ_sel)
                delta_meta = torch.FloatTensor([reward_meta]) + self.discount * Qmax_meta - targetQ_meta[0, meta_action]

                targetQ_meta[0, meta_action] = targetQ_meta[0, meta_action] + self.alpha * delta_meta
                self.TargetQ_meta.append(targetQ_meta)
            # record position and hidden state
            hidden = torch.cat([self.hidden1, self.hidden2], dim = 1)
            self.Hiddens.append(hidden.clone())

            if train_hidden == True:
                self.Pos.append(self.placefield())
            elif decode == True:
                self.Pos.append(torch.FloatTensor([pos0[0], pos0[1]]).resize(1, 2))
    #         print (pos1)
            return pos0, state_t0, reward, done

            # suppose walk in an empty room
        def step_empty(self, stim = torch.zeros(1, 9), action_ = 0, epsilon = 'default', T0 = 100, T = 400, time_stop = 60, open_loop = True):
            self.t += 1
            # set e in any case
            if epsilon !=  'default':
                self.e = epsilon
            if self.t < T0+time_stop and self.t>T0:
                state = stim
            else:
                state = torch.zeros(1, 9)
            pos_reward = self.set_reward[0]
            self.values, self.hidden = self.net(torch.FloatTensor(state).resize(1, 9), self.hidden, self.action, self.placefield(pos_reward))
            action = self.values.data.cpu().numpy().argmax()
            if random.random()< self.e:
                action = self.sample()
            if open_loop == True:
                self.action = torch.eye(4)[action_].resize(1, 4)
            else:
                self.action = torch.eye(4)[action].resize(1, 4)
            if self.t > T:
                done = True
            else:
                done = False
            return done, action


        def episode(self, epochs = 10, epsilon = 'default', reward_control = None, size_range = (10, 20), prob = 5 * [0.2], train_hidden = False, test = False, decode = False):
            if reward_control == None:
                self.reset(reward_control = np.random.randint(len(self.set_reward)), size_range = size_range, prob = prob)
            else:
                self.reset(reward_control = reward_control, size_range = size_range, prob = prob)
            done = False
            # train only hidden to output
            if epsilon != 'default':
                self.e = epsilon
            k = 0
            self.Hiddens_batch = []
            self.Selectors_batch = []
            self.Targets_batch = []
            self.TargetQ_meta_batch = []
            self.Pos_batch = []
            def Done(decode = decode):
                # data record
                if len(self.Qs1) >= 1:
                    self.Hiddens_batch.extend(self.Hiddens)
                    self.Targets_batch.append([torch.stack(self.Qs1), torch.stack(self.Qs2)])
                    self.TargetQ_meta_batch.extend(self.TargetQ_meta)
                if decode == True:
                    self.Pos_batch.extend(self.Pos)
                self.trace1 = []
                self.trace2 = []
                self.Qs1 = []
                self.Qs2 = []
                self.Hiddens = []
                self.TargetQ_meta = []
                self.Pos = []
                # reset
                done = False
                if reward_control == None:
                    self.reset(reward_control = np.random.randint(len(self.set_reward)), size = self.size)
                else:
                    self.reset(reward_control = reward_control, size = self.size)
                # init net
                self.hidden1 = self.net.initHidden()
                self.hidden2 = self.net.initHidden()
                self.action = self.net.initAction()

            for i in range(epochs):
                k += 1
                for t in range(self.size * 10):
                    _, state_t0, reward, done = self.step(self.maxplay, epsilon = epsilon, train_hidden = train_hidden, test = test, decode = decode)
                    if done == True:
                        Done()
                Done()
                
        # the reward for meta selector is target Q , the higer target Q , the bettter reward it could get
        def train_selector(self, rls_meta, trial):
            hiddens = torch.stack(self.Hiddens_batch).view(len(self.Hiddens_batch), -1)
            hiddens = torch.cat((hiddens, torch.ones(len(hiddens), 1)), 1)
            # print (self.TargetQ_meta_batch)
            targetQ_meta = torch.stack(self.TargetQ_meta_batch).view(len(self.TargetQ_meta_batch), -1)
            if trial == 0:
                rls_meta.LeastSquare(hiddens, targetQ_meta)
            else:
                rls_meta.update_beta(hiddens.clone(), targetQ_meta.clone(), trial)
            self.net.has.data = rls_meta.beta[:-1]
            self.net.bas.data = rls_meta.beta[-1].resize(1, 2)

        # if use random tensor , there is no memory leak,  if use hiddens and targets for beta but not really update weight, still leak,  so the error is in between
        def train_rls(self, rls1, rls2, trial):
            hiddens = torch.stack(self.Hiddens_batch).view(len(self.Hiddens_batch), -1)

            # kill memory because of that , but not shown here
            # if only doing least square, again no leak
            # for net1
            hiddens1 = hiddens[:, :512]
            hiddens1 = torch.cat((hiddens1, torch.ones(len(hiddens), 1)), 1)
            hiddens2 = hiddens[:, 512:]
            hiddens2 = torch.cat((hiddens2, torch.ones(len(hiddens), 1)), 1)
            targets1 = [t[0] for t in self.Targets_batch]
            targets2 = [t[1] for t in self.Targets_batch]
            targets1 = torch.cat(targets1, dim = 0).squeeze()
            targets2 = torch.cat(targets2, dim = 0).squeeze()
            # print (len(targets1), len(targets2))
            if trial == 0:
                rls1.LeastSquare(hiddens1, targets1)
            else:
                rls1.update_beta(hiddens1.clone(), targets1.clone(), trial)
            self.net.h2o1.data = rls1.beta[:-1]
            self.net.bo1.data = rls1.beta[-1].resize(1, 4)
            # net2
            if trial == 0:
                rls2.LeastSquare(hiddens2, targets2)
            else:
                rls2.update_beta(hiddens2.clone(), targets2.clone(), trial)
            self.net.h2o2.data = rls2.beta[:-1]
            self.net.bo2.data = rls2.beta[-1].resize(1, 4)
            # print (torch.norm(self.net.h2o2))


        def train_sl(self, rls, trial):
            hiddens = torch.stack(self.Hiddens_batch).view(len(self.Hiddens_batch), -1)
            hiddens = torch.cat((hiddens, torch.ones(len(hiddens), 1)), 1)
            poss = torch.stack(self.Pos_batch).squeeze()
            if trial == 0:
                rls.LeastSquare(hiddens, poss)
            else:
                rls.update_beta(hiddens, poss, trial)
            self.net.h2p_rls.data = rls.beta[:-1]
            self.net.bp_rls.data = rls.beta[-1].resize(1, 2)

        def experiment(self, rls_q1, rls_q2, rls_meta, iterations=10, epochs=20, epsilon=0.0,
                       reward_control=None, train_hidden=False,decode=False, size_range=[10],
                       test = False, train_selector = True):
            # initialize, might take data during test
            self.trace1 = []
            self.trace2 = []
            self.Qs1 = []
            self.Qs2 = []
            self.Hiddens = []
            self.TargetQ_meta = []
            self.Pos = []
            for i in range(iterations):
                self.episode(epochs=epochs, epsilon=epsilon, reward_control=reward_control, size_range=size_range,
                             prob=np.ones(len(size_range)) / len(size_range), train_hidden=train_hidden, test=test,
                             decode=decode)
                self.train_rls(rls_q1, rls_q2, i)
                if train_selector == True:
                    self.train_selector(rls_meta, i)
                self.Hiddens_batch = []
                self.Targets_batch = []
                self.TargetQ_meta_batch = []
                self.Pos_batch = []
            process = psutil.Process(os.getpid())
            print('clear session data', i, process.memory_info().rss)

    game = ValueMaxGame(grid_size=(15, 15), holes=holes, random_seed=4, set_reward=[(0.5, 0.25), (0.5, 0.75)], input_type =0, task = task)
    return game



