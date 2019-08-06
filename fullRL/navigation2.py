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

import POMDPgame_r
from POMDPgame_r import*

import RNN 
from RNN import * 




class ValueMaxGame(Game): 
    
    def __init__(self, e = 0.2, holes = 3, grid_size = 8, random_seed = 0, set_reward = 0, time_limit = 200, input_type = 0, action_control = 1, lam = 0.5, discount = 0.99, alpha = 0.5):
        Game.__init__(self, discount = discount, grid_size = grid_size, holes = holes, time_limit = time_limit, random_seed = random_seed,
                     set_reward = set_reward, input_type = input_type)
        # need to have randomness
        self.e = e
        self.net = RNN(9, 512, 4, k_action = 1)
        self.decoder = decoder(512, self.grid_size[0] + self.grid_size[1] + 8)
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
        self.max_size = 20
        if len(set_reward) != 0:
            for i in range(len(set_reward)):
                self.y_mid += self.set_reward[i][0] 
                self.x_mid += self.set_reward[i][1] 
            self.y_mid /= len(set_reward)
            self.x_mid /= len(set_reward)
           
    def sample(self):
        # choose between 0, 1,2,3
        np.random.seed()
        return np.random.randint(0,4)
    # logis function for target function
    def placefield_dirac(self, pos, beta, size = 15): 
        pos_ = (pos[0] - self.y_mid, pos[1] - self.x_mid)
#         print (pos_, self.set_reward)
#         print (self.grid_size)
        field =np.zeros((2, size+4))
#         print (len(field))
        for k in range(2):
            for i in range(field.shape[1]): 
            # distance generation 
                pos_relative = int(pos[k] * 19./(size + 4))
                if i == pos_relative:
                    field[k, i] =  1
#                 print (i - pos[k])e
        field = torch.from_numpy(field).resize(1, 2 * (size+4)).float()
        return field
    # logis function for pretrain
    def placefield_pretrain(self): 
        field =np.zeros((2, self.max_size + 4))
        for k in range(2):
            for i in range(field.shape[1]): 
            # distance generation 
                field[k, i] =  (i- self.agent.pos[k]) ** 2 
        # gaussian density, but before exponential to help learning identity mapping input to output
        field = - 0.1 * torch.from_numpy(field).resize(1, 2 * (self.max_size + 4)).float()
        return field 

    # logis function for q leanrning
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
    
    def maxplay(self, state, cross = False):
        # Move according to action: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        # value for four action     
        action0 = self.action.clone()
        if cross == False: 
            self.values, self.hidden = self.net(state, self.hidden, self.action, self.placefield(self.pos_reward))
        else:
            self.values, self.hidden = self.net(state, self.hidden, self.action, self.placefield(self.pos_reward_))
        action = self.values.data.cpu().numpy().argmax()
        # action to state
        if np.random.random()< self.e:
            action = self.sample()
        self.action = torch.eye(4)[action].resize(1, 4)
        return action, action0  
#  decode is binary  
    def decode(self):
        pos = self.hidden.matmul(self.net.h2p_rls) + self.net.bp_rls
        return pos  
    
    # decode by sgd classifier
    def decode_sgd(self):
        Pos_p = self.hidden.matmul(self.decoder.h2p).data.numpy()  + self.decoder.bp.data.numpy()
        yp = np.array([np.argmax(p[:19]) for p in Pos_p])
        xp = np.array([np.argmax(p[19:]) for p in Pos_p]) 
        return (yp, xp)
    
    # test is for testing phase, decode is to train decoder   
    def step(self, policy, epsilon = 'default', record = False, test = False, cross = False, train_hidden = False, decode = False, decode_sgd = False):
        if epsilon != 'default': 
            self.e = epsilon  
        self.t += 1
        """enviroment dynamics Update state (grid and agent) based on an action"""
        # state to action
        state_t0 = self.visible_state
        pos0 = self.agent.pos
        # network dynamics and decision
        action, action0 = policy(Variable(torch.FloatTensor(state_t0)).resize(1,9), cross = cross)
        # action to state
        self.agent.act(action) 
        state_t1 = self.visible_state
        pos1 = self.agent.pos
        # check stop condition
        def wall(pos1):
            y, x = pos0  
            # wall detection
            if self.grid.grid[pos1] < 0:
                self.agent.pos = pos0
                pos_possible = [pos for pos in [(y-1,x),(y+1,x),(y,x-1),(y,x+1)] if self.grid.grid[pos] >= 0]
                self.agent.pos = pos_possible[np.random.randint(len(pos_possible))]
                pos1 = self.agent.pos
#                # up 
#                 if pos1[0] - y == -1:
#                     action = 0
#                 # down
#                 elif pos1[0] - y == 1:
#                     action = 2
#                 # right
#                 elif pos1[1] - x == 1:
#                     action = 1
#                 # left 
#                 elif pos1[1] - x == -1:
#                     action = 3
#                 self.action = torch.eye(4)[action].resize(1, 4)    
                self.t += 1  
                return True
            else:
                return False
        def rewarding(pos1):
            collision = wall(pos1)
            # punishment by cold water or click
            if collision == True:
                reward = -0.5 
            else:
                reward = -0.01 
            if self.grid.grid[pos1]>0:
                reward = self.grid.grid[pos1]
            # death  
            elif self.t >= self.time_limit:
                reward = -1
        # Check if agent won (reached the goal) or lost (health reached 0)
        # attention! 需要括号， 否则reward会被更新
            done =(reward>0 or self.t >= self.time_limit)
            return reward, done  
        reward, done = rewarding(pos1)
        # update value function
        def TD(decode = False):
            realQ,_  = self.net(Variable(torch.FloatTensor(state_t1)).resize(1,9), self.hidden, self.action, self.placefield(self.pos_reward))
            # target Q is for state before updated, it only needs to update the value assocate with action taken
            targetQ = self.values.clone().detach()
            # new Q attached with the new state 
            # max of q for calculating td error
            Qmax = torch.max(realQ) 
            if done != True:
                delta  =  torch.FloatTensor([reward]) + self.discount*Qmax  - targetQ[0, action]
        # the virtual max action reaches terminal , there is no q max term assciate with next max value 
            elif done == True:
                delta  =  torch.FloatTensor([reward]) - targetQ[0, action]
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
        # record position and hidden state, control gradient here
        self.Hiddens.append(self.hidden)
        if train_hidden == True or decode_sgd == True:
            self.Pos.append(self.placefield(self.agent.pos))
        elif decode == True:
            self.Pos.append(torch.FloatTensor([pos0[0], pos0[1]]).resize(1, 2))
        return pos0, state_t0, reward, done
    

    
    def episode(self, epochs = 10, epsilon = 'default', reward_control = None, size_range = (10, 20), prob = 5 * [0.2], train_hidden = False, test = False, decode = False, decode_sgd = False):   
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
        self.Targets_batch = []
        self.Pos_batch = []
        def Done(decode = decode):
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
                self.reset(reward_control = np.random.randint(len(self.set_reward)), size = self.size)
            else:
                self.reset(reward_control = reward_control, size = self.size)    
                
#         process = psutil.Process(os.getpid())
#         print('start episode', process.memory_info().rss) 
        for i in range(epochs):
            k += 1
            for t in range(self.size * 10):
                _, state_t0, reward, done = self.step(self.maxplay, epsilon = epsilon, train_hidden = train_hidden, test = test, decode = decode, decode_sgd = decode_sgd)
                if done == True:
                    Done()
            Done()


    # if use random tensor , there is no memory leak,  if use hiddens and targets for beta but not really update weight, still leak,  so the error is in between 
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
    # train for Q readout
    def train_sgd(self, lr_rate = 1e-5, batch_size = 1):
        hiddens = torch.stack(self.Hiddens_batch).view(len(self.Hiddens_batch), -1)
        targets = torch.stack(self.Targets_batch).squeeze()
        # pair h and q
        data = [(h, p) for h, p in zip(hiddens, targets)]
        trainloader = torch.utils.data.DataLoader(data, batch_size = batch_size)
        optimizer = torch.optim.Adam(
                [ 
                {'params': self.net.h2h, 'lr': lr_rate, 'weight_decay':0},
                {'params': self.net.bh, 'lr': lr_rate, 'weight_decay':0},
                {'params': self.net.h2o, 'lr': lr_rate, 'weight_decay':0},
                {'params': self.net.bo, 'lr': lr_rate, 'weight_decay':0},
                ]
        )
        # for a number of iterations, do gradient descent   
        for k in range(10):
            for i, data in enumerate(trainloader, 0):  
                hidden, labels = data
                output = hidden.matmul(self.net.h2o) + self.net.bo
#                 print (predicts.size(), labels.size())
                # cross entropy of mini batch
                loss = torch.sum((output - labels) ** 2)
                # print (output, labels)
                self.Loss += loss
            self.Loss.backward(retain_graph = True)
            optimizer.step()
            self.net.zero_grad()
            print ('loss', self.Loss)
            self.Loss = 0
    def experiment(self, rls_sl, epochs = 20, epsilon = 0.5, reward_control = None, train_hidden = False, train_q = True, decode = False, size_range = [10], test = False):
        # initialize, might take data during test
        self.trace = []
        self.Qs = []
        self.Hiddens = []
        self.Pos = []
        self.episode(epochs = epochs, epsilon = epsilon, reward_control = reward_control, size_range = size_range, prob = np.ones(len(size_range))/len(size_range), train_hidden = train_hidden, test = test, decode = decode)
        if train_hidden == False and train_q == False:
            self.train_sl(rls_sl)
        elif train_hidden == False and train_q == True:
            self.train_sgd()
        self.Hiddens_batch = []
        self.Targets_batch = []
        self.Pos_batch = []
        process = psutil.Process(os.getpid())
        print('clear session data', process.memory_info().rss)
        
    def experiment_decode(self, iterations = 10, epochs = 10, epsilon = 0.0, reward_control = 0, size_range = [15]):
        # initialize, might take data during test
        self.trace = []
        self.Qs = []
        self.Hiddens = []
        self.Pos = []
        for i in range(iterations):
            # store the place information in vector format, so train hidden = true   
            self.episode(epochs = epochs, epsilon = epsilon, reward_control = reward_control, size_range = size_range, prob = np.ones(len(size_range))/len(size_range), test = True, decode = False, decode_sgd = True)
            self.train_sgd(control = reward_control)
            self.Hiddens_batch = []
            self.Targets_batch = []
            self.Pos_batch = []
            if i%1 == 0:
                print ('loss', self.Loss)
                self.Loss = 0
    

def Test(game, weights = 0, reward_control = [0], cross = False, size = 15, test = 1, limit_set = 2, matrix = False, epsilon = 0):
    if weights != 0: 
        game.net.load_state_dict(torch.load(weights))
    Rewards = 0
    iters = 0
    error = 0
    step = size//15 + 1
    Rewards_matrix = []
    for j in np.arange (2 * VISIBLE_RADIUS, size + 2 * VISIBLE_RADIUS, step):
        for i in np.arange (2 * VISIBLE_RADIUS, size + 2 * VISIBLE_RADIUS, step):
            game.reset(set_agent = (j,i), reward_control = reward_control, size = size, limit_set = limit_set, test = test)

            done = False
            game.hidden = game.net.initHidden()
            pos_r = game.Set_reward[game.reward_control]
            while not done:
                _, state, reward, done = game.step(game.maxplay, epsilon = epsilon, test = True, cross = cross) # Down
            # suppose goes up, and return to reward in middle and goe left 
            if i<=pos_r[1]:
                path_optimal = np.abs(2 * VISIBLE_RADIUS - j) + np.abs(pos_r[0] - 2 * VISIBLE_RADIUS) + np.abs(pos_r[1] - i)
            if i>pos_r[1]:
                path_optimal = np.abs(2 * VISIBLE_RADIUS - j) + np.abs(pos_r[0] - 2 * VISIBLE_RADIUS) + np.abs(game.grid_size[0]+1 - i)\
                + np.abs(pos_r[1] - (game.grid_size[1]+1))
            if reward == 1:
                reward = path_optimal/game.t
                if reward >=1:
                    reward = 1
            else:
                reward = reward
#             reward = 0.5 * (reward + 1)
            Rewards_matrix.append(reward)
            Rewards += reward
#             print (path_optimal, game.t, pos_r)
            iters += 1
    game.Qs = []
    game.Hiddens = []
    game.Pos = []
    if matrix == False:
        return Rewards/(iters)
    else:
        return Rewards/(iters), Rewards_matrix 



def decodetest(game, weights = 0, epsilon = 0, reward_control = [0], size = 15, sgd = False):
    if weights != 0: 
        game.net.load_state_dict(torch.load(weights))
    # for the decoding upon each location 
    decodes = np.ones((size + 4, size + 4)) * 0
    visit = np.ones((size + 4, size + 4)) * 1e-5
    step = 1
    Dist = []
    cross_entropy = 0 
    N = 0
    for j in range (2 * VISIBLE_RADIUS, size + 2 * VISIBLE_RADIUS):
        for i in range (2 * VISIBLE_RADIUS, size + 2 * VISIBLE_RADIUS):
            game.reset(set_agent = (j, i), reward_control = reward_control, size = size)
            done = False
            start = 0
            game.hidden = game.net.initHidden()
            Y, X, Yp, Xp = [], [], [], []
            while not done:
                # not let the data to accumulate by TD, so test = true
                pos0, state, reward, done  = game.step(game.maxplay, epsilon = epsilon, test = True) # Down
                if sgd == False:
                    pos_rls = game.decode()
                    y, x =  pos_rls.data.numpy()[0]
                    if reward_control == 0:  
                        pos_log_prob = game.hidden.matmul(game.net.h2pr0) + game.net.bpr0 
                    else:
                        pos_log_prob = game.hidden.matmul(game.net.h2pr1) + game.net.bpr1
                else:
                    pos = game.decode_sgd()
                    y, x = pos
                Y.append(pos0[0])
                X.append(pos0[1])
                Yp.append(y)
                Xp.append(x)
                # start only when there is visual stimulus 
                if np.sum(state) != 0:
                    start = 1
                # count manhaton distance between real and predicted 
                if start == 1:
    #                     print (pos.data.numpy()[0], game.agent.pos[0])
                    manhantondist = np.abs((y - pos0[0])) + np.abs((x - pos0[1]))
                    decodes[pos0] += manhantondist
                    visit[pos0] += 1
                    Dist.append(manhantondist)
                    # predict probability 
                    q = F.softmax(pos_log_prob.view(-1), dim = 0)
#                     q = F.softmax(game.placefield((y, x)).view(-1), dim = 0)
                    # real target
                    p = game.placefield_dirac(pos0, beta = 1e5, size = 20).view(-1)
#                     q = F.softmax(game.placefield_dirac((y, x), beta = 1e5, size = 20).view(-1), dim = 0)
                    cross_entropy += -torch.sum(p * torch.log(q + 1e-5)) 
                    N += 1
            game.Qs = []
            game.Hiddens = []
            game.Pos = []
    return Dist, decodes[2*VISIBLE_RADIUS:size+2*VISIBLE_RADIUS, 2*VISIBLE_RADIUS:size+2*VISIBLE_RADIUS], visit[2*VISIBLE_RADIUS:size+2*VISIBLE_RADIUS, 2*VISIBLE_RADIUS:size+2*VISIBLE_RADIUS], cross_entropy.data.numpy()/N
    