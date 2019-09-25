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

GOAL_VALUE = 1
EDGE_VALUE = -1
HOLE_VALUE = -1
VISIBLE_RADIUS = 1
GRID_SIZE = 8
NUM_HOLES = 4

GOAL_VALUE = 1
EDGE_VALUE = -1
HOLE_VALUE = -1
VISIBLE_RADIUS = 1
GRID_SIZE = 8
NUM_HOLES = 4

def Test(task, game, weight = 0, size = 15, test_size = 0, limit_set = 8):
    if weight != 0:
        game.net.load_state_dict(torch.load(weight))
    def TestBasic(game, reward_control= 0, size=15, test = test_size, limit_set = limit_set, start=[]):
        Rewards = 0
        iters = 0
        error = 0
        step = size // 15 + 1
        for j in np.arange(2 * VISIBLE_RADIUS, size + 2 * VISIBLE_RADIUS, step):
            for i in np.arange(2 * VISIBLE_RADIUS, size + 2 * VISIBLE_RADIUS, step):
                if start == []:
                    pos = (j, i)
                else:
                    pos = (start[0], start[1])
                game.reset(set_agent=pos, reward_control=reward_control, size=size, limit_set=limit_set, test = test)

                done = False
                game.hidden = game.net.initHidden()
                pos_r = game.Set_reward[game.reward_control]
                while not done:
                    _, state, reward, done = game.step(game.maxplay, epsilon=0.00, test=True)  # Down
                if i <= pos_r[1]:
                    path_optimal = np.abs(2 * VISIBLE_RADIUS - j) + np.abs(pos_r[0] - 2 * VISIBLE_RADIUS) + np.abs(
                        pos_r[1] - i)
                if i > pos_r[1]:
                    path_optimal = np.abs(2 * VISIBLE_RADIUS - j) + np.abs(pos_r[0] - 2 * VISIBLE_RADIUS) + np.abs(
                        game.grid_size[0] + 1 - i) \
                                   + np.abs(pos_r[1] - (game.grid_size[1] + 1))
                if reward == 1:
                    reward = path_optimal / game.t
                    if reward >= 1:
                        reward = 1
                else:
                    reward = reward
                Rewards += reward
                #             print (path_optimal/game.t)
                iters += 1
        game.Qs = []
        game.Hiddens = []
        game.Pos = []
        return Rewards / (iters)

    def TestBar(game, reward_control=0, cross=False, size=15, test = test_size, limit_set = limit_set, map_set=[], start=[]):
        Rewards = 0
        iters = 0
        error = 0
        for j in np.arange(2 * VISIBLE_RADIUS, size + 2 * VISIBLE_RADIUS):
            for i in np.arange(2 * VISIBLE_RADIUS, size + 2 * VISIBLE_RADIUS):
                if start == []:
                    pos = (j, i)
                else:
                    pos = (start[0], start[1])
                game.reset(set_agent=pos, reward_control=reward_control, size=size, limit_set=limit_set, test=test, map_set=map_set, train=False)
                done = False
                game.hidden = game.net.initHidden()
                pos_r = game.Set_reward[game.reward_control]
                while not done:
                    _, state, reward, done = game.step(game.maxplay, epsilon=0.00, test=True, cross=cross)  # Down
                if i <= pos_r[1]:
                    path_optimal = np.abs(2 * VISIBLE_RADIUS - j) + np.abs(pos_r[0] - 2 * VISIBLE_RADIUS) + np.abs(
                        pos_r[1] - i)
                if i > pos_r[1]:
                    path_optimal = np.abs(2 * VISIBLE_RADIUS - j) + np.abs(pos_r[0] - 2 * VISIBLE_RADIUS) + np.abs(
                        game.grid_size[0] + 1 - i) \
                                   + np.abs(pos_r[1] - (game.grid_size[1] + 1))
                if reward == 1:
                    reward = path_optimal / game.t
                    if reward >= 1:
                        reward = 1
                else:
                    reward = reward
                Rewards += reward
                #             print (path_optimal/game.t)
                iters += 1
        game.Qs = []
        game.Hiddens = []
        game.Pos = []
        return Rewards / (iters)

    def TestHole(game, reward_control= 0, cross=False, size=15, test = test_size, limit_set = limit_set, seed=0, holes = 50):
        Rewards = 0
        iters = 0
        error = 0
        step = size // 15 + 1
        game.seed = seed
        game.holes = holes
        for pos in zip(*np.where(game.grid.grid == 0)):
            game.reset(set_agent=pos, reward_control=reward_control, size=size, limit_set=limit_set, test=test,
                       train=False)
            j, i = game.agent.pos
            done = False
            game.hidden = game.net.initHidden()
            pos_r = game.Set_reward[game.reward_control]
            while not done:
                _, state, reward, done = game.step(game.maxplay, epsilon=0.00, test=True, cross=cross)  # Down
            if i <= pos_r[1]:
                path_optimal = np.abs(2 * VISIBLE_RADIUS - j) + np.abs(pos_r[0] - 2 * VISIBLE_RADIUS) + np.abs(
                    pos_r[1] - i)
            if i > pos_r[1]:
                path_optimal = np.abs(2 * VISIBLE_RADIUS - j) + np.abs(pos_r[0] - 2 * VISIBLE_RADIUS) + np.abs(
                    game.grid_size[0] + 1 - i) \
                               + np.abs(pos_r[1] - (game.grid_size[1] + 1))
            if reward == 1:
                reward = path_optimal / game.t
                if reward >= 1:
                    reward = 1
            else:
                reward = reward
            Rewards += reward
            #             print (path_optimal/game.t)
            iters += 1
        game.Qs = []
        game.Hiddens = []
        game.Pos = []
        return Rewards / (iters)

    def TestScale(game, reward_control= 0, cross=False, size = 50, test = test_size, limit_set = limit_set):
        Rewards = 0
        iters = 0
        error = 0
        for j in np.arange(2 * VISIBLE_RADIUS, size + 2 * VISIBLE_RADIUS):
            for i in np.arange(2 * VISIBLE_RADIUS, size + 2 * VISIBLE_RADIUS):
                game.reset(set_agent=(j, i), reward_control=reward_control, size=size, limit_set=limit_set, test=test)

                done = False
                game.hidden = game.net.initHidden()
                pos_r = game.Set_reward[game.reward_control]
                while not done:
                    _, state, reward, done = game.step(game.maxplay, epsilon=0.00, test=True, cross=cross)  # Down
                if i <= pos_r[1]:
                    path_optimal = np.abs(2 * VISIBLE_RADIUS - j) + np.abs(pos_r[0] - 2 * VISIBLE_RADIUS) + np.abs(
                        pos_r[1] - i)
                if i > pos_r[1]:
                    path_optimal = np.abs(2 * VISIBLE_RADIUS - j) + np.abs(pos_r[0] - 2 * VISIBLE_RADIUS) + np.abs(
                        game.grid_size[0] + 1 - i) \
                                   + np.abs(pos_r[1] - (game.grid_size[1] + 1))
                if reward == 1:
                    reward = path_optimal / game.t
                    if reward >= 1:
                        reward = 1
                else:
                    reward = reward
                Rewards += reward
                #             print (path_optimal/game.t)
                iters += 1
        print(pos_r)
        game.Qs = []
        game.Hiddens = []
        game.Pos = []
        return Rewards / (iters)

    def TestScale_x(game, reward_control = 0, cross=False, size=15, test = test_size, limit_set = limit_set, scale = 3, start=[]):
        Rewards = 0
        iters = 0
        error = 0
        for j in np.arange(2 * VISIBLE_RADIUS, size + 2 * VISIBLE_RADIUS):
            for i in np.arange(2 * VISIBLE_RADIUS, scale * size + 2 * VISIBLE_RADIUS):
                if start == []:
                    pos = (j, i)
                else:
                    pos = (start[0], start[1])
                game.reset(set_agent=pos, reward_control=reward_control, size=size, limit_set=limit_set, test=test,
                           scale = scale, train=False)

                done = False
                game.hidden = game.net.initHidden()
                pos_r = game.Set_reward[game.reward_control]
                while not done:
                    _, state, reward, done = game.step(game.maxplay, epsilon=0.00, test=True, cross=cross)  # Down
                if i <= pos_r[1]:
                    path_optimal = np.abs(2 * VISIBLE_RADIUS - j) + np.abs(pos_r[0] - 2 * VISIBLE_RADIUS) + np.abs(
                        pos_r[1] - i)
                if i > pos_r[1]:
                    path_optimal = np.abs(2 * VISIBLE_RADIUS - j) + np.abs(pos_r[0] - 2 * VISIBLE_RADIUS) + np.abs(
                        game.grid_size[0] + 1 - i) \
                                   + np.abs(pos_r[1] - (game.grid_size[1] + 1))
                if reward == 1:
                    reward = path_optimal / game.t
                    if reward >= 1:
                        reward = 1
                else:
                    reward = reward
                Rewards += reward
                #             print (path_optimal/game.t)
                iters += 1
        game.Qs = []
        game.Hiddens = []
        game.Pos = []
        return Rewards / (iters)

    def TestScale_y(game,reward_control = 0, cross=False, size=15, test = test_size, limit_set = limit_set, scale = 3, start=[]):
        Rewards = 0
        iters = 0
        error = 0
        for j in np.arange(2 * VISIBLE_RADIUS, scale * size + 2 * VISIBLE_RADIUS):
            for i in np.arange(2 * VISIBLE_RADIUS, size + 2 * VISIBLE_RADIUS):
                if start == []:
                    pos = (j, i)
                else:
                    pos = (start[0], start[1])
                game.reset(set_agent=pos, reward_control=reward_control, size=size, limit_set=limit_set, test=test,
                            train=False, scale = scale)

                done = False
                game.hidden = game.net.initHidden()
                pos_r = game.Set_reward[game.reward_control]
                while not done:
                    _, state, reward, done = game.step(game.maxplay, epsilon=0.00, test=True, cross=cross)  # Down
                if i <= pos_r[1]:
                    path_optimal = np.abs(2 * VISIBLE_RADIUS - j) + np.abs(pos_r[0] - 2 * VISIBLE_RADIUS) + np.abs(
                        pos_r[1] - i)
                if i > pos_r[1]:
                    path_optimal = np.abs(2 * VISIBLE_RADIUS - j) + np.abs(pos_r[0] - 2 * VISIBLE_RADIUS) + np.abs(
                        game.grid_size[0] + 1 - i) \
                                   + np.abs(pos_r[1] - (game.grid_size[1] + 1))
                if reward == 1:
                    reward = path_optimal / game.t
                    if reward >= 1:
                        reward = 1
                else:
                    reward = reward
                Rewards += reward
                #             print (path_optimal/game.t)
                iters += 1
        game.Qs = []
        game.Hiddens = []
        game.Pos = []
        return Rewards / (iters)
    

    def TestImplicit(game, reward_control = 0, cross=False, size=15, test = test_size, limit_set = limit_set):
        Rewards = 0
        iters = 0
        error = 0
        for j in np.arange(2 * VISIBLE_RADIUS, size + 2 * VISIBLE_RADIUS):
            for i in np.arange(2 * VISIBLE_RADIUS, size + 2 * VISIBLE_RADIUS):
                game.reset(set_agent=(j, i), reward_control=reward_control, size=size, limit_set=limit_set, test=test)

                done = False
                game.hidden = game.net.initHidden()
                pos_r = game.Set_reward[game.reward_control]
                while not done:
                    _, state, reward, done = game.step(game.maxplay, epsilon=0.00, test=True, cross=cross)  # Down
                if i <= pos_r[1]:
                    path_optimal = np.abs(2 * VISIBLE_RADIUS - j) + np.abs(pos_r[0] - 2 * VISIBLE_RADIUS) + np.abs(
                        pos_r[1] - i)
                if i > pos_r[1]:
                    path_optimal = np.abs(2 * VISIBLE_RADIUS - j) + np.abs(pos_r[0] - 2 * VISIBLE_RADIUS) + np.abs(
                        game.grid_size[0] + 1 - i) \
                                   + np.abs(pos_r[1] - (game.grid_size[1] + 1))
                if reward == 1:
                    reward = path_optimal / game.t
                    if reward >= 1:
                        reward = 1
                else:
                    reward = reward
                Rewards += reward
                #             print (path_optimal/game.t)
                iters += 1
        game.Qs = []
        game.Hiddens = []
        game.Pos = []
        return Rewards / (iters)
    performance = 0
    for context in [0, 1]:
        if task == 'basic':
            performance += TestBasic(game, reward_control = context, size = size)
        if task == 'hole':
            performance += TestHole(game, reward_control = context, holes=50, seed = 0, size = size, )
        if task == 'bar':
            performance += TestBar(game, reward_control = context, size = size)
        if task == 'scale':
            performance += TestScale(game, reward_control = context, size = size)
        if task == 'scale_x':
            performance += TestScale_x(game, scale = 3, reward_control = context, size = size)
        if task == 'scale_y':
            performance += TestScale_y(game, scale = 3, reward_control = context, size = size)
        if task == 'scale_xy':
            performance += TestScale_xy(game, scale = (3, 3), reward_control = context, size = size)
        if task == 'implicit':
            performance += TestImplicit(game, reward_control = context, size = size)
    print (task + '', 'performance', performance/2)
    # plt.matshow(game.grid.grid)
    return performance/2

