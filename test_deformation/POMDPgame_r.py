import numpy as np
from itertools import count
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.nn import init

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.animation
from IPython.display import HTML

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
# enviroment and behaviour, without moving out of wall 
class Grid():
    def __init__(self, n_holes = 4, grid_size = GRID_SIZE, random_seed = 0, set_reward = 0, ry = 5, rx = 5, train = True, direction = None):
        random.seed(random_seed)
        # check type of grid,  attention type is not a string
        if type(grid_size) == int:
            self.grid_size_y = self.grid_size_x = grid_size
        elif type(grid_size) == tuple:
            y, x = grid_size
            self.grid_size_y = y
            self.grid_size_x = x
        self.n_holes = n_holes
        #  Define the surronding using padding 
        padded_size_y = self.grid_size_y + 4 * VISIBLE_RADIUS
        padded_size_x = self.grid_size_x + 4 * VISIBLE_RADIUS
        #  intialize grid with zeros, attention y and x order
        self.grid = np.zeros((padded_size_y, padded_size_x)) # Padding for edges
        #  intialize border with predefined negative values
        self.grid[0:2*VISIBLE_RADIUS, :] = EDGE_VALUE
        self.grid[-2*VISIBLE_RADIUS:, :] = EDGE_VALUE
        self.grid[:, 0:2*VISIBLE_RADIUS] = EDGE_VALUE
        self.grid[:, -2*VISIBLE_RADIUS:] = EDGE_VALUE
        if set_reward == 0:
            gy = random.randint(0, self.grid_size_y) + 2*VISIBLE_RADIUS
            gx = random.randint(0, self.grid_size_x) + 2*VISIBLE_RADIUS
            while self.grid[gy,gx] == HOLE_VALUE:    
                gy = random.randint(0, self.grid_size_y) + 2*VISIBLE_RADIUS
                gx = random.randint(0, self.grid_size_x) + 2*VISIBLE_RADIUS
            self.grid[gy,gx] = GOAL_VALUE
        else:
            for pos_reward in set_reward:
                radius = 0 
                self.grid[pos_reward[0]-radius: pos_reward[0]+1+radius, pos_reward[1]-radius: pos_reward[1]+1+radius] = GOAL_VALUE
    
    def visible1(self, pos):
        # observable area is the squre around the agent, so 3x3 region , problem is when the agent is going to the 
        # edge and corner of the grid
        y, x = pos
        y_relative = y * 19./(self.grid_size_y + 4)
        x_relative = x * 19./(self.grid_size_x + 4)
        visible = self.grid[y-VISIBLE_RADIUS:y+VISIBLE_RADIUS+1, x-VISIBLE_RADIUS:x+VISIBLE_RADIUS+1]
        if np.sum(visible) != 0 and (x ==2 or x == self.grid_size_x + 1): 
             visible = np.multiply(visible, y_relative * np.ones((3,3)) )
        elif np.sum(visible) != 0 and (y ==2 or y == self.grid_size_y + 1): 
             visible = np.multiply(visible, x_relative * np.ones((3,3)) )
        return visible
    
    def visible(self, pos):
        # observable area is the squre around the agent, so 3x3 region , problem is when the agent is going to the 
        # edge and corner of the grid
        y, x = pos
        visible = self.grid[y-VISIBLE_RADIUS:y+VISIBLE_RADIUS+1, x-VISIBLE_RADIUS:x+VISIBLE_RADIUS+1]
        return visible
    
    
class Agent():
    def reset(self, grid_size, set_agent = 0):
        if type(grid_size) == tuple:
            self.grid_size_y,  self.grid_size_x = grid_size
        else:
            self.grid_size_y = self.grid_size_x = grid_size
        # position initialize
        if set_agent == 0:
            random.seed()
            self.pos = (np.random.randint(self.grid_size_y) + 2*VISIBLE_RADIUS, np.random.randint(self.grid_size_x) + 2*VISIBLE_RADIUS)
        else : 
            self.pos = set_agent
        
    # moves in four direction, Implement the relfective behaviour when arriving upon the wall  ,  one way is to let the agent ran randomly when it clicks to wall, the other way is reflective boundary.  
    def act(self, action):
        # Move according to action: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        y, x = self.pos

        # up 
        if action == 0: y -= 1
        # right
        elif action == 1: x += 1
        # down
        elif action == 2: y += 1
        # left
        elif action == 3: x -= 1
        self.pos = (y, x)
        
            
            
        

        

# The setting of enviroment basically makes everything moves inside, there is little regularity, 
# 
class Game():
    # 初始化，初始grid和agent
    def __init__(self, grid_size = 8, holes = 4, discount = 0.99, time_limit = 200, random_seed = 0, set_reward = 0, input_type = 0):
        self.discount = discount
        self.time_limit = time_limit
        self.grid_size = grid_size
        self.set_reward = set_reward
        self.seed = random_seed
        Set_reward = []
        # for the reward draw
        for pos in self.set_reward:
            y, x = pos
            Set_reward.append((2 * VISIBLE_RADIUS + int(self.grid_size[0] * y), 2 * VISIBLE_RADIUS + int(self.grid_size[1] * x)))
        self.grid = Grid(n_holes = holes, grid_size = grid_size, random_seed = self.seed, set_reward = Set_reward, train = False)
        self.agent = Agent()
        self.History = []
        self.values = self.grid.grid.copy()
        self.values.fill(0)
        self.t = 0
        self.seed = random_seed
        self.seed_range = 2
        self.holes = holes
        self.input_type = input_type
    # set limit sizes 
    def reset(self, set_agent = 0, action = True, reward_control = 0, size = None, size_range = np.arange(10, 51, 10), prob = 5 * [0.2] , limit_set = 8, test = None, context = (0.5, 0.25), train = True, map_set = []):
        """Start a new episode by resetting grid and agent"""
        # reset the reward so that it will not be erased in time 
        # set size
        if size == None:
            size = size_range[np.random.choice(len(size_range), p=prob)]

        else:
            size = size
        self.size = size
        # set reward
        if len(self.set_reward) != 0:
            self.time_limit = int(self.size * limit_set)
            radius = self.size//10 - 1
            if test!= None:
                radius = test
            k = np.random.randint(1, 4)
            self.Set_reward = []
            for pos in self.set_reward:
                y, x = pos
                self.Set_reward.append((2 * VISIBLE_RADIUS + int(k * size * y), 2 * VISIBLE_RADIUS + int(size * x)))
            self.grid = Grid(n_holes = 0, grid_size = (k * self.size, self.size), random_seed = 0, set_reward = self.Set_reward, train = train)
            self.grid_size = (self.grid.grid_size_y, self.grid.grid_size_x)
            if len(map_set) != 0:
                self.grid.grid = map_set
            self.grid.grid[np.where(self.grid.grid == 1)] = 0
            self.reward_control = reward_control
            # this variable is used to select which reward chosen as target
            self.pos_reward = self.Set_reward[reward_control]
            # select the reward 
            self.grid.grid[self.pos_reward[0]-radius: self.pos_reward[0]+1+radius, self.pos_reward[1]-radius: self.pos_reward[1]+1+radius] = 1
        else:
            y, x = context
            self.grid.grid[np.where(self.grid.grid == 1)] = 0
            self.pos_reward = (2 * VISIBLE_RADIUS + int(self.size * y), 2 * VISIBLE_RADIUS + int(self.size * x))
        # set position 
        self.agent.reset(self.grid_size, set_agent = set_agent)
        self.hidden = self.net.initHidden()
        if action == True:
            self.action = self.net.initAction()
        self.t = 0
        self.reward = 0
 
           
    
    @property
    def visible_state(self):
        """Return the visible area surrounding the agent, and current agent health"""
        if self.input_type == 0:
            visible = self.grid.visible(self.agent.pos)
        elif self.input_type == 1:
            visible = self.grid.visible1(self.agent.pos)
        return visible.flatten()
    
    def stimulus(self, pos):
        visible = self.grid.visible1(pos)
        return visible.flatten()
    
    @staticmethod    
    def sample():
        # choose between 0, 1,2,3
        np.random.seed()
        return np.random.randint(0,4)


    
def animate(history):
    frames = len(history)
    print("Rendering %d frames..." % frames)
    fig = plt.figure(figsize=(6, 2))
    fig_grid = fig.add_subplot(111)
  
    
    def render_frame(i):
        grid, time = history[i]
        # Render grid
        fig_grid.matshow(grid, vmin=-1, vmax=1, cmap='jet')

    anim = matplotlib.animation.FuncAnimation(
        fig, render_frame, frames=frames, interval=100
    )

    plt.close()
    display(HTML(anim.to_html5_video()))    
    
def animate_group(History):
    k = 0
    fig = plt.figure(figsize=(6, 6))
    for history in History:
        frames = len(history)
        print("Rendering %d frames..." % frames)
        
        fig_grid = fig.add_subplot(151 + k)
        k += 1

        def render_frame(i):
            grid, time = history[i]
            # Render grid
            fig_grid.matshow(grid, vmin=-1, vmax=1, cmap='jet')

        anim = matplotlib.animation.FuncAnimation(
            fig, render_frame, frames=frames, interval=100
        )

        plt.close()
    display(HTML(anim.to_html5_video()))   
    
def move(pos, act):
    i,j = pos
    pos_possible = [(i-1,j),(i,j+1),(i+1,j),(i,j-1)]
    return pos_possible[act]
 

    
    
    