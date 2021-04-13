# based on below link
# https://towardsdatascience.com/beginners-guide-to-custom-environments-in-openai-s-gym-989371673952
import gym
from matplotlib import pyplot as plt
import numpy as np
import time

'''
action -
UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3

map - 
bottom left is (0, 0). right and upward indicates +x and +y each
'''


ACTION_TO_COORDINATE = {0: (0, 1), 1: (1, 0), 2: (-1, 0), 3: (0, -1)}
UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3
X_MIN, X_MAX = 0, 9
Y_MIN, Y_MAX = 0, 6
TERMINATION_STATE =  np.array([7, 3])

class WindyGridWorld(gym.Env):

    def __init__(self):
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Discrete(2)
        self.state = np.array([0, 0])

        # plot member
        self.render_on = False

        # max step
        self.max_step = 200
        self.n_step = 0

    def step(self, action):
        ## update state
        # update state with action
        action = int(action)
        self.state = self.state + ACTION_TO_COORDINATE[action]

        # check if it's windy
        if 5 <= self.state[0] <= 7:
            self.state = self.state + ACTION_TO_COORDINATE[UP]
        # check boundary
        self.state[0] = self.clip(self.state[0], X_MIN, X_MAX)
        self.state[1] = self.clip(self.state[1], Y_MIN, Y_MAX)

        ## set reward and done
        self.n_step += 1
        if set(self.state) == set(TERMINATION_STATE):
            reward = 1
            done = True
        elif self.n_step >= self.max_step:
            reward = -1
            done = True
        else:
            reward = -1
            done = False

        info = {}

        return self.state, reward, done, info

    def reset(self):
        self.state = np.array([0, 3])
        self.n_step = 0
        return self.state

    def clip(self, v, min, max):
        v = min if v < min else v
        v = max if v > max else v
        return v

    def close(self):
        exit(0)

    def render(self):
        if not self.render_on:
            #plt.ion()
            self.render_on = True
            self.fig = plt.figure(figsize=(8, 6))
            self.ax = self.fig.add_axes([0,0,1,1])

        plt.cla()
        self.ax.grid()
        self.ax.set_xticks(np.arange(X_MIN, X_MAX+1, 1))
        self.ax.set_yticks(np.arange(Y_MIN, Y_MAX+1, 1))
        self.ax.set_xlim([X_MIN, X_MAX + 1])
        self.ax.set_ylim([Y_MIN, Y_MAX + 1])

        # draw arrow
        self.ax.plot([6.5, 6.5], [0, Y_MAX + 1], 'b', linewidth=20)
        self.ax.plot([5.5, 6.5], [Y_MAX - 1, Y_MAX + 1], 'b', linewidth=20)
        self.ax.plot([7.5, 6.5], [Y_MAX - 1, Y_MAX + 1], 'b', linewidth=20)

        # draw goal
        g_draw = TERMINATION_STATE + (0.5, 0.5)
        self.ax.scatter(*g_draw, facecolor='green', s=400)

        # draw state
        s_draw = self.state + (0.5, 0.5)
        self.ax.scatter(*s_draw, facecolor='black')

        plt.draw()
        plt.pause(0.001)

    def render_qvalues(self, qvalues_dict): # gym API와는 맞지 않지만 편의상 여기에 구현...
        if not self.render_on:
            #plt.ion()
            self.render_on = True
            self.fig = plt.figure(figsize=(8, 6))
            self.ax = self.fig.add_axes([0,0,1,1])

        plt.cla()
        self.ax.grid()
        self.ax.set_xticks(np.arange(X_MIN, X_MAX + 1, 1))
        self.ax.set_yticks(np.arange(Y_MIN, Y_MAX + 1, 1))
        self.ax.set_xlim([X_MIN, X_MAX + 1])
        self.ax.set_ylim([Y_MIN, Y_MAX + 1])

        for key, qvalue in qvalues_dict.items():
            state, action = key
            coord = np.array(state) + (0.4, 0.45)
            coord = coord + np.array(ACTION_TO_COORDINATE[action]) / 3
            self.ax.text(*coord, '{:.1f}'.format(qvalue), fontsize=8)


        plt.show()

