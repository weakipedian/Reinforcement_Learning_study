import numpy as np
from copy import copy

class Poilcy_iteration():

    def __init__(self):
        self.width = 3
        self.height = 3

        self.V = np.zeros(shape=(self.width,self.height))
        self.V_old = np.zeros(shape=(self.width,self.height), dtype=np.float16)

        self.actions = {"left":(0,-1),
                        "right":(0,1),
                        "up":(-1,0),
                        "down":(1,0)}

        self.gamma = 0.9
        #Random policy
        self. policy = np.ones((len(self.actions),self.width,self.height)) * 0.25
        self.Q = np.zeros((len(self.actions),self.width,self.height))
        self.reward = -1

    def evaluation(self):
        self.V = np.zeros(shape=(self.width,self.height))
        for x in range(self.width):
            for y in range(self.height):
                for idx, action in enumerate(self.actions):
                    try:
                        if x + self.actions[action][0] < 0 or y + self.actions[action][1] < 0 :
                            pass
                        else:
                            self.V[x][y] += self.policy[idx][x+self.actions[action][0]][y+self.actions[action][1]]* ( self.reward+ self.gamma* self.V_old[x+self.actions[action][0]][y+self.actions[action][1]])

                    except IndexError:
                        pass
        self.V_old = copy(self.V)
        return self.V


policy = Poilcy_iteration()
for _ in range(500):
    print(policy.evaluation())


