import numpy as np
from copy import copy

class Poilcy_iteration():

    def __init__(self):
        self.width = 5
        self.height = 5

        self.V = np.zeros(shape=(self.height,self.width))
        self.V_old = np.zeros(shape=(self.height,self.width), dtype=np.float16)

        self.actions = {"left":(0,-1),
                        "right":(0,1),
                        "up":(-1,0),
                        "down":(1,0)}
        self.terminal_state = [(0,0), (self.width-1,self.height-1)]#, (self.width-1, 0), (0,self.height-1)]

        self.gamma = 0.9
        #Random policy
        self.policy = np.ones(shape=(len(self.actions),self.height,self.width)) * 0.25
        self.Q = np.zeros((len(self.actions),self.height,self.width))
        self.reward = -1

    def evaluation(self,policy):
        # Get value function
        self.V = np.zeros(shape=(self.height,self.width))
        for x in range(self.height):
            for y in range(self.width):
                for idx, action in enumerate(self.actions):

                    try:
                        if (x,y) in self.terminal_state:
                            self.V[x][y] = 0
                        else:
                            if x + self.actions[action][0] < 0 or y + self.actions[action][1] < 0 or \
                                    x + self.actions[action][0]>= self.height or y + self.actions[action][1] >= self.width :
                                self.V[x][y] += policy[idx][x][y] * (self.reward + self.gamma * self.V_old[x][y])

                            else:
                                self.V[x][y] += policy[idx][x][y] *\
                                                ( self.reward+ self.gamma* self.V_old[x+self.actions[action][0]][y+self.actions[action][1]])


                    except IndexError:
                        print("INDEXERROR",idx,x,y)
        self.V_old = copy(self.V)

        #Get action-value function
        for x in range(self.height):
            for y in range(self.width):
                for idx, action in enumerate(self.actions):
                    try:
                        if (x,y) in self.terminal_state:
                            self.Q[idx][x][y] = self.reward + self.gamma * self.V[x][y]
                        else:
                            if x + self.actions[action][0] < 0 or y + self.actions[action][1] < 0 or \
                                    x + self.actions[action][0] >= self.height or y + self.actions[action][1] >= self.width:
                                self.Q[idx][x][y] = self.reward + self.gamma * self.V[x][y]

                            else:
                                self.Q[idx][x][y] = self.reward + self.gamma * self.V[x + self.actions[action][0]][y + self.actions[action][1]]

                    except IndexError:
                        pass



        return self.V, self.Q

    def improvement(self, q_func):

        q_argmax = np.argwhere(q_func == np.amax(q_func,axis=0))
        get_num_argmax = np.zeros(shape=(self.width,self.height))
        for idx, x, y in q_argmax:
            get_num_argmax[x][y] += 1
        prob =  1/get_num_argmax

        self.policy = np.zeros(shape=(len(self.actions),self.height,self.width))
        for idx,x,y in q_argmax:
            self.policy[idx][x][y] = prob[x][y]

        return self.policy


policy = Poilcy_iteration()
pol = policy.policy
iter = 300
for _ in range(iter):
    V,Q = policy.evaluation(pol)
    print("V\n",V)
    #print("Q\n", Q)

    pol = policy.improvement(Q)
    #print("POLICY\n",pol)


