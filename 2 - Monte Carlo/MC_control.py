## implement GLIE MC Carlo Control
# Policy evaluation : update q-value with MC prediction in every-step.
# Policy improvement : e-greedy policy improvement

from windy_grid_world import WindyGridWorld
import random


class MonteCarloControlAgent:
    def __init__(self, env):
        self.discount_factor = 0.95
        self.queue = []
        self.qvalues = {}
        self.counter = {}
        self.env = env

    def clear(self):
        self.queue = []

    def add_info(self, state, action, reward):
        self.queue.append((state, action, reward))

    def update_q_with_mc_prediction(self):
        '''
        This should be called after one episode
        :return:
        '''
        for i in range(len(self.queue)): # we can remove inner loop by reserve iteration, but i'm lazy to do that
            s, a, r = self.queue[i]

            # calc returns
            _, _, Gt = self.queue[-1]
            for j in range(len(self.queue)-1, i, -1):
                _, _, r = self.queue[j]
                Gt = r + self.discount_factor * Gt

            # update qvalues
            self.counter[(tuple(s), a)] = self.counter.get((tuple(s), a), 0) + 1
            self.qvalues[(tuple(s), a)] = self.qvalues.get((tuple(s), a), 0) + \
                                   (Gt - self.qvalues.get((tuple(s), a), 0)) / \
                                          self.counter[(tuple(s), a)]

    def get_action_e_greedy(self, s, e):
        # we have action 0, 1, 2, 3
        if e < random.random() : # with prob 1-e, use greedy action
            actions = (0, 1, 2, 3)
            qvalue = -99999999999999999
            action_cand = 1 # right. heuristic
            for action in actions:
                q_tmp = self.qvalues.get((tuple(s), action), 0) # MC control converges regardless of initial value
                # so use 0 as a initial value has no problem
                if q_tmp >= qvalue:
                    action_cand = action
                    qvalue = q_tmp
        else: # with prob e
            action_cand = env.action_space.sample()  ##
        return action_cand

if __name__ == '__main__':
    env = WindyGridWorld()
    state = env.reset()
    mc = MonteCarloControlAgent(env)

    max_episode = 8000

    for i in range(0, max_episode):
        mc.clear()
        state = env.reset()
        done = False
        cnt = 0

        while not done:
            cnt += 1

            # interact with env
            if i > 0.999 * max_episode: # use only one of two render function
                env.render()
                # env.render_qvalues(mc.qvalues)

            action = mc.get_action_e_greedy(state, (max_episode - i) / max_episode)
            next_state, reward, done, info = env.step(action)
            # update queue
            mc.add_info(state, action, reward)
            state = next_state

            if done:
                mc.update_q_with_mc_prediction()

                # print log
                cur_episode = i + 1
                if cur_episode % 100 == 0:
                    print('cur episode : ', cur_episode)
                    print('length of episode : ', len(mc.queue))
