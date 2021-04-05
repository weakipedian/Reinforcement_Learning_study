import random

# host dependent relative path
import sys
sys.path.append('../../reinforcement-learning-kr/1-grid-world/1-policy-iteration')
from environment import GraphicDisplay, Env

import os
os.chdir('../../reinforcement-learning-kr/1-grid-world/1-policy-iteration')

class PolicyIteration:
    def __init__(self, env):
        self.env = env
        self.value_table = [[.0] * env.width for _ in range(env.height)]
        self.policy_table = [[[.25, .25, .25, .25]] * env.width for _ in range(env.height)]

        self.policy_table[2][2] = []
        self.discount_factor = 0.9

    #벨만 기대 방정식을 통해 다음 가치함수를 계산하는 정책 평가
    def policy_evaluation(self):
        next_value_table = [[.00] * self.env.width for _ in range(self.env.height)]

        #모든 상태에 대해서 벨만 기대 방정식을 계산
        for state in self.env.get_all_states():
            value = .0
            if state == [2, 2]: # end state?
                next_value_table[state[0]][state[1]] = .0
                continue

            #벨만 기대 방정식
            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value += self.get_policy(state)[action] * \
                    (reward + self.discount_factor * next_value)
            next_value_table[state[0]][state[1]] = round(value, 2)

        self.value_table = next_value_table

    # 현재 가치함수에 대해 탐욕 정책 발전
    def policy_improvement(self):
        next_policy = self.policy_table
        for state in self.env.get_all_states():
            if state == [2, 2]:
                continue
            value = -99999
            max_index = []
            result_action_prob = [.0, .0, .0, .0]

            for index, action in enumerate(self.env.possible_actions):
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                temp = reward + self.discount_factor * next_value

                # 받을 보상이 최대인 행동의 인덱스를 추출(여러개 리턴 가능)
                if temp == value:
                    max_index.append(index)
                elif temp > value:
                    value = temp
                    max_index.clear()
                    max_index.append(index)

            # 행동의 확률 계산
            prob = 1 / len(max_index)
            for index in max_index:
                result_action_prob[index] = prob

            next_policy[state[0]][state[1]] = result_action_prob

        self.policy_table = next_policy

    def get_action(self, state):
        random_pick = random.randrange(100) / 100
        policy = self.get_policy(state)
        policy_sum = 0.0

        for index, action_prob in enumerate(policy):
            policy_sum += action_prob
            if random_pick < policy_sum:
                return index

    def get_policy(self, state):
        if state == [2, 2]:
            return 0.0
        return self.policy_table[state[0]][state[1]]

    def get_value(self, state):
        return round(self.value_table[state[0]][state[1]], 2)

if __name__ == '__main__':
    env = Env()
    policy_iteration = PolicyIteration(env)
    grid_world = GraphicDisplay(policy_iteration)
    grid_world.mainloop()