# host dependent relative path
import sys
sys.path.append('../../reinforcement-learning-kr/1-grid-world/2-value-iteration')
from environment import GraphicDisplay, Env

import os
os.chdir('../../reinforcement-learning-kr/1-grid-world/2-value-iteration')

class ValueIteration:
    def __init__(self, env):
        # 환경 객체 생성
        self.env = env
        # 가치함수를 2차원 리스트로 초기화
        self.value_table = [[.00] * env.width for _ in range(env.height)]
        self.discount_factor = 0.9

    def value_iteration(self):
        next_value_table = [[0.00] * self.env.width for _ in range(self.env.height)]

        for state in self.env.get_all_states():
            if state == [2, 2]:
                next_value_table[state[0]][state[1]] = 0.0
                continue
            value_list = []

            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value_list.append((
                    reward + self.discount_factor * next_value
                ))

            # 최댓값을 다음 가치함수로 대입
            next_value_table[state[0]][state[1]] = round(max(value_list), 2)
        self.value_table = next_value_table

    def get_action(self, state):
        action_list = []
        max_value = -99999

        if state == [2, 2]:
            return []

        for action in self.env.possible_actions:
            next_state = self.env.state_after_action(state, action)
            reward = self.env.get_reward(state, action)
            next_value = self.get_value(next_state)
            value = (reward + self.discount_factor * next_value)

            if value > max_value:
                action_list.clear()
                action_list.append(action)
                max_value = value
            elif value == max_value:
                action_list.append(action)

        return action_list

    def get_value(self, state):
        return round(self.value_table[state[0]][state[1]], 2)

if __name__ == '__main__':
    env = Env()
    value_iteration = ValueIteration(env)
    grid_world = GraphicDisplay(value_iteration)
    grid_world.mainloop()
