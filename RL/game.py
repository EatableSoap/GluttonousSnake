from GluttonousSnake.GA.SnakeClass_NoGraph import Snake
# from GluttonousSnake.Snake.SnakeClass import Snake
import torch
import numpy as np


# delete Net()
class SnakeGame(Snake):
    def __init__(self, row=10, column=10, seeds=None, Fps=100):
        super(SnakeGame, self).__init__(row, column, seeds=seeds)
        self.dir_dict = {
            '[0.0, -1.0]': [1.0, 0.0, 0.0, 0.0],
            '[0.0, 1.0]': [0.0, 1.0, 0.0, 0.0],
            '[-1.0, 0.0]': [0.0, 0.0, 1.0, 0.0],
            '[1.0, 0.0]': [0.0, 0.0, 0.0, 1.0],
            '[0, -1]': [1.0, 0.0, 0.0, 0.0],
            '[0, 1]': [0.0, 1.0, 0.0, 0.0],
            '[-1, 0]': [0.0, 0.0, 1.0, 0.0],
            '[1, 0]': [0.0, 0.0, 0.0, 1.0],
            '[1.0, 0.0, 0.0, 0.0]': [0, -1],  # up
            '[0.0, 1.0, 0.0, 0.0]': [0, 1],  # down
            '[0.0, 0.0, 1.0, 0.0]': [-1, 0],  # left
            '[0.0, 0.0, 0.0, 1.0]': [1, 0]  # right
        }
        self.over = False
        self.seeds = seeds

    def returnFeature(self):
        feature = []
        head_dir = self.dir_dict[str((np.array(self.snake_list[0], dtype=float) -
                                      np.array(self.snake_list[1], dtype=float)).tolist())]
        tail_dir = self.dir_dict[str((np.array(self.snake_list[-2], dtype=float) -
                                      np.array(self.snake_list[-1], dtype=float)).tolist())]
        feature += head_dir + tail_dir

        # eight direction to get features
        dirs = [[0, -1], [1, -1], [1, 0], [1, 1],
                [0, 1], [-1, 1], [-1, 0], [-1, -1]]
        for direc in dirs:
            x = self.snake_list[0][0] + direc[0]
            y = self.snake_list[0][1] + direc[1]
            dis = 1.0
            see_food = 0.0
            see_self = 0.0
            while 0 <= x < self.Column and 0 <= y < self.Row:
                if [x, y] == self.Food_pos:
                    see_food = 1.0
                elif [x, y] in self.snake_list:
                    see_self = 1.0
                dis += 1
                x += direc[0]
                y += direc[1]
            feature += [1.0 / dis, see_food, see_self]
        return np.array(feature, dtype=float)

    def move_snake(self, snke_list, direc, rush):
        isEat = False
        if direc != [0, 0]:
            head_0 = snke_list[0]
            temp_head = head_0.copy()
            temp_head[0] = temp_head[0] + direc[0]
            temp_head[1] = temp_head[1] + direc[1]
            snke_list.insert(0, temp_head)
            if not self.snake_eat(snke_list, self.Food_pos):
                self.Energy -= 1
                snke_list.pop(-1)
            else:
                isEat = True
                self.Score += 1
                self.Energy = min(int(self.Column * self.Row), self.Energy + int(self.Column * self.Row * 0.6))
            if rush:
                self.Steps += 1
            else:
                self.Steps += 2
            del temp_head
        return isEat, snke_list


class SnakeGameAI(SnakeGame):
    def __init__(self, row=20, column=20, Fps=50, seeds=None):
        super(SnakeGameAI, self).__init__(row, column, Fps=Fps, seeds=seeds)

    def CalDistance(self):
        return (abs(self.snake_list[0][0] - self.Food_pos[0])
                + abs(self.snake_list[0][1] - self.Food_pos[1]))

    def play_step(self, action, base_reward=10, reward_scale=1):
        # 1. move
        self.game_loop()
        previous_dis = self.CalDistance()
        isEat, self.snake_list = self.move_snake(self.snake_list, action, False)
        current_dist = self.CalDistance()

        # 2.check game is over
        is_done = False
        reward = 0
        if self.game_over(self.snake_list):
            is_done = True
            reward += -base_reward / reward_scale
            return reward, is_done, self.Score

        # 3. food is eaten
        if isEat:
            self.food(self.snake_list)
            self.Score += 1
            reward += base_reward * reward_scale
        else:
            reward += -1 / reward_scale

        reward += (previous_dis-current_dist)*5
        # 加入对距离的奖励机制可以大大提高平均分数

        # 4. return info
        return reward, is_done, self.Score
