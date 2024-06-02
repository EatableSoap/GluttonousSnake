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
                # self.draw_a_unit(self.canvas, snke_list[-1][0], snke_list[-1][1], unit_color="white")
                snke_list.pop(-1)
                # self.draw_a_unit(self.canvas, snke_list[1][0], snke_list[1][1])
                # self.draw_a_unit(self.canvas, temp_head[0], temp_head[1], unit_color="purple")
                # self.draw_a_unit(self.canvas, snke_list[-1][0], snke_list[-1][1], unit_color='orange')
            else:
                isEat = True
                self.Score += 1
                self.Energy = min(int(self.Column * self.Row), self.Energy + int(self.Column * self.Row * 0.6))
            if rush:
                self.Steps += 1
            else:
                self.Steps += 2
            # self.draw_a_unit(self.canvas, snke_list[-1][0], snke_list[-1][1], unit_color="orange")
            # self.draw_a_unit(self.canvas, snke_list[1][0], snke_list[1][1])
            # self.draw_a_unit(self.canvas, temp_head[0], temp_head[1], unit_color="purple")
            # self.setlable()
            # self.win.update()
            del temp_head
        return isEat, snke_list


class SnakeGameAI(SnakeGame):
    def __init__(self, row=20, column=20, Fps=50):
        super(SnakeGameAI, self).__init__(row, column, Fps=Fps, seeds=None)

    def play_step(self, action):
        # 1. move
        self.game_loop()
        isEat, self.snake_list = self.move_snake(self.snake_list, action, False)

        # 2.check game is over
        is_done = False
        reward = 0
        if self.game_over(self.snake_list):
            is_done = True
            reward -= 10
            return reward, is_done, self.Score

        # 3. food is eaten
        if isEat:
            self.food(self.snake_list)
            self.Score += 1
            reward = 10

        # 4. return info
        return reward, is_done, self.Score
