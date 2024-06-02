# coding=utf-8
import tkinter as tk

from GluttonousSnake.Snake.SnakeClass import Snake
import torch
import numpy as np
from agent import Agent


class SnakeGame(Snake):
    def __init__(self, row=10, column=10, seeds=None):
        super(SnakeGame, self).__init__(row, column, seeds)
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
                self.str_energy.set('Energy:' + str(self.Energy))
                self.draw_a_unit(self.canvas, snke_list[-1][0], snke_list[-1][1], unit_color="white")
                snke_list.pop(-1)
                self.draw_a_unit(self.canvas, snke_list[1][0], snke_list[1][1])
                self.draw_a_unit(self.canvas, temp_head[0], temp_head[1], unit_color="purple")
                self.draw_a_unit(self.canvas, snke_list[-1][0], snke_list[-1][1], unit_color='orange')
            else:
                isEat = True
                self.Score += 1
                self.Energy = min(int(self.Column * self.Row), self.Energy + int(self.Column * self.Row * 0.6))
                self.str_score.set('Score:' + str(self.Score))
                self.str_energy.set('Energy:' + str(self.Energy))
                self.draw_a_unit(self.canvas, snke_list[-1][0], snke_list[-1][1], unit_color="orange")
                self.draw_a_unit(self.canvas, snke_list[1][0], snke_list[1][1])
                self.draw_a_unit(self.canvas, temp_head[0], temp_head[1], unit_color="purple")
            if rush:
                self.Steps += 1
            else:
                self.Steps += 2
            self.str_time.set('Time:' + str(self.Steps))
            self.win.update()
            del temp_head
        return isEat, snke_list


class SnakeGameAI(SnakeGame):
    def __init__(self, row=20, column=20, Fps=50):
        super(SnakeGameAI, self).__init__(row, column, Fps)

    def play_step(self, action):
        # 1. move
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

    def game_loop(self):
        self.win.update()
        self.food(self.snake_list)
        if self.winFlag:
            self.over_label = tk.Label(self.win, text='You Win!', font=('楷体', 25), width=15, height=1)
            self.over_label.place(x=(self.Width - 260) / 2, y=(self.Height - 40) / 2, bg=None)
            self.win.update()
            self.win.quit()
            return True
        _, self.snake_list = self.move_snake(self.snake_list, self.Dirc, False)
        if self.game_over(self.snake_list):
            self.over_label = tk.Label(self.win, text='Game Over', font=('楷体', 25), width=15, height=1)
            self.over_label.place(x=(self.Width - 260) / 2, y=(self.Height - 40) / 2, bg=None)
            self.win.update()
            self.win.quit()
            return True
        else:
            return False


if __name__ == '__main__':
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    # game = SnakeGame(20,20)
    game = SnakeGameAI()
    agent.model.load_model(r'model.pth')
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        cur_dir = [game.snake_list[1][0] - game.snake_list[0][0],
                   game.snake_list[1][1] - game.snake_list[0][1]]  # the oppsite dirc of the current dirc
        cur_dir_reverse = 1.0 - torch.tensor(game.dir_dict[str(cur_dir)], dtype=torch.float32)
        del_idx = torch.argmin(cur_dir_reverse).tolist()
        sort_idx = np.argsort(-final_move.detach()).tolist()
        max_i = None
        for i in sort_idx:
            if i != del_idx:
                max_i = i
                break
        if max_i is None:
            max_i = (sort_idx[0] + 1) % 4
        temp_dir = [0.0, 0.0, 0.0, 0.0]
        temp_dir[max_i] = 1.0
        # reward, is_done, score = game.play_step(game.dir_dict[str(temp_dir)])
        game.Dirc = game.dir_dict[str(temp_dir)]
        is_done = game.game_loop()
        game.win.after(game.Fps)

        if is_done:
            agent.n_games += 1
            game.Restart_game()
            print('Game', agent.n_games, 'Score', game.Score, 'Record:', record)
