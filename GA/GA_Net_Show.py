from GA_Net_train import Individual,SnakeNet
import os.path
import tkinter
import torch
from GluttonousSnake.Snake.SnakeClass import Snake
import numpy as np
import pickle


class SnakeGame(Snake):
    def __init__(self, row=5, column=5,Fps=100):
        super(SnakeGame, self).__init__(row, column,Fps)
        # 方向字典，无需写函数
        self.dir_dict = {
            '[0.0, -1.0]': [1.0, 0.0, 0.0, 0.0],
            '[0.0, 1.0]': [0.0, 1.0, 0.0, 0.0],
            '[-1.0, 0.0]': [0.0, 0.0, 1.0, 0.0],
            '[1.0, 0.0]': [0.0, 0.0, 0.0, 1.0],
            '[0, -1]': [1.0, 0.0, 0.0, 0.0],
            '[0, 1]': [0.0, 1.0, 0.0, 0.0],
            '[-1, 0]': [0.0, 0.0, 1.0, 0.0],
            '[1, 0]': [0.0, 0.0, 0.0, 1.0],
            '[1.0, 0.0, 0.0, 0.0]': [0.0, -1.0],
            '[0.0, 1.0, 0.0, 0.0]': [0.0, 1.0],
            '[0.0, 0.0, 1.0, 0.0]': [-1.0, 0.0],
            '[0.0, 0.0, 0.0, 1.0]': [1.0, 0.0]
        }
        self.net = SnakeNet()

    def returnFeature(self):
        feature = []
        head_dir = self.dir_dict[
            str((np.array(self.snake_list[0], dtype=float) - np.array(self.snake_list[1], dtype=float)).tolist())]
        tail_dir = self.dir_dict[
            str((np.array(self.snake_list[-2], dtype=float) - np.array(self.snake_list[-1], dtype=float)).tolist())]
        feature += head_dir + tail_dir

        # 遍历各个方向，获取数据
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
            feature += [1.0 / dis, see_food, see_self]  # 八个视角分别为距离边缘距离倒数，视野内是否有食物、有自身
        return torch.tensor(feature, dtype=torch.float32)

    # 重写gameloop，不用手动更新了
    def game_loop(self):
        self.food(self.snake_list)
        if self.winFlag:
            self.Restart_game()
            return False
        in_features = self.returnFeature()
        idx = self.net.forward(in_features).contiguous()  # 获取最大概率索引
        cur_dir = [self.snake_list[1][0] - self.snake_list[0][0],
                   self.snake_list[1][1] - self.snake_list[0][1]]  # 当前方向的反方向
        cur_dir_reverse = 1.0 - torch.tensor(self.dir_dict[str(cur_dir)], dtype=torch.float32)
        del_idx = torch.argmin(cur_dir_reverse).tolist()
        idx = torch.log_softmax(idx, 0).detach().numpy()
        sort_idx = np.argsort(-idx).tolist()
        max_i = 0
        for i in sort_idx:
            if i != del_idx:
                max_i = i
                break
        temp_dir = [0.0, 0.0, 0.0, 0.0]
        temp_dir[max_i] = 1.0
        self.snake_list = self.move_snake(self.snake_list, self.dir_dict[str(temp_dir)], False)
        del idx
        if self.game_over(self.snake_list):
            self.Restart_game()
            return False
        else:
            self.win.after(self.Fps,self.game_loop)
            return True

    def Restart_game(self, event=None):
        self.winFlag = 0
        self.pause_flag = -1
        self.Dirc = [0, 0]
        self.Score = 0
        self.Energy = int(self.Column * self.Row * 0.4)
        self.Time = 0
        self.snake_list = self.ramdom_snake()
        self.Food_pos = []
        self.Have_food = False
        # 这里取消注释外加import NOGraph下另外一个SnakeClass可以看见具体在训练什么
        self.canvas.delete(tkinter.ALL)  # 重写了这个和删除了一些不必要东西
        self.put_a_background(self.canvas, color='white')
        self.draw_the_snake(self.canvas, self.snake_list)
        self.setlable()
        self.game_loop()
        self.win.mainloop()


if __name__ == '__main__':
    # 读取最优样本
    last_generation = None
    path = os.listdir(r'./Best')
    path = sorted(path, key=lambda x: int(x.split('-')[0]), reverse=True)
    if path:
        with open(r'./Best/' + path[0], 'rb') as f:
            last_generation = pickle.load(f)
    game = SnakeGame(10,10,50)
    game.net.load_state_dict(Individual.gene2State(last_generation))
    game.game_loop()
    game.win.mainloop()
