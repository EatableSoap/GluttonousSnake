import torch
from GluttonousSnake.Snake.SnakeClass import Snake
import GA
import numpy as np


class SnakeNet(torch.nn.Module):
    def __init__(self, in_features=None):
        super(SnakeNet, self).__init__()
        self.in_features = in_features
        self.Input = torch.nn.Linear(32, 20)
        self.HiddenLayer = torch.nn.Linear(20, 12)
        self.Output = torch.nn.Linear(12, 4)
        self.relu = torch.nn.ReLU
        self.sigmoid = torch.nn.Sigmoid

    def forward(self):
        x = self.Input(self.in_features)
        x = self.relu(x)
        x = self.HiddenLayer(x)
        x = self.sigmoid(x)
        output = self.Output(x)
        return torch.argmax(output)


class SnakeGame(Snake):
    def __init__(self, row=20, column=20, Fps=100, Unit_size=40):
        super(SnakeGame, self).__init__(row, column, Fps, Unit_size)
        self.dir_dict = {
            '[0, -1]': [1.0, 0.0, 0.0, 0.0],
            '[0, 1]': [0.0, 1.0, 0.0, 0.0],
            '[-1, 0]': [0.0, 0.0, 1.0, 0.0],
            '[1, 0]': [0.0, 0.0, 0.0, 1.0],
            '[1, 0, 0, 0]': [0.0,-1.0],
            '[0, 1, 0, 0]': [0.0,1.0],
            '[0, 0, 1, 0]': [-1.0,0.0],
            '[0, 0, 0, 1]': [1.0,0.0]
        }

    def returnFeature(self):
        feature = []
        head_dir = self.dir_dict[str((np.array(self.snake_list[0]) - np.array(self.snake_list[1])).tolist())]
        tail_dir = self.dir_dict[str((np.array(self.snake_list[-2]) - np.array(self.snake_list[-1])).tolist())]
        feature += head_dir + tail_dir

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
        return torch.tensor(feature)

    # 重写gameloop，不用手动更新了
    def game_loop(self):
        self.win.update()
        self.food(self.snake_list)
        if self.winFlag:
            self.win.destroy()
        self.snake_list = self.move_snake(self.snake_list, self.Dirc, False)
        if self.game_over(self.snake_list):
            self.win.destroy()
            return False
        else:
            self.win.after(self.Fps, self.game_loop)
            return True


class Individual:
    def __init__(self, gene=None):
        # self.genes = SnakeNet()
        self.gene = gene  # gene has a shape like tensor(inpur*hidden1+hidden1*hidden2+hidden2*output)
        # if state_dict is not None:
        #     self.genes.load_state_dict(state_dict)
        self.fitness = 0

    def gene2State(self):
        I_W = self.gene[:640].contiguous().view(20, 32)
        I_B = self.gene[640:660].contigugous()
        H_W = self.gene[660:900].contiguous().view(12, 20)
        H_B = self.gene[900:912].contiguous()
        O_W = self.gene[912:960].contiguous().view(12, 4)
        O_B = self.gene[960:964].contiguous()
        state_dict = {'Input.weight': I_W, 'Input.bias': I_B, 'HiddenLayer.weight': H_W, 'HiddenLayer.bias': H_B,
                      'Output.weight': O_W, 'Output.bias': O_B}
        return state_dict

    def getFitnessParam(self, score, time):
        self.fitness = GA.CalFitness(score, time)
        return


def genetic_alo():
    return