import random
import torch
from GluttonousSnake.Snake.SnakeClass import Snake
import GA
import numpy as np
import pickle


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
    def __init__(self, row=10, column=10, Fps=10, Unit_size=20):
        super(SnakeGame, self).__init__(row, column, Fps, Unit_size)
        self.dir_dict = {
            '[0, -1]': [1.0, 0.0, 0.0, 0.0],
            '[0, 1]': [0.0, 1.0, 0.0, 0.0],
            '[-1, 0]': [0.0, 0.0, 1.0, 0.0],
            '[1, 0]': [0.0, 0.0, 0.0, 1.0],
            '[1, 0, 0, 0]': [0.0, -1.0],
            '[0, 1, 0, 0]': [0.0, 1.0],
            '[0, 0, 1, 0]': [-1.0, 0.0],
            '[0, 0, 0, 1]': [1.0, 0.0]
        }
        self.net = SnakeNet()

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
            # self.win.destroy()
            return False
        # TODO 记得修改一个能用的
        self.net.in_features = self.returnFeature()
        idx = self.net.forward().numpy()[0]  # 获取最大概率索引
        temp_dir = [0, 0, 0, 0]
        temp_dir[idx] = 1
        self.Dirc = self.dir_dict[str(temp_dir)]
        self.snake_list = self.move_snake(self.snake_list, self.Dirc, False)
        if self.game_over(self.snake_list):
            # self.win.destroy()
            return False
        else:
            self.win.after(self.Fps, self.game_loop)
            return True


class Individual:
    def __init__(self, generation=0, No=0, gene=None):
        # self.genes = SnakeNet()
        self.gene = gene  # gene has a shape like tensor(inpur*hidden1+hidden1*hidden2+hidden2*output)
        # if state_dict is not None:
        #     self.genes.load_state_dict(state_dict)
        self.fitness = 0
        self.generation = generation
        self.No = No

    def gene2State(self):
        I_W = self.gene[:640].contiguous().view(20, 32)
        I_B = self.gene[640:660].contiguous()
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

    def CalFitness(self, snakegame: SnakeGame):
        snakegame.net.load_state_dict(self.gene2State())
        snakegame.game_loop()
        snakegame.win.mainloop()
        self.getFitnessParam(snakegame.Score, snakegame.Time)
        snakegame.Restart_game()


def genetic_algorithm(population_size, num_generation, gene_length=None, LastGeneration=None):
    No = 1
    game = SnakeGame()
    # 初始化群体
    population = []
    if not LastGeneration:
        while len(population) < population_size:
            snake = Individual(1, No, torch.randn(964))
            No += 1
            game.net.load_state_dict(snake.gene2State())
            game.game_loop()
            game.win.mainloop()
            snake.getFitnessParam(game.Score, game.Time)
            game.Restart_game()
            population.append(snake)
    else:
        while len(population) < population_size:
            population.append(LastGeneration)
    # 进化
    for _ in range(num_generation):
        No = 1
        parents = GA.SelectChild(population)
        next_generation = []
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            gene1, gene2 = GA.UniformCrossover(parent1, parent2)
            child1 = Individual(parent1.generation + 1, No, gene1)
            No += 1
            child2 = Individual(parent2.generation + 1, No, gene2)
            No += 1
            child1 = GA.Variation(child1)
            child2 = GA.Variation(child2)
            child1.CalFitness(game)
            child2.CalFitness(game)
            with open(r'./' + str(child1.generation) + '/' + str(child1.No) + '.pkl', 'wb') as f:
                pickle.dump(child1, f)
            with open(r'./' + str(child2.generation) + '/' + str(child2.No) + '.pkl', 'wb') as f:
                pickle.dump(child2, f)
            next_generation.extend([child1, child2])
        population = next_generation
    best_individual = max(population, key=lambda x: x.fitness)
    print(f"最优适应度: {best_individual.fitness}")
    return best_individual


if __name__ == '__main__':
    best = genetic_algorithm(100, 20)
    with open(r'./best/' + str(best.No) + '.pkl', 'wb') as f:
        pickle.dump(best, f)
    print(f"最好个体: Generation {best.generation}_{best.No} \t 适应度{best.fitness}")
    # TODO 载入上一代功能
