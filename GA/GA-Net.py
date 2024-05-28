import os.path
import random
import tkinter

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
        self.relu = torch.relu
        self.sigmoid = torch.sigmoid

    def forward(self):
        x = self.Input(self.in_features)
        x = self.relu(x)
        x = self.HiddenLayer(x)
        x = self.sigmoid(x)
        output = self.Output(x)
        return output


class SnakeGame(Snake):
    def __init__(self, row=5, column=5, Fps=0, Unit_size=20):
        super(SnakeGame, self).__init__(row, column, Fps, Unit_size)
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
        return torch.tensor(feature)

    # 重写gameloop，不用手动更新了
    def game_loop(self):
        self.win.update()
        self.food(self.snake_list)
        if self.winFlag:
            # self.win.destroy()
            self.win.quit()
            return False
        self.net.in_features = self.returnFeature()
        idx = self.net.forward().contiguous()  # 获取最大概率索引
        cur_dir = [self.snake_list[1][0] - self.snake_list[0][0],
                   self.snake_list[1][1] - self.snake_list[0][1]]  # 当前方向的反方向
        cur_dir_reverse = 1.0 - torch.tensor(self.dir_dict[str(cur_dir)])
        del_idx = torch.argmin(cur_dir_reverse)
        idx = torch.log_softmax(idx, 0).detach().numpy()
        max_num = idx[0]
        max_i = 0
        for i in range(4):
            if idx[i] > max_num and i != del_idx:
                max_num = idx[i]
                max_i = i
        temp_dir = [0.0, 0.0, 0.0, 0.0]
        temp_dir[max_i] = 1.0
        self.Dirc = self.dir_dict[str(temp_dir)]
        self.snake_list = self.move_snake(self.snake_list, self.Dirc, False)
        del idx
        if self.game_over(self.snake_list):
            # self.win.destroy()
            self.win.quit()
            return False
        else:
            self.win.after(self.Fps, self.game_loop)
            return True

    def Restart_game(self, event=None):
        self.canvas.delete(tkinter.ALL)  # 重写了这个和删除了一些不必要东西
        self.winFlag = 0
        self.pause_flag = -1
        self.Dirc = [0, 0]
        self.Score = 0
        self.Energy = int(self.Column * self.Row * 0.25)
        self.Time = 0
        self.snake_list = self.ramdom_snake()
        self.Food_pos = []
        self.Have_food = False
        self.put_a_background(self.canvas, color='white')
        self.draw_the_snake(self.canvas, self.snake_list)
        self.setlable()


class Individual:
    def __init__(self, generation=0, No=0, gene=None):
        self.gene = gene  # gene has a shape like tensor(inpur*hidden1+hidden1*hidden2+hidden2*output)
        self.fitness = 0
        self.generation = generation
        self.No = No

    # 基因为一个tensor，需要将其按一定格式读入Model中
    def gene2State(self):
        I_W = self.gene[:640].contiguous().view(20, 32)
        I_B = self.gene[640:660].contiguous()
        H_W = self.gene[660:900].contiguous().view(12, 20)
        H_B = self.gene[900:912].contiguous()
        O_W = self.gene[912:960].contiguous().view(4, 12)
        O_B = self.gene[960:964].contiguous()
        state_dict = {'Input.weight': I_W, 'Input.bias': I_B, 'HiddenLayer.weight': H_W, 'HiddenLayer.bias': H_B,
                      'Output.weight': O_W, 'Output.bias': O_B}
        return state_dict

    # 计算适应度
    def getFitnessParam(self, score, time, distance, energy):
        self.fitness = (2000.0 * score + 100.0 * time - 0.2 * distance - 10.0 / (energy + 1)) * 10000
        return

    def CalFitness(self, snakegame: SnakeGame):
        snakegame.net.load_state_dict(self.gene2State())
        snakegame.game_loop()
        snakegame.win.mainloop()
        dist = abs(snakegame.snake_list[0][0] - snakegame.Food_pos[0]) + abs(
            snakegame.snake_list[0][1] - snakegame.Food_pos[1])
        self.getFitnessParam(snakegame.Score, snakegame.Time, dist, snakegame.Energy)
        snakegame.Restart_game()


# 遗传算法主方法
def genetic_algorithm(population_size, num_generation, gene_length=None, LastGeneration=None):
    No = 1
    game = SnakeGame(row=10, column=10, Fps=0)
    # 初始化群体或读取上一代最优个体基因
    population = []
    if not LastGeneration:
        while len(population) < population_size:
            snake = Individual(0, No, torch.randn(964))
            No += 1
            snake.CalFitness(game)
            population.append(snake)
    else:
        while len(population) < population_size:
            population.append(LastGeneration)
    # 进化
    for _ in range(num_generation):
        f_mean = 0
        for i in population:
            f_mean += i.fitness
        f_mean /= population_size
        best_individual = max(population, key=lambda x: x.fitness)
        with open(r'./All/' + str(best_individual.generation) + '-' + str(best_individual.No) + '.pkl', 'wb') as ch:
            pickle.dump(best_individual, ch)
        # logging
        with open(r'./evoluiton.txt', 'a') as log:
            log.write(
                f'当前代数: {best_individual.generation}\t最优个体: {best_individual.No}\t适应性: {best_individual.fitness}\n')
        No = 1
        parents = GA.SelectChild(population)
        # next_generation = [Individual(best_individual.generation + 1, No, best_individual.gene.contiguous()),
        #                    Individual(best_individual.generation + 1, No, best_individual.gene.contiguous())]
        next_generation = []
        # 保留上一代部分优秀样本，确保优秀样本能参与交叉变异，模型才能收敛
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            gene1, gene2 = GA.UniformCrossover(parent1, parent2, best_individual.fitness, f_mean)
            child1 = Individual(parent1.generation + 1, No, gene1)
            No += 1
            child2 = Individual(parent2.generation + 1, No, gene2)
            No += 1
            child1 = GA.Variation(child1, num_generation, parents[0].generation)
            child2 = GA.Variation(child2, num_generation, parents[0].generation)
            child1.CalFitness(game)
            child2.CalFitness(game)
            # # 保存样本
            # with open(r'./All/' + str(child1.generation) + '-' + str(child1.No) + '.pkl', 'wb') as ch:
            #     pickle.dump(child1, ch)
            # with open(r'./All/' + str(child2.generation) + '-' + str(child2.No) + '.pkl', 'wb') as ch:
            #     pickle.dump(child2, ch)
            next_generation.extend([child1, child2])
        population = next_generation
    best_individual = max(population, key=lambda x: x.fitness)
    return best_individual


if __name__ == '__main__':
    # 读取最优样本
    last_generation = None
    path = os.listdir(r'./Best')
    if path:
        with open(r'./Best/' + path[-1], 'rb') as f:
            last_generation = pickle.load(f)
    best = genetic_algorithm(100, 50, None, last_generation)
    with open(r'./best/' + str(best.generation) + '-' + str(best.No) + '.pkl', 'w+b') as f:
        pickle.dump(best, f)
    print(f"最好个体:{best.generation}-{best.No} \t 适应度{best.fitness}")
