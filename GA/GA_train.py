import os.path
import random

import torch
from SnakeClass_NoGraph import Snake
import numpy as np
import pickle
from Net import SnakeNet


class SnakeGame(Snake):
    def __init__(self, row=10, column=10):
        super(SnakeGame, self).__init__(row, column)
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
        self.over = False

    def returnFeature(self):
        feature = []
        head_dir = self.dir_dict[str((np.array(self.snake_list[0], dtype=float) -
                                      np.array(self.snake_list[1], dtype=float)).tolist())]
        tail_dir = self.dir_dict[str((np.array(self.snake_list[-2], dtype=float) -
                                      np.array(self.snake_list[-1], dtype=float)).tolist())]
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
        return torch.tensor(feature).float()

    # 重写gameloop，不用手动更新了
    def game_loop(self):
        self.food(self.snake_list)
        if self.winFlag:
            # self.win.quit()
            return False
        in_features = self.returnFeature()
        idx = self.net.predic(in_features)
        temp_dir = [0.0, 0.0, 0.0, 0.0]
        temp_dir[idx] = 1.0
        self.snake_list = self.move_snake(self.snake_list, self.dir_dict[str(temp_dir)])
        if self.game_over(self.snake_list):
            # self.win.quit()
            self.over = True
            return False
        else:
            # self.win.after(self.Fps, self.game_loop)
            return True

    def Restart_game(self, event=None):
        self.winFlag = 0
        self.Dirc = [0, 0]
        self.Score = 0
        self.Energy = int(self.Column * self.Row * 0.4)
        self.Steps = 0
        self.snake_list = self.ramdom_snake()
        self.Food_pos = []
        self.Have_food = False
        self.over = False

        # 这里取消注释外加import NOGraph下另外一个SnakeClass可以看见具体在训练什么
        # self.canvas.delete(tkinter.ALL)  # 重写了这个和删除了一些不必要东西
        # self.put_a_background(self.canvas, color='white')
        # self.draw_the_snake(self.canvas, self.snake_list)
        # self.setlable()


class Individual:
    def __init__(self, gene=None, generation=0):
        self.gene = gene  # gene has a shape like tensor(inpur*hidden1+hidden1*hidden2+hidden2*output)
        self.fitness = 0
        self.score = 0
        self.generation = generation
        self.game = SnakeGame(10, 10)

    def CalFitness(self):
        self.game.net.setweight(self.gene)
        while self.game.game_loop():
            continue
        # snakegame.game_loop()
        # snakegame.win.mainloop()
        # dist = abs(snakegame.snake_list[0][0] - snakegame.Food_pos[0]) + abs(
        #     snakegame.snake_list[0][1] - snakegame.Food_pos[1])
        # self.fitness = (2500 * self.game.Score ** 2 + 10000 / (1 + 1 / self.game.Steps ** 0.05))
        self.fitness = (self.game.Score * 100 + 1 / self.game.Steps)
        self.score = self.game.Score
        # del self.game
        self.game.Restart_game()


class GA:
    def __init__(self, population_size, select_size, evo_num, mutate_rate,
                 cross_rate, gene_size=964):
        self.population = []
        self.best: Individual = Individual()
        self.population_size = population_size
        self.select_size = select_size
        self.gene_size = gene_size
        self.evo_num = evo_num
        self.generation = 0
        self.mutate_rate = mutate_rate
        self.crosser_rate = cross_rate

    def SelectChild(self):
        sorted_child = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        for i in sorted_child:
            i.generation += 1
        self.population = sorted_child[:self.select_size]

    def RoundSelect(self):
        rate_all = []
        for i in range(len(self.population)):
            rate = 2 - 1.1 + 2 * (1.1 - 1.0) * i / (len(self.population) - 1)
            rate_all.append(rate)
        select = []
        for i in range(2):
            idx = random.uniform(0, sum(rate_all))
            ind = 0
            cur = 0
            while cur < idx:
                cur += rate_all[ind]
                ind += 1
            select.append(self.population[ind - 1])
        return select

    def Crossover(self, parent1, parent2):
        # # 自适应交叉概率
        # if max(parent1.fitness, parent2.fitness) < f_avg:
        #     cross_rate = P_max
        # else:
        #     cross_rate = P_max - (P_max - P_min) *
        #     (max(parent1.fitness, parent2.fitness) - f_avg) / (f_max - f_avg + 1)
        # cross_rate = self.crosser_rate
        gene1 = parent1.gene.copy()
        gene2 = parent2.gene.copy()
        # for i in range(len(gene1)):
        #     rate = random.uniform(0, 1)
        #     if rate <= cross_rate:
        #         gene1[i], gene2[i] = gene2[i], gene1[i]
        # return gene1, gene2
        # 单点交叉
        pos = random.randint(0, len(parent1.gene))
        gene1[:pos + 1], gene2[:pos + 1] = gene2[:pos + 1], gene1[:pos + 1]
        return gene1, gene2

    def Variation(self, individual, total_N, cur_n):
        # 自适应变异概率
        mutation_rate = self.mutate_rate  # * (1.0 - random.random() ** (1.0 - cur_n / total_N))
        mutation_array = np.random.random(964) < mutation_rate
        mutation = np.random.standard_cauchy(size=964)  # * cur_n / total_N
        # + np.random.normal(size=964) * (1.0 - cur_n / total_N)
        # mutation *= 0.2
        individual.gene[mutation_array] += mutation[mutation_array]
        # 避免数据过大
        for i in range(len(individual.gene)):
            if individual.gene[i] > 1:
                individual.gene[i] /= 10
            if individual.gene[i] < -1:
                individual.gene[i] /= 10
        return individual

    # 初始化种群
    def initialize(self):
        while len(self.population) < self.population_size:
            snake = Individual(np.random.uniform(-0.1, 0.1, self.gene_size), 0)
            self.population.append(snake)

    # 进化
    def evo(self):
        if not self.generation:
            self.initialize()
        for n in range(self.evo_num):
            for snake in self.population:
                snake.CalFitness()
            self.generation += 1
            self.best = max(self.population, key=lambda x: x.fitness)
            # 计算平均分
            score_mean = 0
            for inv in self.population:
                score_mean += inv.score
            score_mean /= len(self.population)

            # os.chmod(r'D:/Data/All/')
            with open(r'D:/Data/All/evolution.txt', 'a+') as log:
                log.write(f'当前代数：{self.generation - 1}\t 最佳得分：{self.best.score}\t'
                          f'平均分数：{score_mean}\n')
            print(f'当前代数：{self.generation - 1}\t 最佳得分：{self.best.score}\t'
                  f'平均分数：{score_mean}')
            if n % 20 == 0:
                with open(r'D:/Data/All/' + str(self.best.generation)
                          + '.pkl', 'w+b') as f:
                    pickle.dump(self.best, f)

            sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
            self.best = sorted_population[0]

            self.SelectChild()
            while len(self.population) < self.population_size:
                parents = self.RoundSelect()
                parent1, parent2 = parents[0:2]
                gene1, gene2 = self.Crossover(parent1, parent2)
                child1 = Individual(gene1, self.generation)
                child2 = Individual(gene2, self.generation)
                child1 = self.Variation(child1, self.evo_num, self.generation)
                child2 = self.Variation(child2, self.evo_num, self.generation)
                self.population.extend([child1, child2])
            random.shuffle(self.population)
        return self.best


if __name__ == '__main__':
    random.seed(100)
    populations = 1000
    generations = 1000
    ele_individual = 100
    mutation_scale = 0.1
    crossMax = 1.5
    crossMin = 0.4
    # 读取最优样本
    last_generation = None
    path = os.listdir(r'D:/Data/Best')
    path = sorted(path, key=lambda x: int(x.split('.')[0]), reverse=True)
    if path:
        with open(r'D:/Data/Best/' + path[0], 'rb') as f:
            last_generation = pickle.load(f)
    GA = GA(populations, ele_individual, generations, mutation_scale, crossMin).evo()
    BestIndividual = GA.best
    with open(r'D:/Data/Best/' + str(BestIndividual.generation) + '.pkl',
              'w+b') as f:
        pickle.dump(BestIndividual, f)
    print(f"最好个体:{BestIndividual.generation}\t 得分{BestIndividual.score}")
