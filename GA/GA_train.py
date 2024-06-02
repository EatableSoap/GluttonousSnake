import os.path
import random

import torch
from SnakeClass_NoGraph import Snake
import numpy as np
import pickle
from Net import SnakeNet


class SnakeGame(Snake):
    def __init__(self, row=10, column=10, seeds=None):
        super(SnakeGame, self).__init__(row, column, seeds)
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
        self.seeds = seeds

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
        cur_dir = [self.snake_list[1][0] - self.snake_list[0][0],
                   self.snake_list[1][1] - self.snake_list[0][1]]  # 当前方向的反方向
        cur_dir_reverse = 1.0 - torch.tensor(self.dir_dict[str(cur_dir)], dtype=torch.float32)
        del_idx = torch.argmin(cur_dir_reverse).tolist()
        sort_idx = np.argsort(-idx.detach()).tolist()
        max_i = 0
        for i in sort_idx:
            if i != del_idx:
                max_i = i
                break
        temp_dir = [0.0, 0.0, 0.0, 0.0]
        temp_dir[max_i] = 1.0
        _,self.snake_list = self.move_snake(self.snake_list, self.dir_dict[str(temp_dir)])
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
    def __init__(self, gene=None, generation=0, seeds=None):
        self.gene = gene  # gene has a shape like tensor(inpur*hidden1+hidden1*hidden2+hidden2*output)
        self.fitness = 0
        self.score = 0
        self.generation = generation
        self.game = SnakeGame(10, 10, seeds)

    def CalFitness(self):
        self.game.net.setweight(self.gene)
        while self.game.game_loop():
            continue
        # snakegame.game_loop()
        # snakegame.win.mainloop()
        # dist = abs(snakegame.snake_list[0][0] - snakegame.Food_pos[0]) + abs(
        #     snakegame.snake_list[0][1] - snakegame.Food_pos[1])
        # TODO 后续优化目标，加入动态适应度调整
        self.fitness = ((1000 * self.game.Score ** 2 + 5000 / (self.game.Steps + 1)
                         - max(0.0, (self.game.Score - 10) / (self.game.Score + 1) ** 0.85)
                         * self.game.Steps))*self.generation/20
        # self.fitness = (1000 * self.game.Score ** 2 + 5000 / (self.game.Steps + 1)
        #                 - max(0.0, (self.game.Score - 10) / (self.game.Score + 1) ** 0.9) * self.game.Steps)
        # 发现，当分数达到10分左右
        # .时，平均分迎来突变，因此猜测一个分段fitness要更适合GA算法,收敛于均分13左右
        # 加入对步数的惩罚
        # self.fitness = 2500*self.game.Score**2+10000/(self.game.Steps**0.01+1/self.game.Steps**0.01)
        # 由于随Step增大而导致分数减少，因此前期容易陷入尽快撞墙的局部最优
        # self.fitness = (self.game.Score * 100 + 1 / self.game.Steps)
        # 对步数的乘法惩罚不足
        self.score = self.game.Score
        # del self.game
        self.game.Restart_game()


class GA:
    def __init__(self, population_size, select_size, evo_num, mutate_rate,
                 cross_max, cross_min, gene_size=964):
        self.population = []
        self.best: Individual = Individual()
        self.population_size = population_size
        self.select_size = select_size
        self.gene_size = gene_size
        self.evo_num = evo_num
        self.generation = 0
        self.mutate_rate = mutate_rate
        self.crosser_max = cross_max
        self.crosser_min = cross_min
        self.seeds = random.randint(0, 10000)  # 保存种子用于复现成果
        self.use_seeds = True

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

    def Crossover(self, parent1, parent2, f_avg, f_max):
        # 自适应交叉概率
        if max(parent1.fitness, parent2.fitness) < f_avg:
            cross_rate = self.crosser_max
        else:
            cross_rate = (self.crosser_max - (self.crosser_max - self.crosser_min) *
                          (max(parent1.fitness, parent2.fitness) - f_avg) / (f_max - f_avg + 1))
        # cross_rate = self.crosser_rate
        gene1 = parent1.gene.copy()
        gene2 = parent2.gene.copy()
        for i in range(len(gene1)):
            rate = random.uniform(0, 1)
            if rate <= cross_rate:
                gene1[i], gene2[i] = gene2[i], gene1[i]
        return gene1, gene2
        # 单点交叉
        # pos = random.randint(0, len(parent1.gene))
        # gene1[:pos + 1], gene2[:pos + 1] = gene2[:pos + 1], gene1[:pos + 1]
        # return gene1, gene2

    def Variation(self, individual, total_N, cur_n):
        # 自适应变异概率，后期变大
        mutation_rate = self.mutate_rate + 0.1 * (1.0 - random.random() ** (1.0 - cur_n / total_N))
        mutation_array = np.random.random(964) < mutation_rate
        mutation = np.random.standard_cauchy(size=self.gene_size) * max(1.0, 2 * cur_n / total_N)
        individual.gene[mutation_array] += mutation[mutation_array]
        return individual

    # 初始化种群
    def initialize(self):
        while len(self.population) < self.population_size:
            snake = Individual(np.random.uniform(-1, 1, self.gene_size), 0, self.seeds)
            self.population.append(snake)

    # 进化
    def evo(self, lastPopulation=None):
        # 初始化种群或者读取优秀群体
        if not self.generation and not lastPopulation:
            self.initialize()
        else:
            self.population = lastPopulation
        for n in range(self.evo_num):
            for snake in self.population:
                snake.game.enableseed = self.use_seeds
                snake.CalFitness()
            self.generation += 1
            self.best = max(self.population, key=lambda x: x.fitness)
            # 计算平均分
            score_mean = 0
            f_mean = 0
            for inv in self.population:
                f_mean += inv.fitness
                score_mean += inv.score
            score_mean /= len(self.population)
            f_mean /= len(self.population)
            if f_mean >= 5.0:
                self.use_seeds = False
            with open(r'D:/Data/All/evolution.txt', 'a+') as log:
                log.write(f'当前代数：{self.generation - 1}\t 最佳得分：{self.best.score}\t'
                          f'平均分数：{score_mean}\n')
            print(f'当前代数：{self.generation - 1}\t 最佳得分：{self.best.score}\t'
                  f'平均分数：{score_mean}')
            if n != 0 and n % 20 == 0:
                with open(r'D:/Data/All/' + str(self.best.generation)
                          + '_' + str(self.seeds) + '.pkl', 'w+b') as temp_best:
                    pickle.dump(self.best, temp_best)
                with open(r'D:/Data/All/' + str(self.best.generation)
                          + '_' + str(self.seeds) + '_population' + '.pkl', 'w+b') as temp_population:
                    pickle.dump(self.population, temp_population)

            self.SelectChild()
            while len(self.population) < self.population_size:
                parents = self.RoundSelect()
                parent1, parent2 = parents[0:2]
                gene1, gene2 = self.Crossover(parent1, parent2, f_mean, self.best.fitness)
                child1 = Individual(gene1, self.generation, self.seeds)
                child2 = Individual(gene2, self.generation, self.seeds)
                child1 = self.Variation(child1, self.evo_num, self.generation)
                child2 = self.Variation(child2, self.evo_num, self.generation)
                self.population.extend([child1, child2])
            random.shuffle(self.population)
        return self.best


if __name__ == '__main__':
    random.seed(100)
    populations = 400
    generations = 2000
    ele_individual = 100
    mutation_scale = 0.1
    crossMax = 0.8
    crossMin = 0.4
    # 读取最优样本(其实是最新样本)
    last_population = None
    path = os.listdir(r'D:/Data/Best')
    path = sorted(path, key=lambda x: int(x.split('_')[0]), reverse=True)

    GA = GA(populations, ele_individual, generations, mutation_scale, crossMax, crossMin)
    if path:
        for i in path:
            if i.endswith('_population.pkl'):
                last_population = i
                break
        with open(r'D:/Data/Best/' + last_population, 'rb') as f:
            last_population = pickle.load(f)
        # last_seeds = path[0].split('_')[1].split('.')[0]
        # GA.seeds = last_seeds
        GA.generation = last_population[0].generation
    GA.evo(last_population)
    BestIndividual = GA.best
    with open(r'D:/Data/Best/' + str(BestIndividual.generation) + '_'
              + str(GA.seeds) + '.pkl', 'w+b') as f:
        pickle.dump(BestIndividual, f)
    with open(r'D:/Data/Best/' + str(BestIndividual.generation) + '_'
              + str(GA.seeds) + '_population' + '.pkl', 'w+b') as f:
        pickle.dump(GA.population, f)
    print(f"最好个体:{BestIndividual.generation}\t 得分{BestIndividual.score}")
# TODO list
'''
目前来说，程序基本没问题，但仍存在可优化地方：
适应度函数增加分段优化，给予更大区分度
加入自适应变异概率，增加种群多样性
还需要跑更多代数
'''