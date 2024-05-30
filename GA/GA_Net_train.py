import os.path
import random

import torch
from SnakeClass_NoGraph import Snake
# from GluttonousSnake.Snake.SnakeClass import Snake
import numpy as np
import pickle
import multiprocessing as mp


class SnakeNet(torch.nn.Module):
    def __init__(self, in_feature=32, hidden1_featute=20, hidden2_feature=12, out_feature=4):
        super(SnakeNet, self).__init__()
        self.Input = torch.nn.Linear(in_feature, hidden1_featute)
        self.HiddenLayer = torch.nn.Linear(hidden1_featute, hidden2_feature)
        self.Output = torch.nn.Linear(hidden2_feature, out_feature)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        y = self.Input(x)
        y = self.sigmoid(y)
        y = self.HiddenLayer(y)
        y = self.sigmoid(y)
        y = self.Output(y)
        y = self.sigmoid(y)
        return y


class SnakeGame(Snake):
    def __init__(self, row=5, column=5, Fps=100):
        super(SnakeGame, self).__init__(row, column, Fps)
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
            return False
        else:
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
        # self.canvas.delete(tkinter.ALL)  # 重写了这个和删除了一些不必要东西
        # self.put_a_background(self.canvas, color='white')
        # self.draw_the_snake(self.canvas, self.snake_list)
        # self.setlable()


class Individual:
    def __init__(self, generation=0, No=0, gene=None):
        self.gene = gene  # gene has a shape like tensor(inpur*hidden1+hidden1*hidden2+hidden2*output)
        self.fitness = 0
        self.generation = generation
        self.No = No
        self.score = 0

    # 基因为一个tensor，需要将其按一定格式读入Model中
    def gene2State(self):
        with torch.no_grad():
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
        self.fitness = (score + 2 ** min(score + 6, 13)) ** 2 + 10000 / (
                time ** 0.01 + 1 / time ** 0.01) - 100.0 * distance
        # / - 10.0 / (energy + 1)
        self.score = score
        return

    def CalFitness(self, snakegame: SnakeGame):
        snakegame.net.load_state_dict(self.gene2State())
        while snakegame.game_loop():
            return
        # snakegame.win.mainloop()
        dist = abs(snakegame.snake_list[0][0] - snakegame.Food_pos[0]) + abs(
            snakegame.snake_list[0][1] - snakegame.Food_pos[1])
        self.getFitnessParam(snakegame.Score, snakegame.Time, dist, snakegame.Energy)
        snakegame.Restart_game()


# 选择父母
def SelectChild(childs, f_mean):
    sorted_child = sorted(childs, key=lambda x: x.fitness)
    rate_all = [0]
    # f_all = 0
    # for i in childs:
    #     f_all += i.fitness - f_mean
    # f_all = max(1e-5, f_all)
    for i in range(len(childs)):
        rate = 2 - 1.3 + 2 * (1.3 - 1.0) * i / (len(childs) - 1)
        # rate = max((childs[i].fitness - f_mean),1e-5) / (f_all + 1) * 100000
        rate_all.append(rate_all[i] + rate)
    select = []
    for i in range(2):
        idx = random.uniform(1, rate_all[-1])
        ind = 0
        while rate_all[ind] < idx:
            ind += 1
        # select.append((childs[ind - 1]))
        select.append(sorted_child[ind - 1])
    return select


def Crossover(parent1, parent2, f_max, f_avg, P_max=0.95, P_min=0.75):
    # 自适应交叉概率
    if max(parent1.fitness, parent2.fitness) < f_avg:
        cross_rate = P_max
    else:
        cross_rate = P_max - (P_max - P_min) * (max(parent1.fitness, parent2.fitness) - f_avg) / (f_max - f_avg + 1)
    gene1 = parent1.gene.contiguous()
    gene2 = parent2.gene.contiguous()
    temp1 = gene1.numpy()
    temp2 = gene2.numpy()
    for i in range(len(temp1)):
        rate = random.uniform(0, 1)
        if rate <= cross_rate:
            temp1[i], temp2[i] = temp2[i], temp1[i]
    return gene1, gene2


def Variation(individual, total_N, cur_n, scale):
    # 自适应变异概率
    mutation_rate = scale * (1.0 - random.random() ** (1.0 - cur_n / total_N))
    gene = individual.gene.numpy()
    mutation_array = np.random.random(964) < mutation_rate
    mutation = np.random.standard_cauchy(size=964)
    mutation *= 0.1
    gene[mutation_array] += mutation[mutation_array]
    return individual


# 遗传算法主方法
def genetic_algorithm(population_size, num_generation, ele_indi, mutationScale, Pmax, Pmin, gene_length=None,
                      LastGeneration=None, Fps=100, foldID=1):
    # best=[]
    No = 1
    game = SnakeGame(row=10, column=10)
    # 初始化群体或读取上一代最优个体基因
    population = []
    if not LastGeneration:
        while len(population) < population_size:
            snake = Individual(0, No, torch.tensor(np.random.uniform(-1, 1, 964), dtype=torch.float64))
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
        # best.append(best_individual)
        with open(r'./All/' + str(foldID) + '/' + str(best_individual.generation) + '-' + str(
                best_individual.No) + '.pkl', 'wb') as ch:
            pickle.dump(best_individual, ch)
        # logging
        with open(r'./All/'+str(foldID) + '/evoluiton.txt', 'a') as log:
            log.write(
                f'当前代数: {best_individual.generation}\t最优个体: {best_individual.No}\t'
                f'适应性: {best_individual.fitness}\t 得分: {best_individual.score}\n')
        No = 1
        # 精英个体保留
        # next_generation = [Individual(best_individual.generation + 1, No, best_individual.gene.contiguous())]
        # next_generation[0].fitness = best_individual.fitness
        # 不保留
        # next_generation = [best_individual]*int(population_size/2)
        next_generation = []
        while len(next_generation) < population_size:
            parents = SelectChild(population, f_mean)
            parent1, parent2 = parents[0:2]
            # parent1, parent2 = random.sample(parents, 2)
            gene1, gene2 = Crossover(parent1, parent2, best_individual.fitness, f_mean, Pmax, Pmin)
            child1 = Individual(parent1.generation + 1, No, gene1)
            No += 1
            child2 = Individual(parent2.generation + 1, No, gene2)
            No += 1
            child1 = Variation(child1, parents[0].generation+num_generation, parents[0].generation, mutationScale)
            child2 = Variation(child2, parents[0].generation+num_generation, parents[0].generation, mutationScale)
            child1.CalFitness(game)
            child2.CalFitness(game)
            # # 保存样本
            # with open(r'./All/' + str(child1.generation) + '-' + str(child1.No) + '.pkl', 'wb') as ch:
            #     pickle.dump(child1, ch)
            # with open(r'./All/' + str(child2.generation) + '-' + str(child2.No) + '.pkl', 'wb') as ch:
            #     pickle.dump(child2, ch)
            next_generation.extend([child1, child2])
        next_generation = sorted(next_generation, key=lambda n: n.fitness, reverse=True)
        population = sorted(population, key=lambda n: n.fitness, reverse=True)
        for x in range(1, ele_indi + 1):
            worst_individual = next_generation[-x]
            last_best = Individual(worst_individual.generation, worst_individual.No, population[x - 1].gene)
            last_best.fitness = population[x - 1].fitness
            last_best.score = population[x - 1].score
            next_generation.pop(-x)
            next_generation.append(last_best)
        population = next_generation
    best_individual = max(population, key=lambda n: n.fitness)
    # best.append(best_individual)
    # best.sort(key=lambda x:x.fitness,reverse=True)
    del game
    return best_individual


def mpGA(task, resultQueue):
    while True:
        atask = task.get()
        if atask is None:
            break
        foldID, population_num, generation_num, ele, mutation, cross_max, cross_min = atask
        # 读取最优样本
        mp_last_generation = None
        mp_path = os.listdir(r'./Best')
        mp_path = sorted(mp_path, key=lambda x: int(x.split('-')[0]), reverse=True)
        if mp_path:
            with open(r'./Best/' + mp_path[0], 'rb') as f:
                mp_last_generation = pickle.load(f)
        mp_best_individual = genetic_algorithm(population_num, generation_num, ele, mutation,
                                               cross_max, cross_min, mp_last_generation, Fps=0,foldID=foldID)
        resultQueue.put(mp_best_individual)
        # with open(r'./best/' + str(best.generation) + '-' + str(best.No) + '.pkl', 'w+b') as f:
        #     pickle.dump(best, f)
        # print(f"最好个体:{best.generation}-{best.No} \t 适应度{best.fitness}")


if __name__ == '__main__':
    populations = 1000
    generations = 100
    ele_individual = 0
    mutation_scale = 0.2
    crossMax = 0.6
    crossMin = 0.4
    isMutiProcess = True
    for _ in range(20):
        if isMutiProcess:
            n_workers = mp.cpu_count()
            task_queue = mp.Queue()
            result_queue = mp.Queue()
            workers = []
            for _ in range(n_workers):
                worker = mp.Process(target=mpGA, args=(task_queue, result_queue))
                worker.start()
                workers.append(worker)

            for fold_id in range(10):
                task_queue.put((fold_id, populations, generations, ele_individual, mutation_scale, crossMax, crossMin))

            result = []
            for _ in range(10):
                result.append(result_queue.get())
            best = max(result, key=lambda n: n.fitness)
            with open(r'./best/' + str(best.generation) + '-' + str(best.No) + '.pkl', 'w+b') as f:
                pickle.dump(best, f)
            print(f"最好个体:{best.generation}-{best.No} \t 适应度{best.fitness}\t 得分: {best.score}")

            for _ in range(n_workers):
                task_queue.put(None)
            for worker in workers:
                worker.join()
        else:
            # 读取最优样本
            last_generation = None
            path = os.listdir(r'./Best')
            path = sorted(path, key=lambda x: int(x.split('-')[0]), reverse=True)
            if path:
                with open(r'./Best/' + path[0], 'rb') as f:
                    last_generation = pickle.load(f)
            BestIndividual = genetic_algorithm(populations, generations, ele_individual,
                                               mutation_scale, crossMax, crossMin, None, last_generation, Fps=0)
