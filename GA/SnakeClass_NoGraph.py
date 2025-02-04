import random


class Snake:
    # 生成食物,判断是否有食物和获胜条件
    def food(self, snke_list):
        if self.seeds is not None and self.enableseed:  # 没有种子或平均分数小于10时，可认为在探索阶段
            random.seed(self.seeds)
        if self.Have_food:
            return
        valid_position = [i for i in self.game_map if i not in snke_list]
        if valid_position:
            self.Food_pos = random.choice(valid_position)
            self.Have_food = True
        else:
            self.winFlag = 1
        del valid_position
        return

    # 判断蛇吃食物
    def snake_eat(self, snke_list, pos):
        head_0 = snke_list[0]
        if head_0 == pos:
            self.Have_food = False
            return True
        else:
            return False

    # 移动蛇
    def move_snake(self, snke_list, direc, rush):
        isEat = False
        if direc != [0, 0]:
            head_0 = snke_list[0]
            temp_head = head_0.copy()
            temp_head[0] = temp_head[0] + direc[0]
            temp_head[1] = temp_head[1] + direc[1]
            snke_list.insert(0, temp_head)
            self.Steps += 1
            # 如果吃到就不弹出蛇尾
            if not self.snake_eat(snke_list, self.Food_pos):
                self.Energy -= 1
                snke_list.pop(-1)
            else:
                isEat = True
                self.Score += 1
                self.Energy = min(int(self.Column * self.Row), self.Energy + int(self.Column * self.Row * 0.6))
        return isEat, snke_list

    # 游戏结束
    def game_over(self, snke_list):
        x = snke_list[0][0]
        y = snke_list[0][1]
        set_list = snke_list[1:]
        # 判断头和身体是否重叠或判断超出边界
        if (self.over or snke_list[0] in set_list or x < 0
                or x > self.Column - 1 or y < 0 or y > self.Row - 1 or self.Energy <= 0):
            return True
        else:
            return False

    def Restart_game(self, event=None):
        self.over = False
        self.winFlag = 0
        self.Dirc = [0, 0]
        self.Score = 0
        self.Energy = int(self.Column * self.Row * 0.5)
        self.Steps = 0
        self.snake_list = self.ramdom_snake()
        self.Food_pos = []
        self.Have_food = False

    def ramdom_snake(self):
        if self.seeds is not None and self.enableseed:
            random.seed(self.seeds)
        # 随机生成蛇的位置
        rx = random.randint(1, self.Row - 2)
        delta = random.choice([1, -1])
        ry = random.randint(1, self.Row - 2)
        head = [rx, ry]
        rate = random.random()
        if rate >= 0.5:
            return [head, [rx + delta, ry]]
        else:
            return [head, [rx, ry + delta]]

    def __init__(self, row=20, column=20, useSeeds=True, seeds=None):
        self.enableseed = useSeeds
        self.seeds = seeds
        self.over = False
        self.winFlag = 0
        self.Row = row
        self.Column = column
        self.Dirc = [0, 0]
        self.Score = 0
        self.game_map = [[x, y] for x in range(column) for y in range(row)]
        self.snake_list = self.ramdom_snake()
        self.Energy = int(self.Column * self.Row)
        self.Steps = 0
        self.Food_pos = []
        self.Have_food = False

    def game_loop(self):
        self.food(self.snake_list)
        if self.winFlag:
            return False
        _,self.snake_list = self.move_snake(self.snake_list, self.Dirc,False)
        if self.game_over(self.snake_list):
            return False
        return True
