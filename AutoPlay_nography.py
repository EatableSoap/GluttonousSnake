import copy
import operator
from GluttonousSnake.GA.SnakeClass_NoGraph import Snake
import heapq

scores = []
step = []


# 搜索树
class Node:
    def __init__(self, parent=None, Position=None):
        self.parent = parent
        self.Position = Position
        self.f = 0
        self.g = 0
        self.h = 0

    def __eq__(self, other):
        return self.Position == other.Position

    # 用于维护优先队列
    def __lt__(self, other):
        return self.f < other.f


class FindWay(Snake):
    global scores, step

    def __init__(self, row=40, column=40, Fps=100, Unit_size=20, visualize=False, g_value: float = 1):
        super().__init__(row, column, Fps, Unit_size)
        self.path = []
        self.visualize = visualize
        self.g_value = g_value

    # 判断是否需要将一个邻接点加入opnelist中
    def judge_open(self, open_list, neighbor):
        for node in open_list:
            # 如果已有f更小的相同节点，则不加入
            if neighbor == node[1] and neighbor.f >= node[1].f:  # eval('neighbor.f' + signal + 'node[1].f'):
                return False
        return True

    # 由于贪吃蛇仅能沿头的两边和前方移动，故无需考虑对角代价，因此每一步的实际代价均为哈密顿距离
    def AStar(self, body, goal):
        start = Node(None, body[0])
        end = Node(None, goal)

        open_list = []  # 存储可访问节点
        closed_list = []  # 存储已访问节点

        heapq.heappush(open_list, (start.f, start))
        while open_list:
            current = heapq.heappop(open_list)[1]
            closed_list.append(current)

            if current == end:
                path = []
                while current.parent:
                    path.append(current.Position)
                    current = current.parent
                del open_list
                return path[::-1]

            [x, y] = current.Position
            neighbors = [[x + 1, y], [x - 1, y], [x, y - 1], [x, y + 1]]

            for next_node in neighbors:
                if (0 <= next_node[0] < self.Column and 0 <= next_node[1]
                        < self.Row and next_node not in body):
                    neighbor = Node(current, next_node)
                    if neighbor in closed_list:
                        continue
                    neighbor.g = current.g + self.g_value
                    neighbor.h = abs(neighbor.Position[0] - end.Position[0]) + abs(
                        neighbor.Position[1] - end.Position[1])
                    neighbor.f = neighbor.g + neighbor.h

                    # 判断最短路径
                    if self.judge_open(open_list, neighbor):
                        heapq.heappush(open_list, (neighbor.f, neighbor))
                    del neighbor
        del open_list
        return None

    # TODO 或许有修改空间,wander原理
    def FarfromFood(self, body, goal):
        [x, y] = body[0]
        direction = [body[1][0] - body[0][0], body[1][1] - body[0][1]]
        near = []
        max_dist = -1
        max_path = []
        for i in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
            temp = [x + i[0], y + i[1]]
            if i != direction:
                near.append(temp)
        for next_node in near:
            if 0 <= next_node[0] < self.Column and 0 <= next_node[1] < self.Row and next_node not in body:
                dist = abs(next_node[0] - goal[0]) + abs(next_node[1] - goal[1])
                if dist > max_dist:
                    max_dist = dist
                    max_path = next_node
        if not max_path:
            return [[-1, -1]]
        return [max_path]

    # TODO 优化，将可填充空间尽可能填充
    def FindLonggest(self, body, tail):
        [x, y] = body[0]
        direction = [body[0][0] - body[1][0], body[0][1] - body[1][1]]
        near = []
        path = []
        for i in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
            temp = [x + i[0], y + i[1]]
            if not operator.eq(i, direction) and not operator.eq(temp, self.Food_pos) and \
                    0 <= temp[0] < self.Column and 0 <= temp[1] < self.Row and temp not in body:
                near.append(temp)
        near.sort(key=lambda next_node: abs(next_node[0] - tail[0]) + abs(next_node[1] - tail[1]), reverse=True)
        for i in near:
            sim_body = copy.deepcopy(body)
            sim_body.insert(0, i)
            sim_body.pop(-1)
            record = self.AStar(sim_body[:-1], sim_body[-1])
            del sim_body
            if record and len(record) > 1:
                path.append(i)
                break
        return path

    def SolveWay(self, body, goal):
        path = self.AStar(body[:-1], goal)
        if path:
            if len(body) + 1 == self.Column * self.Row:
                return path
            else:
                sim_body = body.copy()
                for d in range(len(path)):
                    sim_body.insert(0, path[d])
                    if d != len(path) - 1:
                        sim_body.pop(-1)
                sim_path = self.AStar(sim_body[:-1], sim_body[-1])
                del sim_body
                if sim_path:
                    return path
        longest = self.FindLonggest(body[:-1], body[-1])
        if longest:
            return longest
        return self.FarfromFood(body[:-1], self.Food_pos)

    def move(self):
        path = self.SolveWay(self.snake_list, self.Food_pos)
        if not operator.eq(path, [[-1, -1]]):
            self.path = path
            assert self.path is not None, 'path is None'
            temp_path = [self.snake_list[0]] + self.path[:-1].copy()
            path_len = len(self.path)
            for i in range(path_len):
                self.path[i] = [self.path[i][0] - temp_path[i][0],
                                self.path[i][1] - temp_path[i][1]]
        else:
            self.snake_list = [[-2, -2]]

    def game_loop(self):
        self.food(self.snake_list)
        if self.winFlag:
            scores.append(self.Score)
            step.append(self.Steps)
            self.Restart_game()
        if not self.path:
            self.move()
        try:
            self.Dirc = self.path.pop(0)
            _, self.snake_list = self.move_snake(self.snake_list, self.Dirc, False)
        except:
            self.snake_list = [[0, 0], [0, 0]]
        if self.game_over(self.snake_list):
            scores.append(self.Score)
            step.append(self.Steps)
            self.Restart_game()

    def Restart_game(self, event=None):
        self.winFlag = 0
        self.Dirc = [0, 0]
        self.Score = 0
        self.Energy = 400
        self.Steps = 0
        self.snake_list = self.ramdom_snake()
        self.Food_pos = []
        self.Have_food = False


if __name__ == '__main__':
    '''
    这个就是用来快速跑数据的，实际上和AutoPlay是一个东西
    '''
    game = FindWay(Fps=0, column=10, row=10, Unit_size=20, visualize=False, g_value=1)
    while True:
        game.game_loop()
