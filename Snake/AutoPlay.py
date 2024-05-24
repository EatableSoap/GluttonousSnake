import operator
import tkinter as tk
from snake_class import Snake
import heapq


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
    def __init__(self, row=40, column=40, Fps=100, Unit_size=20):
        super().__init__(row, column, Fps, Unit_size)
        self.path = []

    # 判断是否需要将一个邻接点加入opnelist中
    def judge_open(self, open_list, neighbor):
        for node in open_list:
            # 如果已有f更小的相同节点，则不加入
            if neighbor == node[1] and neighbor.f > node[1].f:  # eval('neighbor.f' + signal + 'node[1].f'):
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
                return path[::-1]

            [x, y] = current.Position
            neighbors = [[x + 1, y], [x - 1, y], [x, y - 1], [x, y + 1]]

            for next_node in neighbors:
                if (0 <= next_node[0] < self.Column and 0 <= next_node[1]
                        < self.Row and next_node not in body[:-1]):
                    neighbor = Node(current, next_node)
                    if neighbor in closed_list:
                        continue
                    neighbor.g = current.g + 1
                    neighbor.h = abs(neighbor.Position[0] - end.Position[0]) + abs(
                        neighbor.Position[1] - end.Position[1])
                    neighbor.f = neighbor.g + neighbor.h

                    # 判断最短路径
                    if self.judge_open(open_list, neighbor):
                        heapq.heappush(open_list, (neighbor.f, neighbor))
                        # if next_node[0] != goal[0] and next_node[1] != goal[1]:
                        #     self.draw_a_unit(self.canvas, next_node[0], next_node[1], unit_color='blue')

        while open_list:
            heapq.heappop(open_list)
        return None

    def FarfromFood(self, body, goal):
        [x, y] = body[0]
        direction = [body[0][0] - body[1][0], body[0][1] - body[1][1]]
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
        return max_path

    def FindLonggest(self, body, tail):
        [x, y] = body[0]
        direction = [body[0][0] - body[1][0], body[0][1] - body[1][1]]
        near = []
        path = []
        for i in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
            temp = [x + i[0], y + i[1]]
            if i != direction and operator.eq(temp, self.Food_pos):
                near.append(temp)
        near.sort(key=lambda next_node: abs(next_node[0] - tail[0]) + abs(next_node[1] - tail[1]), reverse=True)
        for i in near:
            sim_body = body.copy().append(i).pop(0)
            if len(self.AStar(sim_body, sim_body[-1])) > 1:
                path.append(i)
        return path

    def SolveWay(self, body, goal):
        print('P1')
        path = self.AStar(body, goal)
        if path:
            if len(body) + 1 == self.Column * self.Row:
                print('P2')
                return path
            else:
                print('P3')
                sim_body = body.copy()
                for d in range(len(path)):
                    sim_body.insert(0, path[d])
                    if d != len(path) - 1:
                        sim_body.pop(-1)
                sim_path = self.AStar(sim_body, sim_body[-1])
                if sim_path:
                    return path
        print('P4')
        longest = self.FindLonggest(body, body[-1])
        if longest:
            return longest
        print('P5')
        return self.FarfromFood(body, self.Food_pos)

    def move(self):
        self.path = self.SolveWay(self.snake_list, self.Food_pos)
        assert self.path is not None and self.path != [[]], 'path is None'
        temp_path = [self.snake_list[0]] + self.path[:-1].copy()
        path_len = len(self.path)
        print(self.path)
        for i in range(path_len):
            self.path[i] = [self.path[i][0] - temp_path[i][0],
                            self.path[i][1] - temp_path[i][1]]

    def game_loop(self):
        self.win.update()
        self.food(self.snake_list)
        if not self.path:
            self.move()
        else:
            self.Dirc = self.path.pop(0)
        self.snake_list = self.move_snake(self.snake_list, self.Dirc, False)
        if self.game_over(self.snake_list):
            self.over_label = tk.Label(self.win, text='Game Over', font=('楷体', 25), width=15, height=1)
            self.over_label.place(x=(self.Width - 260) / 2, y=(self.Height - 40) / 2, bg=None)
            self.win.update()
        elif self.pause_flag == -1:
            self.win.after(self.Fps, self.game_loop)
        else:
            second = tk.Tk()
            pause_button = tk.Button(second, width=20, height=1, text="Continue",
                                     command=lambda: self.continue_game(second),
                                     relief='raised')
            pause_button.pack_configure(expand=1)
            second.geometry("%dx%d+%d+%d" % (20, 30, 400, 400))


if __name__ == '__main__':
    game = FindWay(Fps=100, column=10, row=10, Unit_size=20)
    game.game_loop()
    game.win.mainloop()
