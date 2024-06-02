import random
import tkinter as tk
import random as rd


class Snake:
    # 绘制一个像素
    def draw_a_unit(self, can, col, row, unit_color="green"):
        x1 = col * self.Unit_size
        y1 = row * self.Unit_size
        x2 = (col + 1) * self.Unit_size
        y2 = (row + 1) * self.Unit_size
        can.create_rectangle(x1, y1, x2, y2, fill=unit_color, outline="")
        self.win.update()

    # 绘制背景
    def put_a_background(self, can, color='white'):
        for x in range(self.Column):
            for y in range(self.Row):
                self.draw_a_unit(can, x, y, unit_color=color)
                self.game_map.append([x, y])
        self.win.update()

    # 绘制蛇
    def draw_the_snake(self, can, snake, color='green'):
        for i in range(len(snake)):
            if i == 0:
                self.draw_a_unit(can, snake[i][0], snake[i][1], unit_color='purple')
            else:
                self.draw_a_unit(can, snake[i][0], snake[i][1], unit_color=color)

    # 判断方向
    def find_dirction(self, event):
        if not self.pause_flag == 1:
            # 判断是否可以向上向下操作
            # 如果snake_list[0] 和 [1] 的x轴坐标相同，意味着不可以改变上、下方向
            # 若y轴坐标相同，意味着不可以改变左、右方向
            ch = event.keysym
            delta_x = self.Dirc[0]
            delta_y = self.Dirc[1]
            if ch == 'w':
                if self.snake_list[0][0] != self.snake_list[1][0]:  # or snake_list[0][1] < snake_list[1][1]:
                    delta_x = 0
                    delta_y = -1
                if self.snake_list[0][1] < self.snake_list[1][1]:
                    self.move_snake(self.snake_list, [delta_x, delta_y], True)
            elif ch == 's':
                if self.snake_list[0][0] != self.snake_list[1][0]:  # or snake_list[0][1] > snake_list[1][1]:
                    delta_x = 0
                    delta_y = 1
                if self.snake_list[0][1] > self.snake_list[1][1]:
                    self.move_snake(self.snake_list, [delta_x, delta_y], True)
            elif ch == 'a':
                if self.snake_list[0][1] != self.snake_list[1][1]:  # or self.snake_list[0][0] < self.snake_list[1][0]:
                    delta_y = 0
                    delta_x = -1
                if self.snake_list[0][0] < self.snake_list[1][0]:
                    self.move_snake(self.snake_list, [delta_x, delta_y], True)
            elif ch == 'd':
                if self.snake_list[0][1] != self.snake_list[1][1]:  # or self.snake_list[0][0] > self.snake_list[1][0]:
                    delta_y = 0
                    delta_x = 1
                if self.snake_list[0][0] > self.snake_list[1][0]:
                    self.move_snake(self.snake_list, [delta_x, delta_y], True)
            self.Dirc = [delta_x, delta_y]
        return

    # 生成食物,判断是否有食物和获胜条件
    def food(self, snke_list):
        if self.Have_food:
            return
        valid_position = [i for i in self.game_map if i not in snke_list]
        if valid_position:
            self.Food_pos = rd.choice(valid_position)
            self.draw_a_unit(self.canvas, self.Food_pos[0], self.Food_pos[1], unit_color='red')
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
            return 1
        else:
            return 0

    # 移动蛇
    def move_snake(self, snke_list, direc, rush):
        if direc != [0, 0]:
            head_0 = snke_list[0]
            temp_head = head_0.copy()
            temp_head[0] = temp_head[0] + direc[0]
            temp_head[1] = temp_head[1] + direc[1]
            snke_list.insert(0, temp_head)
            if not self.snake_eat(snke_list, self.Food_pos):
                self.Energy -= 1
                self.str_energy.set('Energy:' + str(self.Energy))
                self.draw_a_unit(self.canvas, snke_list[-1][0], snke_list[-1][1], unit_color="white")
                snke_list.pop(-1)
                self.draw_a_unit(self.canvas, snke_list[1][0], snke_list[1][1])
                self.draw_a_unit(self.canvas, temp_head[0], temp_head[1], unit_color="purple")
                self.draw_a_unit(self.canvas, snke_list[-1][0], snke_list[-1][1], unit_color='orange')
            else:
                self.Score += 1
                self.Energy = min(int(self.Column * self.Row), self.Energy + int(self.Column * self.Row * 0.6))
                self.str_score.set('Score:' + str(self.Score))
                self.str_energy.set('Energy:' + str(self.Energy))
                self.draw_a_unit(self.canvas, snke_list[-1][0], snke_list[-1][1], unit_color="orange")
                self.draw_a_unit(self.canvas, snke_list[1][0], snke_list[1][1])
                self.draw_a_unit(self.canvas, temp_head[0], temp_head[1], unit_color="purple")
            if rush:
                self.Time += 1
            else:
                self.Time += 2
            self.str_time.set('Time:' + str(self.Time))
            self.win.update()
            del temp_head
        return snke_list

    # 游戏结束
    def game_over(self, snke_list):
        x = snke_list[0][0]
        y = snke_list[0][1]
        set_list = snke_list[1:]
        # 判断头和身体是否重叠或判断超出边界
        if (self.over or snke_list[0] in set_list or x < 0 or x > self.Column - 1
                or y < 0 or y > self.Row - 1 or self.Energy <= 0):
            self.pause_flag = 1
            return True
        else:
            return False

    def continue_game(self, windows):
        self.pause_flag = -1
        self.win.after(1000, self.game_loop)
        windows.destroy()

    def Exit_game(self, event=None):
        exit()

    def Pause_game(self, event=None):
        self.pause_flag = -self.pause_flag

    def game_loop(self):
        self.win.update()
        self.food(self.snake_list)
        if self.winFlag:
            self.over_label = tk.Label(self.win, text='You Win!', font=('楷体', 25), width=15, height=1)
            self.over_label.place(x=(self.Width - 260) / 2, y=(self.Height - 40) / 2, bg=None)
            self.win.update()
        self.snake_list = self.move_snake(self.snake_list, self.Dirc, False)
        if self.game_over(self.snake_list):
            self.over_label = tk.Label(self.win, text='Game Over', font=('楷体', 25), width=15, height=1)
            self.over_label.place(x=(self.Width - 260) / 2, y=(self.Height - 40) / 2, bg=None)
            self.win.update()
            return False
        else:
            return True
        # elif self.pause_flag == -1:
        #     self.win.after(self.Fps, self.game_loop)
        # else:
        #     second = tk.Tk()
        #     pause_button = tk.Button(second, width=20, height=1, text="Continue",
        #                              command=lambda: self.continue_game(second),
        #                              relief='raised')
        #     pause_button.pack_configure(expand=1)
        #     second.geometry("%dx%d+%d+%d" % (20, 30, 400, 400))

    def addFps(self, event=None):
        self.Fps += 10
        return

    def subFps(self, event=None):
        self.Fps = max(0, self.Fps - 10)
        return

    def key_bind(self, canvas):
        canvas.focus_set()
        canvas.bind("<KeyPress-a>", self.find_dirction)
        canvas.bind("<KeyPress-d>", self.find_dirction)
        canvas.bind("<KeyPress-w>", self.find_dirction)
        canvas.bind("<KeyPress-s>", self.find_dirction)
        canvas.bind("<KeyPress-r>", self.Restart_game)
        canvas.bind("<KeyPress-Escape>", self.Exit_game)
        canvas.bind("<KeyPress-space>", self.Pause_game)
        canvas.bind("<KeyPress-Up>", self.addFps)
        canvas.bind("<KeyPress-Down>", self.subFps)

    def Restart_game(self, event=None):
        if self.over_label is not int:
            self.over_label.destroy()
            self.win.update()
        self.winFlag = 0
        self.pause_flag = -1
        self.Dirc = [0, 0]
        self.Score = 0
        self.Energy = int(self.Column * self.Row)
        self.Time = 0
        self.snake_list = self.ramdom_snake()
        self.Food_pos = []
        self.Have_food = False
        self.over = False
        self.put_a_background(self.canvas)
        self.draw_the_snake(self.canvas, self.snake_list)
        self.setlable()

    def ramdom_snake(self):
        if not self.seeds:
            random.seed(self.seeds)
        # 随机生成蛇的位置
        rx = rd.randint(1, self.Row - 2)
        delta = rd.choice([1, -1])
        ry = rd.randint(1, self.Row - 2)
        head = [rx, ry]
        rate = random.random()
        if rate >= 0.5:
            return [head, [rx + delta, ry]]
        else:
            return [head, [rx, ry + delta]]

    def setlable(self):
        self.str_score.set('Score:' + str(self.Score))
        self.str_energy.set('Energy:' + str(self.Energy))
        self.str_time.set("Time:" + str(self.Time))

    def set_in_windows(self):
        screenWidth = self.win.winfo_screenwidth()  # 获取显示区域的宽度
        screenHeight = self.win.winfo_screenheight()  # 获取显示区域的高度
        left = (screenWidth - self.Width) / 2
        top = (screenHeight - (self.Height + 2 * self.Unit_size)) / 2
        self.win.geometry("%dx%d+%d+%d" % (self.Width, self.Height + 4 * self.Unit_size, left, top))
        return left, top

    def __init__(self, row=40, column=40, Fps=100, Unit_size=20, seeds=None):
        self.seeds = seeds
        self.over = False
        self.winFlag = 0
        self.Fps = Fps
        self.pause_flag = -1
        self.over_label = 1
        self.Unit_size = Unit_size  # 一个像素大小
        self.Row = row
        self.Column = column
        self.Height = row * self.Unit_size
        self.Width = column * self.Unit_size
        self.win = tk.Tk()
        self.win.attributes("-transparentcolor", "white")
        self.win.title("Snake")
        self.canvas = tk.Canvas(self.win, width=self.Width, height=self.Height)  # * self.Unit_size)
        self.canvas.grid()
        self.Dirc = [0, 0]
        self.Score = 0
        self.game_map = []
        self.snake_list = self.ramdom_snake()
        self.Energy = int(self.Column * self.Row)
        self.Time = 0
        self.Food_pos = []
        self.Have_food = False
        self.put_a_background(self.canvas)
        self.draw_the_snake(self.canvas, self.snake_list)
        self.str_score = tk.StringVar()
        self.str_energy = tk.StringVar()
        self.str_time = tk.StringVar()
        self.score_label = tk.Label(self.win, textvariable=self.str_score, font=('楷体', int(0.75 * Unit_size)),
                                    width=10, height=1)
        self.energy_label = tk.Label(self.win, textvariable=self.str_energy, font=('楷体', int(0.75 * Unit_size)),
                                     width=10,
                                     height=1)
        self.time_label = tk.Label(self.win, textvariable=self.str_time, font=('楷体', int(0.75 * Unit_size)), width=10,
                                   height=1)
        self.setlable()
        # self.score_label.place(x=(self.Width - 400) / 2, y=self.Height)
        # self.energy_label.place(x=(self.Width - 100) / 2, y=self.Height)
        # self.time_label.place(x=(self.Width + 200) / 2, y=self.Height)
        self.score_label.grid()
        self.energy_label.grid()
        self.time_label.grid()
        self.set_in_windows()
        # 绑定按键
        self.key_bind(self.canvas)


if __name__ == '__main__':
    game = Snake(Unit_size=20)
    game.game_loop()
    game.win.mainloop()
