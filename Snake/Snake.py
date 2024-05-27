import tkinter as tk
import random as rd


# 绘制一个像素
def draw_a_unit(can, col, row, unit_color="green"):
    x1 = col * Unit_size
    y1 = row * Unit_size
    x2 = (col + 1) * Unit_size
    y2 = (row + 1) * Unit_size
    can.create_rectangle(x1, y1, x2, y2, fill=unit_color, outline="Dimgray")


# 绘制背景
def put_a_background(can, color='Dimgray'):
    global game_map
    for x in range(Column):
        for y in range(Row):
            draw_a_unit(can, x, y, unit_color=color)
            game_map.append([x, y])


# 绘制蛇
def draw_the_snake(can, snake, color='green'):
    for i in snake:
        if i == head:
            draw_a_unit(can, i[0], i[1], unit_color='black')
        else:
            draw_a_unit(can, i[0], i[1], unit_color=color)


# 判断方向
def find_dirction(event):
    global Dirc
    # 判断是否可以向上向下操作
    # 如果snake_list[0] 和 [1] 的x轴坐标相同，意味着不可以改变上、下方向
    # 若y轴坐标相同，意味着不可以改变左、右方向
    ch = event.keysym
    delta_x = Dirc[0]
    delta_y = Dirc[1]
    if ch == 'w':
        if snake_list[0][0] != snake_list[1][0]:  # or snake_list[0][1] < snake_list[1][1]:
            delta_x = 0
            delta_y = -1
        if snake_list[0][1] < snake_list[1][1]:
            move_snake(snake_list, [delta_x, delta_y], True)
    elif ch == 's':
        if snake_list[0][0] != snake_list[1][0]:  # or snake_list[0][1] > snake_list[1][1]:
            delta_x = 0
            delta_y = 1
        if snake_list[0][1] > snake_list[1][1]:
            move_snake(snake_list, [delta_x, delta_y], True)
    elif ch == 'a':
        if snake_list[0][1] != snake_list[1][1]:  # or snake_list[0][0] < snake_list[1][0]:
            delta_y = 0
            delta_x = -1
        if snake_list[0][0] < snake_list[1][0]:
            move_snake(snake_list, [delta_x, delta_y], True)
    elif ch == 'd':
        if snake_list[0][1] != snake_list[1][1]:  # or snake_list[0][0] > snake_list[1][0]:
            delta_y = 0
            delta_x = 1
        if snake_list[0][0] > snake_list[1][0]:
            move_snake(snake_list, [delta_x, delta_y], True)
    Dirc = [delta_x, delta_y]
    return


# 生成食物,判断是否有食物
def food(snke_list):
    global Have_food, Food_pos, canvas
    if Have_food:
        return
    valid_position = [i for i in game_map if i not in snke_list]
    Food_pos = rd.choice(valid_position)
    draw_a_unit(canvas, Food_pos[0], Food_pos[1], unit_color='red')
    Have_food = True
    return


# 判断蛇吃食物
def snake_eat(snke_list, pos):
    global Have_food
    head_0 = snke_list[0]
    if head_0 == pos:
        Have_food = False
        return 1
    else:
        return 0


# 移动蛇
def move_snake(snke_list, direc, rush):
    global canvas, Score, Energy, Time
    if direc != [0, 0]:
        global Food_pos
        head_0 = snke_list[0]
        temp_head = head_0.copy()
        temp_head[0] = temp_head[0] + direc[0]
        temp_head[1] = temp_head[1] + direc[1]
        snke_list.insert(0, temp_head)
        if not snake_eat(snke_list, Food_pos):
            Energy -= 1
            str_energy.set('Energy:' + str(Energy))
            draw_a_unit(canvas, snke_list[-1][0], snke_list[-1][1], unit_color="Dimgray")
            snke_list.pop(-1)
            draw_a_unit(canvas, snke_list[1][0], snke_list[1][1])
            draw_a_unit(canvas, temp_head[0], temp_head[1], unit_color="black")
        else:
            Score += 1
            Energy = min(400, Energy + 199)
            str_score.set('Score:' + str(Score))
            str_energy.set('Energy:' + str(Energy))
            draw_a_unit(canvas, snke_list[1][0], snke_list[1][1])
            draw_a_unit(canvas, temp_head[0], temp_head[1], unit_color="black")
        if rush:
            Time += 1
        else:
            Time += 2
        str_time.set('Time:' + str(Time))
    return snke_list


# 游戏结束
def game_over(snke_list):
    global Energy
    x = snke_list[0][0]
    y = snke_list[0][1]
    set_list = snke_list[1:]
    # 判断头和身体是否重叠或判断超出边界
    if snke_list[0] in set_list or x < 0 or x > Column - 1 or y < 0 or y > Row - 1 or Energy <= 0:
        return 1
    else:
        return 0


def continue_game(windows):
    global pause_flag
    pause_flag = -1
    win.after(1000, game_loop)
    windows.destroy()


def Exit_game(event=None):
    exit()


pause_flag = -1


def Pause_game(event=None):
    global pause_flag
    pause_flag = -pause_flag


def game_loop():
    global snake_list, Dirc, pause_flag, Time
    win.update()
    food(snake_list)
    snake_list = move_snake(snake_list, Dirc, False)
    if game_over(snake_list):
        global over_label
        over_label = tk.Label(win, text='Game Over', font=('楷体', 25), width=15, height=1)
        over_label.place(x=(Width - 260) / 2, y=(Height - 40) / 2, bg=None)
        return
    # if not snake_list:
    #     return
    # TODO 修改FPS
    if pause_flag == -1:
        win.after(100, game_loop)
    else:
        second = tk.Tk()
        pause_button = tk.Button(second, width=20, height=1, text="Continue", command=lambda: continue_game(second),
                                 relief='raised')
        pause_button.pack_configure(expand=1)
        second.geometry("%dx%d+%d+%d" % (20, 30, 400, 400))


def key_bind(canvas):
    canvas.focus_set()
    canvas.bind("<KeyPress-a>", find_dirction)
    canvas.bind("<KeyPress-d>", find_dirction)
    canvas.bind("<KeyPress-w>", find_dirction)
    canvas.bind("<KeyPress-s>", find_dirction)
    canvas.bind("<KeyPress-r>", Restart_game)
    canvas.bind("<KeyPress-Escape>", Exit_game)
    canvas.bind("<KeyPress-space>", Pause_game)


over_label = 1


def Restart_game(event):
    global over_label, win, canvas, Dirc, head, Score, game_map, snake_list, \
        Food_pos, Have_food, str_score, str_energy, Energy, Time
    if not over_label is int:
        over_label.destroy()
    Dirc = [0, 0]
    Score = 0
    Energy = 400
    Time = 0
    rx = rd.randint(1, 19)
    deltax = rd.choice([1, -1])
    ry = rd.randint(1, 19)
    head = [rx, ry]
    snake_list = [head, [rx + deltax, ry]]
    Food_pos = []
    Have_food = False
    put_a_background(canvas)
    draw_the_snake(canvas, snake_list)
    str_score.set('Score:' + str(Score))
    str_energy.set('Energy:' + str(Energy))
    str_time.set("Time:" + str(Time))
    key_bind(canvas)
    game_loop()
    win.mainloop()


Unit_size = 20  # 一个像素大小

Row = 40
Column = 40
Height = Row * Unit_size
Width = Column * Unit_size
win = tk.Tk()
win.title("Snake")
canvas = tk.Canvas(win, width=Width, height=Height + 2 * Unit_size)
canvas.pack()
Dirc = [0, 0]
Score = 0
game_map = []

# 随机生成蛇的位置
rx = rd.randint(1, 19)
deltax = rd.choice([1, -1])
ry = rd.randint(1, 19)
head = [rx, ry]
snake_list = [head, [rx + deltax, ry]]
Energy = 400
Time = 0
Food_pos = []
Have_food = False
put_a_background(canvas)
draw_the_snake(canvas, snake_list)
str_score = tk.StringVar()
str_energy = tk.StringVar()
str_time = tk.StringVar()
score_label = tk.Label(win, textvariable=str_score, font=('楷体', 20), width=10, height=1)
energy_label = tk.Label(win, textvariable=str_energy, font=('楷体', 20), width=10, height=1)
time_label = tk.Label(win, textvariable=str_time, font=('楷体', 20), width=10, height=1)
str_score.set('Score:' + str(Score))
str_energy.set('Energy:' + str(Energy))
str_time.set("Time:" + str(Time))
score_label.place(x=(Width - 400) / 2, y=Height)
energy_label.place(x=(Width - 100) / 2, y=Height)
time_label.place(x=(Width + 200) / 2, y=Height)
# 绑定按键
key_bind(canvas)
# 游戏主程序
if __name__ == '__main__':
    game_loop()
    win.mainloop()
