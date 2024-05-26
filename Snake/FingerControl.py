import cv2 as cv
import numpy as np
from pynput import keyboard
from SnakeClass import Snake
from FingerModel import FingerModel

# 获取视频地址
path_name = r'D:/AI_project/dataset'
ip = "10.198.247.249"
port = "8080"
url = "http://" + ip + ":" + port + "/video"

# 读取图像，创建游戏类
cap = cv.VideoCapture(url)
game = Snake(Fps=100, row=20, column=20, Unit_size=40)
control = keyboard.Controller()

# 载入模型
Model = FingerModel()
Model.load('./dataset')

# 一些变量初始化
last_frame = np.zeros((108, 108))
count = 0
game_x = 0
game_y = 0
move_list = [[0, 0]]
cv.namedWindow('Phone Camera', 0)
x, y = game.set_in_windows()
cv.moveWindow('Phone Camera', int(x), int(y))
cv.resizeWindow('Phone Camera', game.Width, game.Height)


# 循环读取图像，输出预测坐标
def printpic(local_game):
    global last_frame, count, game_x, game_y, move_list
    ret, frame = cap.read()
    img = frame

    if ret:
        # 每30帧预测一次坐标
        if count == 0:
            local_game.draw_a_unit(local_game.canvas, game_x, game_y, 'white')
            img = Model.cr_otsu(img[:1080, :1080])
            # img_canny = cv.Canny(cv.cvtColor(frame[:1080, :1080], cv.COLOR_BGR2GRAY), 30, 150)
            # img = (img * 0.8 + img_canny * 0.1)
            img = cv.resize(img, (108, 108))  # * 0.95 + last_frame * 0.2
            # last_frame = img
            img = Model.Hog_ft(img)
            pos1 = Model.clf[0].predict(img.reshape(1, -1))
            pos2 = Model.clf[1].predict(img.reshape(1, -1))
            pos_y = int((pos1[0][1]))
            pos_x = int((pos2[0][0]))
            # print(pos_x, pos_y)
            # cv.circle(frame, (pos_x, pos_y), 10, (0, 0, 255), thickness=-1)
            # 预测坐标映射到游戏中
            game_x = pos_x // local_game.Unit_size
            game_y = pos_y // local_game.Unit_size
            # 获取移动序列
            move_list = [[0, 0]]
            head_pos = np.array(local_game.snake_list[0])
            if head_pos[0] != game_x or head_pos[1] != game_y:
                move_vec = np.array([game_x, game_y]) - head_pos
                for i in range(0, abs(move_vec[0])):
                    move_list.append([int(move_vec[0] / abs(move_vec[0])), 0])
                for i in range(0, abs(move_vec[1])):
                    move_list.append([0, int(move_vec[1] / abs(move_vec[1]))])
                local_game.draw_a_unit(local_game.canvas, game_x, game_y, 'yellow')
            if local_game.Have_food and game_x == local_game.Food_pos[0] and game_y == local_game.Food_pos[1]:
                local_game.draw_a_unit(local_game.canvas, local_game.Food_pos[0], local_game.Food_pos[1], 'red')

        # 每5帧移动一次
        if local_game.pause_flag == -1 and count % 5 == 0:
            local_game.move_snake(local_game.snake_list, move_list[0], False)
        if len(move_list) > 1:
            move_list.pop(0)
        temp = cv.resize(frame[:1080, :1080], (local_game.Height, local_game.Width))
        cv.imshow("Phone Camera", temp)
        count += 1
        count %= 30
        # R键重启游戏
        if local_game.game_loop():
            local_game.win.after(0, lambda: printpic(local_game))
        else:
            with keyboard.Events() as events:
                for event in events:
                    if event.key.char == 'r':
                        break
            printpic(local_game)

    else:
        return


printpic(game)
game.win.mainloop()
cap.release()
cv.destroyAllWindows()
