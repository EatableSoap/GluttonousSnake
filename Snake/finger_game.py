import cv2 as cv
import numpy as np
from pynput import keyboard
from snake_class import Snake
from FingerModel import FingerModel

path_name = r'D:/AI_project/dataset'
ip = "10.198.181.67"
port = "8080"
url = "http://" + ip + ":" + port + "/video"

cap = cv.VideoCapture(url)
game = Snake(Fps=100, row=52, column=54, Unit_size=10)
control = keyboard.Controller()

Model = FingerModel()
Model.load(path_name)

last_frame = np.zeros((108, 108))


# pos = game.set_in_windows()
def printpic(local_game):
    global last_frame
    ret, frame = cap.read()
    img = frame
    # cv.moveWindow("", int(pos[0]), int(pos[1]))
    if ret:
        # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        cv.rectangle(frame, (0, 0), (1080, 1080), color=(0, 255, 0))
        img = Model.cr_otsu(img[:1080, :1080])
        img = cv.resize(img, (108, 108))*0.95+last_frame*0.05
        last_frame = img
        # hull=Model.FindFingerPos(img)
        # img = cv.Canny(img, threshold1=0, threshold2=100)  # 八领域法
        # cv.imshow('0', img)
        # img = cv.resize(img,(108, 108))
        img = Model.Hog_ft(img)
        pos = Model.clf.predict(img.reshape(1, -1))
        pos_y = int(pos[0][1] * 10)
        pos_x = int(pos[0][0] * 10)

        # print(pos_x, pos_y)
        # cv.circle(frame, (pos_x, pos_y), 10, (0, 0, 255), thickness=-1)
        cv.imshow("Phone Camera", frame)
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
