import cv2 as cv
from pynput import keyboard
from snake_class import Snake

ip = "10.198.146.150"
port = "4747"
url = "http://" + ip + ":" + port + "/video"

cap = cv.VideoCapture(url)
game = Snake(Fps=100)
control = keyboard.Controller()


# pos = game.set_in_windows()
def printpic(local_game):
    ret, frame = cap.read()
    img = frame
    # cv.moveWindow("", int(pos[0]), int(pos[1]))
    if ret:
        img = cv.resize(img, (local_game.Width, local_game.Height), interpolation=cv.INTER_NEAREST)
        # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        cv.imshow("Phone Camera", img)
        if local_game.game_loop():
            local_game.win.after(0, lambda: printpic(local_game))
        else:
            with keyboard.Events() as events:
                for event in events:
                    if event.key.char=='r':
                        break
            printpic(local_game)

    else:
        return


printpic(game)
game.win.mainloop()
cap.release()
cv.destroyAllWindows()
