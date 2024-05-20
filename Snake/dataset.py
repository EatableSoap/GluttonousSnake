import os
import cvzone
import cv2
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector  # 导入手部检测模块

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75)

detector = HandDetector(maxHands=1,  # 最多检测1只手
                        detectionCon=0.8)  # 最小检测置信度0.8

path_name = r'D:/AI_project/dataset/'
# cap = cv2.VideoCapture('D:\AI_project\dataset')
count = 0
for img_name in os.listdir(path_name):
    temp_name=img_name.split('-')[0]
    count += 1
    img = cv2.imread(path_name + img_name)

    # 图像翻转，使图像和自己呈镜像关系
    img = cv2.flip(img, 1)  # 0代表上下翻转，1代表左右翻转

    # 检测手部关键点。返回手部信息hands，绘制关键点后的图像img
    hands, img = detector.findHands(img, flipType=False)  # 由于上一行翻转过图像了，这里就不用翻转了

    # （4）关键点处理
    if hands:  # 如果检测到手了，那就处理关键点

        # 获得食指指尖坐标(x,y)
        hand = hands[0]  # 获取一只手的全部信息
        lmList = hand['lmList']  # 获得这只手的21个关键点的坐标(x,y,z)
        pointIndex = lmList[8][0:2]  # 只获取食指指尖关键点的（x,y）坐标
        os.rename(path_name + img_name,
                  path_name + temp_name.replace('.jpeg','') + '-' + str(pointIndex[0]) + 'x' + str(pointIndex[1]) + '.jpeg')
        # 以食指指尖为圆心画圈（圆心坐标是元组类型），半径为15，青色填充
        cv2.circle(img, tuple(pointIndex), 15, (255, 0, 0), cv2.FILLED)
    # （5）显示图像
    cv2.imshow('img', img)  # 输入图像显示窗口的名称及图像
    # 每帧滞留1毫秒后消失，并且按下ESC键退出

    if cv2.waitKey(1) & 0xFF == 27:
        break
