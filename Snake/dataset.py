import copy
import math
import os
import pickle
import random

import cv2 as cv
import numpy as np
from skimage import feature as ft


class DateSet:
    def __init__(self):
        self.path = r'D:/AI_project/dataset4'
        return

    def cr_otsu(self, image):
        """YCrCb颜色空间的Cr分量+Otsu阈值分割
        :param image: 图片路径
        :return: None
        """
        # img = cv.imread(image, cv.IMREAD_COLOR)
        img = cv.resize(image, (108, 108))
        ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)

        (y, cr, cb) = cv.split(ycrcb)
        cr1 = cv.GaussianBlur(cr, (5, 5), 0)
        _, skin = cv.threshold(cr1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # cv.namedWindow("image raw", cv.WINDOW_NORMAL)
        # cv.imshow("image raw", img)
        # cv.namedWindow("image CR", cv.WINDOW_NORMAL)
        # cv.imshow("image CR", cr1)
        # cv.namedWindow("Skin Cr+OTSU", cv.WINDOW_NORMAL)
        # cv.imshow("Skin Cr+OTSU", skin)
        #
        # dst = cv.bitwise_and(img, img, mask=skin)
        # cv.namedWindow("seperate", cv.WINDOW_NORMAL)
        # cv.imshow("seperate", dst)
        # cv.waitKey()
        return skin

    def SetPoints(self, windowname, img):
        """
        输入图片，打开该图片进行标记点，返回的是标记的几个点的字符串
        """
        print('(提示：单击需要标记的坐标，Enter确定，Esc跳过，其它重试。)')
        points = []

        def onMouse(event, x, y, a, b):
            if event == cv.EVENT_LBUTTONDOWN:
                cv.circle(temp_img, (x, y), 10, (102, 217, 239), -1)
                points.append([x, y])
                cv.imshow(windowname, temp_img)

        temp_img = img.copy()
        cv.namedWindow(windowname)
        cv.imshow(windowname, temp_img)
        cv.setMouseCallback(windowname, onMouse)
        key = cv.waitKey(0)
        if key == 13:  # Enter
            print('坐标为：', points)
            del temp_img
            cv.destroyAllWindows()
            return str(points)
        elif key == 27:  # ESC
            print('跳过该张图片')
            del temp_img
            cv.destroyAllWindows()
            return
        else:
            print('重试!')
            return self.SetPoints(windowname, img)

    # # 使用opencv为图片打标签
    # mp_drawing = mp.solutions.drawing_utils
    # mp_hands = mp.solutions.hands
    # hands = mp_hands.Hands(
    #     static_image_mode=False,
    #     max_num_hands=2,
    #     min_detection_confidence=0.75,
    #     min_tracking_confidence=0.75)
    #
    # detector = HandDetector(maxHands=1,  # 最多检测1只手
    #                         detectionCon=0.8)  # 最小检测置信度0.8
    #
    # # cap = cv2.VideoCapture('D:\AI_project\dataset')
    # count = 0
    # for img_name in os.listdir(path_name):
    #     temp_name=img_name.split('-')[0]
    #     count += 1
    #     img = cv2.imread(path_name + img_name)
    #
    #     # 图像翻转，使图像和自己呈镜像关系
    #     img = cv2.flip(img, 1)  # 0代表上下翻转，1代表左右翻转
    #
    #     # 检测手部关键点。返回手部信息hands，绘制关键点后的图像img
    #     hands, img = detector.findHands(img, flipType=False)  # 由于上一行翻转过图像了，这里就不用翻转了
    #
    #     # （4）关键点处理
    #     if hands:  # 如果检测到手了，那就处理关键点
    #
    #         # 获得食指指尖坐标(x,y)
    #         hand = hands[0]  # 获取一只手的全部信息
    #         lmList = hand['lmList']  # 获得这只手的21个关键点的坐标(x,y,z)
    #         pointIndex = lmList[8][0:2]  # 只获取食指指尖关键点的（x,y）坐标
    #         os.rename(path_name + img_name,
    #                   path_name + temp_name.replace('.jpeg','') +
    #                   '-' + str(pointIndex[0]) + 'x' + str(pointIndex[1]) + '.jpeg')
    #         # 以食指指尖为圆心画圈（圆心坐标是元组类型），半径为15，青色填充
    #         cv2.circle(img, tuple(pointIndex), 15, (255, 0, 0), cv2.FILLED)
    #     # （5）显示图像
    #     cv2.imshow('img', img)  # 输入图像显示窗口的名称及图像
    #     # 每帧滞留1毫秒后消失，并且按下ESC键退出
    #
    #     if cv2.waitKey(30) & 0xFF == 27:
    #         break

    # 标注数据点
    def Point_IMG(self, path_name):
        for img in os.listdir(path_name):
            cv.namedWindow('Point', 0)
            cv.resizeWindow('Point', 720, 720)
            temp_name = img.replace('.jpeg', '').split('-')[0]
            pos = self.SetPoints('Point', cv.imread(path_name + '/' + img))
            os.rename(path_name + '/' + img,
                      path_name + '/' + temp_name.replace('.jpeg', '') +
                      '-' + pos.replace('[', '').replace(']', '') + '.jpeg')

    # 图像二值化
    def Binary_IMG(self, path_name):
        for img in os.listdir(path_name):
            img_arr = self.cr_otsu(path_name + '/' + img)
            # 显示图像
            # img_arr = cv.imread(path_name+'/'+img)
            # img_arr = cv.Canny(img_arr, threshold1=30, threshold2=180)  # 八领域法
            cv.imwrite(path_name + '/' + img.replace('.jpeg', '') + '_binary.jpeg', img_arr)
            # cv.imshow("title", img_arr)
            # 进程不结束，一直保持显示状态
            # cv.waitKey(0)
            # 销毁所有窗口
            # cv.destroyAllWindows()

    def rotationIMG(self, img, column, row):
        angle = random.uniform(-180.0, 180.0)
        Rimg = copy.copy(img)
        # cv.circle(Rimg,(int(column/10),int(row/10)),5,(0,0,255),-1)
        # cv.imshow('1',Rimg)
        # cv.waitKey(0)
        RColumn = copy.copy(column)
        RRow = copy.copy(row)
        rows, cols = Rimg.shape
        M = cv.getRotationMatrix2D((cols / 2.0, rows / 2.0), angle, 1)
        rotated = cv.warpAffine(Rimg, M, (cols, rows))
        RRow = (RRow - 540) * math.cos(angle) - (RColumn - 540) * math.sin(angle) + 540
        RColumn = (RRow - 540) * math.sin(angle) + (RColumn - 540) * math.cos(angle) + 540
        RRow = min(max(0, RRow), 1080)
        RColumn = min(max(0, RColumn), 1080)
        # cv.circle(rotated,(int(RColumn/10),int(RRow/10)),1,(0,0,255),-1)
        # cv.imshow('0',rotated)
        # cv.waitKey(0)
        return rotated, RColumn, RRow

    def Hog_ft(self, img_arr):
        # for img in os.listdir(path_name):
        # img_arr = self.cr_otsu(path_name + '/' + img)
        # img_arr = cv.imread(path_name + '/' + img)
        img_arr = cv.resize(img_arr, (108, 108))
        # img_arr = cv.cvtColor(img_arr, cv.COLOR_BGR2GRAY)
        features = ft.hog(img_arr, pixels_per_cell=(16, 16), cells_per_block=(4, 4), visualize=False)
        # cv.imshow('0', features[1])
        # cv.waitKey(0)
        # cv.imwrite(path_name + '/' + img.replace('.jpeg', '') + '_hog.png', features[1])
        return features

    # 打包数据集
    def PackData(self, path_name):
        with open(r'D:\AI_project\dataset\dataset.pkl', 'wb') as f:
        # with open(r'D:\AI_project\dataset\dataset_test.pkl', 'wb') as f:
            img_label = []
            for img in os.listdir(path_name):
                column, row = img.replace('.jpeg', '').split('-')[1].split(',')[0:2]
                img = cv.imread(path_name + '/' + img)
                img = self.cr_otsu(img)
                # for i in range(2):
                #     img2, column2, row2 = self.rotationIMG(img, float(column), float(row))
                #     features2 = self.Hog_ft(img2)
                #     temp2 = [np.array([float(column) / 10, float(row) / 10], dtype='float32'), features2]
                #     img_label.append(temp2)
                features = self.Hog_ft(img)
                temp = [np.array([float(column) / 10, float(row) / 10], dtype='float32'), features]
                img_label.append(temp)
                # clbp = local_binary_pattern(cv.imread(path_name+'/'+img), 8, 1, method="ror")
                # img_array = cv.imread(path_name + '/' + img, -1).astype('float32').flatten()
            pickle.dump(img_label, f)


if __name__ == '__main__':
    Processor = DateSet()
    # Processor.Point_IMG(Processor.path)
    # Processor.Binary_IMG(path)
    # Processor.Hog_ft(path)
    Processor.PackData(Processor.path)
    # data=open(r'./dataset.pkl','rb')
    # a=pickle.load(data)
    # cv.waitKey(0)
