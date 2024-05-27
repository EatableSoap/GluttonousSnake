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
        cr1 = cv.GaussianBlur(cr, (3, 3), 0)
        _, skin = cv.threshold(cr1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
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
        # cv.namedWindow('emm', 0)
        # cv.resizeWindow('emm', 720, 720)
        Rimg = img.copy()
        angle = random.uniform(0, 360.0)
        angle_arc = angle * math.pi / 180.0
        scl = random.uniform(0.5, 0.9)
        # cv.circle(Rimg, (int(column), int(row)), 5, (255, 0, 0), -1)
        rows, cols = img.shape
        # 获取中心旋转矩阵
        M = cv.getRotationMatrix2D((540, 540), angle, scl)
        rotated = cv.warpAffine(Rimg, M, (rows, cols))

        # 获取旋转后标签坐标
        RRow = ((row - 540.0) * math.cos(angle_arc) - (column - 540.0) * math.sin(angle_arc)) * scl + 540.0
        RColumn = ((column - 540.0) * math.cos(angle_arc) + (row - 540.0) * math.sin(angle_arc)) * scl + 540.0

        # 约束坐标点范围
        RRow = min(max(0, RRow), 1080)
        RColumn = min(max(0, RColumn), 1080)

        # cv.circle(rotated, (int(RColumn), int(RRow)), 5, (0, 0, 255), -1)
        # cv.imshow('emm', rotated)
        # cv.waitKey(0)
        return rotated, RColumn, RRow

    # hog特征提取
    def Hog_ft(self, img_arr):
        img_arr = cv.resize(img_arr, (108, 108))
        features = ft.hog(img_arr, pixels_per_cell=(16, 16), cells_per_block=(4, 4), visualize=False)
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
                # 数据增广操作
                for i in range(20):
                    img_temp, column_temp, row_temp = self.rotationIMG(img, float(column), float(row))
                    features2 = self.Hog_ft(img_temp)
                    temp2 = [np.array([float(column_temp), float(row_temp)], dtype='float32'), features2]
                    img_label.append(temp2)
                    # cv.circle(img_temp, (int(column_temp / 10), int(row_temp / 10)), 1, (0, 0, 255), -1)
                    # cv.imshow('0', img_temp)
                    # cv.waitKey(0)
                features = self.Hog_ft(img)
                temp = [np.array([float(column), float(row)], dtype='float32'), features]
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
