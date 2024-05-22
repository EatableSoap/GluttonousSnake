import numpy as np
import cv2 as cv
import pickle
from sklearn.svm import SVR
from skimage import feature as ft
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import RegressorChain

path_name = r'D:/AI_project/dataset'


class FingerModel:
    def __init__(self, path=r'D:/AI_project/dataset'):
        model = SVR(C=1.5,kernel='linear')
        self.clf = RegressorChain(model)
        self.x_data = np.zeros(1)
        self.y_data = np.zeros(1)
        self.path = path

    # YCrCb颜色空间的Cr分量+Otsu阈值分割获取二值图
    def cr_otsu(self, image):
        """
        :param image: 图片路径
        :return: None
        """
        # img = cv.imread(image, cv.IMREAD_COLOR)
        ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCR_CB)

        (y, cr, cb) = cv.split(ycrcb)
        cr1 = cv.GaussianBlur(cr, (5, 5), 0)
        _, skin = cv.threshold(cr1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # cv.namedWindow("image raw", cv.WINDOW_NORMAL)
        # cv.imshow("image raw", img)
        # cv.namedWindow("image CR", cv.WINDOW_NORMAL)
        # cv.imshow("image CR", cr1)
        # cv.namedWindow("Skin Cr+OTSU", cv.WINDOW_NORMAL)
        # cv.imshow("Skin Cr+OTSU", skin)

        # dst = cv.bitwise_and(img, img, mask=skin)
        # cv.namedWindow("seperate", cv.WINDOW_NORMAL)
        # cv.imshow("seperate", dst)
        # cv.waitKey()
        return skin

    def Hog_ft(self, img_arr):
        # for img in os.listdir(path_name):
        # img_arr = self.cr_otsu(path_name + '/' + img)
        # img_arr = cv.imread(path_name + '/' + img)
        img_arr = cv.resize(img_arr, (108, 108))
        # img_arr = cv.cvtColor(img_arr, cv.COLOR_BGR2GRAY)
        features = ft.hog(img_arr, pixels_per_cell=(8, 8), cells_per_block=(4, 4), visualize=False)
        # cv.imshow('0', features[1])
        # cv.waitKey(0)
        # cv.imwrite(path_name + '/' + img.replace('.jpeg', '') + '_hog.png', features[1])
        return features

    def FindFingerPos(self, img):
        contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        hull = cv.convexHull(contours[0], returnPoints=False)
        # cv.polylines(img, [hull], True, (255, 255, 0), 2)
        return hull

    def train(self, path):
        # self.clf = RandomForestRegressor()
        self.clf.fit(self.x_data, self.y_data)
        with open(path + r'/clf.pkl', 'wb') as f:
            pickle.dump(self.clf, f)

    def load(self, path):
        with open(path + r'/clf.pkl', 'rb') as f:
            self.clf = pickle.load(f)

    def ReadData(self, path):
        with open(path + r'/dataset_bi_hog.pkl', 'rb') as f:
            datas = pickle.load(f)
            x_data = []
            y_data = []
            for data in datas:
                y_data.append(data[0])
                x_data.append(data[1])
            self.y_data = np.array(y_data).reshape(len(datas), -1)
            self.x_data = np.array(x_data).reshape(len(datas), -1)


# a = FingerModel()
# a.ReadData(path_name)
# # try:
# #     a.load(path_name)
# # except:
# #     print('No Model Load')
# a.train(path_name)
