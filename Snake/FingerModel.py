import numpy as np
import cv2 as cv
import pickle
from sklearn.svm import SVR

path_name = r'D:/AI_project/dataset'


class FingerModel:
    def __init__(self):
        self.clf = SVR()
        self.x_data = np.zeros(1)
        self.y_data = np.zeros(1)

    # YCrCb颜色空间的Cr分量+Otsu阈值分割获取二值图
    def cr_otsu(self, image):
        """
        :param image: 图片路径
        :return: None
        """
        img = cv.imread(image, cv.IMREAD_COLOR)
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

        # dst = cv.bitwise_and(img, img, mask=skin)
        # cv.namedWindow("seperate", cv.WINDOW_NORMAL)
        # cv.imshow("seperate", dst)
        # cv.waitKey()
        return skin

    def train(self,path):
        self.clf = SVR(kernel='linear')
        self.clf.fit(self.x_data,self.y_data.T.ravel())
        with open(path+r'/clf.pkl', 'wb') as f:
            pickle.dump(self.clf, f)

    def load(self,path):
        with open(path+r'/clf.pkl', 'rb') as f:
            self.clf = pickle.load(f)

    def ReadData(self, path):
        with open(path + r'/dataset.pkl', 'rb') as f:
            datas = pickle.load(f)
            x_data = []
            y_data = []
            for data in datas:
                y_data.append(data[0])
                x_data.append(data[1])
            self.y_data = np.array(y_data).reshape(len(datas),-1)
            self.x_data = np.array(x_data).reshape(len(datas),-1)


a = FingerModel()
a.ReadData(path_name)
try:
    a.load(path_name)
except:
    print('No Model Load')
a.train(path_name)
