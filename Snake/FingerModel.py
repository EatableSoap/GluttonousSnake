import numpy as np
import cv2 as cv
import pickle
from sklearn.svm import LinearSVR
from sklearn.multioutput import RegressorChain
from dataset import DateSet


class FingerModel(DateSet):
    def __init__(self, path=r'D:/AI_project/dataset'):
        super().__init__()
        model = LinearSVR(C=3, dual=False, loss='squared_epsilon_insensitive', max_iter=10000000)
        # model=DecisionTreeRegressor(max_depth=1000000)
        self.clf1 = RegressorChain(model, order=[1, 0])
        self.clf2 = RegressorChain(model, order=[0, 1])
        self.clf = [self.clf1, self.clf2]
        # self.clf = MultiOutputRegressor(model)
        self.x_data = np.zeros(1)
        self.y_data = np.zeros(1)
        self.x_test = np.zeros(1)
        self.y_test = np.zeros(1)
        self.path = path

    # 寻找凸包
    # def FindFingerPos(self, img):
    #     contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #     hull = cv.convexHull(contours[0], returnPoints=False)
    #     return hull

    def train(self, path):
        # self.clf = RandomForestRegressor()
        self.clf[0].fit(self.x_data, self.y_data)
        self.clf[1].fit(self.x_data, self.y_data)
        # self.clf.fit(self.x_data,self.y_data)
        with open(path + r'/clf.pkl', 'wb') as f:
            pickle.dump(self.clf, f)

    def load(self, path):
        with open(path + r'/clf.pkl', 'rb') as f:
            self.clf = pickle.load(f)

    # 载入数据集
    def ReadData(self, path):
        with open(path + r'/dataset.pkl', 'rb') as f:
            datas = pickle.load(f)
            x_data = []
            y_data = []
            for data in datas:
                y_data.append(data[0])
                x_data.append(data[1])
            self.y_data = np.array(y_data).reshape(len(datas), -1)
            self.x_data = np.array(x_data).reshape(len(datas), -1)


if __name__ == '__main__':
    # path_name = r'D:/AI_project/dataset'
    path_name = r'./dataset'
    a = FingerModel()
    a.ReadData(path_name)
    # Processor = DateSet()
    # try:
    #     a.load(path_name)
    #     print('Load Model')
    # except:
    #     print('No Model Load')
    # Processor.PackData(Processor.path)
    a.train(path_name)
    cv.waitKey(0)
