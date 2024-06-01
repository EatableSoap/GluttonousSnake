import torch


class SnakeNet(torch.nn.Module):
    def __init__(self, in_feature=32, hidden1_featute=20, hidden2_feature=12, out_feature=4):
        super(SnakeNet, self).__init__()
        self.inft = in_feature
        self.h1 = hidden1_featute
        self.h2 = hidden2_feature
        self.out = out_feature

        self.Input = torch.nn.Linear(in_feature, hidden1_featute)
        self.HiddenLayer = torch.nn.Linear(hidden1_featute, hidden2_feature)
        self.Output = torch.nn.Linear(hidden2_feature, out_feature)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    # 基因为一个tensor，需要将其按一定格式读入Model中
    def setweight(self, weights):
        weights = torch.tensor(weights, dtype=torch.float32)
        with torch.no_grad():
            in2h1 = self.inft * self.h1
            in2h1bias = in2h1 + self.h1
            h12h2 = in2h1bias + self.h1 * self.h2
            h12h2bias = h12h2 + self.h2
            h22out = h12h2bias + self.h2 * self.out
            self.Input.weight.data = weights[0:in2h1].reshape(self.h1, self.inft)
            self.Input.bias.data = weights[in2h1:in2h1bias]
            self.HiddenLayer.weight.data = weights[in2h1bias:h12h2].reshape(self.h2, self.h1)
            self.HiddenLayer.bias.data = weights[h12h2:h12h2bias]
            self.Output.weight.data = weights[h12h2bias:h22out].reshape(self.out, self.h2)
            self.Output.bias.data = weights[h22out:]

    def forward(self, x):
        y = self.Input(x)
        y = self.relu(y)
        y = self.HiddenLayer(y)
        y = self.relu(y)
        y = self.Output(y)
        y = self.sigmoid(y)
        return y

    def predic(self, in_features):
        y = self(in_features)
        return y
