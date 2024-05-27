import torch
import GluttonousSnake.Snake.SnakeClass


class SnakeNet(torch.nn.Module):
    def __init__(self, in_features=None):
        super(SnakeNet, self).__init__()
        self.in_features = in_features
        self.Input = torch.nn.Linear(32, 20)
        self.HiddenLayer = torch.nn.Linear(20, 12)
        self.Output = torch.nn.Linear(12, 4)
        self.relu = torch.nn.ReLU
        self.sigmoid = torch.nn.Sigmoid

    def forward(self):
        x = self.Input(self.in_features)
        x = self.relu(x)
        x = self.HiddenLayer(x)
        x = self.sigmoid(x)
        output = self.Output(x)
        return output


class Individual:
    def __init__(self, path=None):
        self.genes = SnakeNet()
        if path is not None:
            state_dict = torch.load(path)
            self.genes.load_state_dict(state_dict)

