import torch
import torch.optim as optim
import torch.nn as nn
import os


class LinearQnet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

    def save_model(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.mkdir(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load_model(self, file_name='model.pth'):
        model_folder_path = './model/'
        if not os.path.exists(model_folder_path):
            return
        file_name = os.path.join(model_folder_path, file_name)
        self.load_state_dict(torch.load(file_name))


class QTrainer:
    def __init__(self, model, lr, gama):
        self.model = model
        self.lr = lr
        self.gama = gama
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.creterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, is_done):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.long)
        next_state = torch.tensor(next_state, dtype=torch.float)

        if len(state.shape) == 1:
            is_done = (is_done,)
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)

        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(is_done)):
            Q_new = reward[idx]
            if not is_done:
                Q_new = Q_new + self.gama * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action).item()] = Q_new
        self.optimizer.zero_grad()
        loss = self.creterion(target, pred)
        loss.backward()
        self.optimizer.step()
