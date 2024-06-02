import os.path

import torch
import numpy as np
from model import LinearQnet, QTrainer
from collections import deque
import random
from game import SnakeGameAI

LR = 0.001
MEMORY_SIZE = 100_1000
BATCH_SIZE = 100


class Agent:
    def __init__(self):
        self.model = LinearQnet(32, 256, 4)
        self.gama = 0.9
        self.epsilon = 0
        self.n_games = 0
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.trainer = QTrainer(self.model, LR, self.gama)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        # final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
            move = torch.tensor(move, dtype=torch.float32)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            move = self.model(state0)
            # move = torch.argmax(prediction).item()
            # final_move[move] = 1

        return move

    def get_state(self, game):
        state = SnakeGameAI.returnFeature(game)
        return np.array(state)

    def remember(self, state, action, reward, next_state, is_done):
        self.memory.append((state, action, reward, next_state, is_done))

    def train_short_memory(self, state, action, reward, next_state, is_done):
        self.trainer.train_step(state, action, reward, next_state, is_done)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, is_dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, is_dones)


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    # game = SnakeGame(20,20)
    game = SnakeGameAI(20,20,Fps=100)
    if os.path.exists(r'./model/model.pth'):
        agent.model.load_model()
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        cur_dir = [game.snake_list[1][0] - game.snake_list[0][0],
                   game.snake_list[1][1] - game.snake_list[0][1]]  # the oppsite dirc of the current dirc
        cur_dir_reverse = 1.0 - torch.tensor(game.dir_dict[str(cur_dir)], dtype=torch.float32)
        del_idx = torch.argmin(cur_dir_reverse).tolist()
        sort_idx = np.argsort(-final_move.detach()).tolist()
        max_i = None
        for i in sort_idx:
            if i != del_idx:
                max_i = i
                break
        if max_i is None:
            max_i = (sort_idx[0] + 1) % 4
        temp_dir = [0.0, 0.0, 0.0, 0.0]
        temp_dir[max_i] = 1.0
        reward, is_done, score = game.play_step(game.dir_dict[str(temp_dir)])
        state_next = agent.get_state(game)
        agent.train_short_memory(state_old, temp_dir, reward, state_next, is_done)
        agent.remember(state_old, temp_dir, reward, state_next, is_done)
        # game.win.after(game.Fps)

        if is_done:
            agent.n_games += 1
            game.Restart_game()
            agent.train_long_memory()
            if score > record:
                record = score
                agent.model.save_model()
            print('Game', agent.n_games, 'Score', score, 'Record:', record)
            total_score += score
            mean_scores = total_score / agent.n_games
            plot_mean_scores.append(mean_scores)


if __name__ == '__main__':
    train()
