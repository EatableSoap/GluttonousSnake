import os.path
import pickle

import torch
import numpy as np
from model import LinearQnet, QTrainer
from collections import deque
import random
from game import SnakeGameAI
import math

LR = 0.001
MEMORY_SIZE = 100_1000
BATCH_SIZE = 100


class Agent:
    def __init__(self):
        self.model = LinearQnet(32, 256, 4)
        self.gama = 0.9
        self.epsilon = 0.9
        self.n_games = 0
        self.record = 0
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.trainer = QTrainer(self.model, LR, self.gama)

    def get_action(self, state):
        # self.epsilon = 80 - self.n_games
        # final_move = [0, 0, 0]
        if random.random() < self.epsilon:
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


def train(row, col, num):
    score_list = []
    total_score = 0
    max_mean_scores = 0
    agent = Agent()
    # game = SnakeGame(20,20)
    game = SnakeGameAI(row, col, Fps=100)
    if os.path.exists(r'model/model_best.pkl'):
        with open(r'model/model_best.pkl', 'rb') as f:
            try:
                agent = pickle.load(f)
            except EOFError:
                pass
    while agent.n_games < num:
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
        reward, is_done, score = game.play_step(game.dir_dict[str(temp_dir)], 10 + math.ceil(game.Score / 100),
                                                1 + game.Score / 100)
        state_next = agent.get_state(game)
        agent.train_short_memory(state_old, temp_dir, reward, state_next, is_done)
        agent.remember(state_old, temp_dir, reward, state_next, is_done)
        # game.win.after(game.Fps)

        if is_done:
            agent.n_games += 1
            game.Restart_game()
            agent.train_long_memory()
            if score > agent.record:
                agent.record = score
                with open(r'model/model_best.pkl', 'wb') as f:
                    pickle.dump(agent, f)
            total_score += score
            if agent.n_games % 20 == 0:
                agent.epsilon = max(0.05, agent.epsilon * 0.95)
                mean_scores = total_score / 20
                if mean_scores > max_mean_scores:
                    max_mean_scores = mean_scores
                    with open(r'model/model_best.pkl', 'wb') as f:
                        pickle.dump(agent, f)
                total_score = 0
                # plot_mean_scores.append(mean_scores)
                score_list.append([agent.n_games, agent.record, mean_scores])
                print('Game', agent.n_games, '\t', 'Record:', agent.record, '\t'
                      , 'Mean Score', mean_scores)
    return score_list


if __name__ == '__main__':
    game_score = train(10, 10, 2000)
    with open(r'model/train_data.pkl', 'wb') as trainData:
        pickle.dump(game_score, trainData)
