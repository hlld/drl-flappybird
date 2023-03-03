import argparse
import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from copy import deepcopy
from flappybird import FlappyBird


class DQNNet(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(DQNNet, self).__init__()
        self.stem = nn.Sequential(*[
            self.make_conv_block(in_channels, 32, kernel_size=7, stride=4),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self.make_conv_block(32, 64, kernel_size=5, stride=2),
            self.make_conv_block(64, 64, kernel_size=3, stride=1),
        ])
        self.head = nn.Sequential(*[
            nn.Linear(1600, 512),
            nn.Linear(512, num_actions),
        ])

    def forward(self, x):
        x = self.stem(x)
        x = x.flatten(start_dim=1, end_dim=-1)
        return self.head(x)

    @staticmethod
    def make_conv_block(in_channels,
                        out_channels,
                        kernel_size,
                        stride,
                        padding=None):
        padding = kernel_size // 2 if padding is None else padding
        return nn.Sequential(*[
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding),
            nn.ReLU(inplace=True),
        ])


class Trainer(object):
    def __init__(self, **kargs):
        self.device = torch.device(kargs['device'])
        model = DQNNet(4, kargs['num_actions'])
        self.model = model.to(self.device).train()
        self.optimizer = optim.Adam(model.parameters(), lr=kargs['learning_rate'])
        self.game = FlappyBird(kargs['media_path'])
        self.attrs = kargs

    @staticmethod
    def preprocess(image, output_size=(80, 80)):
        image = cv2.resize(image, output_size)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray = cv2.threshold(image_gray, 1, 255, cv2.THRESH_BINARY)[1]
        image_gray = image_gray.reshape(output_size[0], output_size[1], 1)
        return image_gray.transpose((2, 0, 1))[None, :, :, :]

    def train_network(self):
        do_nothing = np.zeros(self.attrs['num_actions'])
        do_nothing[0] = 1
        image_raw, reward_t, terminal = self.game.frame_step(do_nothing)
        state_t = np.tile(self.preprocess(image_raw), (1, 4, 1, 1))
        replay_memory = deque()
        epsilon = self.attrs['initial_epsilon']
        time_steps = 0

        total_steps = self.attrs['observe_steps'] + self.attrs['explore_steps']
        while time_steps < total_steps:
            action_t = np.zeros(self.attrs['num_actions'])
            if time_steps % self.attrs['frame_per_action'] == 0:
                if random.random() <= epsilon:
                    action_index = random.randrange(self.attrs['num_actions'])
                    action_t[action_index] = 1
                else:
                    inputs = torch.from_numpy(state_t).to(self.device, non_blocking=True).float()
                    with torch.no_grad():
                        outputs = self.model.eval()(inputs).cpu().numpy()
                    action_index = np.argmax(outputs[0])
                    action_t[action_index] = 1
            else:
                action_index = 0
                action_t[action_index] = 1

            if epsilon > self.attrs['final_epsilon'] and time_steps > self.attrs['observe_steps']:
                epsilon_range = self.attrs['initial_epsilon'] - self.attrs['final_epsilon']
                epsilon -= epsilon_range / self.attrs['explore_steps']

            image_raw, reward_t, terminal = self.game.frame_step(action_t)
            state_t1 = np.concatenate([self.preprocess(image_raw), state_t[:, :3, :, :]], axis=1)
            replay_memory.append((state_t, action_t, reward_t, state_t1, terminal))
            if len(replay_memory) > self.attrs['replay_memory_size']:
                replay_memory.popleft()

            if time_steps > self.attrs['observe_steps']:
                mini_batch = random.sample(replay_memory, self.attrs['batch_size'])
                state_t_batch = np.concatenate([x[0] for x in mini_batch], axis=0)
                action_t_batch = np.concatenate([x[1][None, :] for x in mini_batch], axis=0)
                reward_t_batch = [x[2] for x in mini_batch]
                state_t1_batch = np.concatenate([x[3] for x in mini_batch], axis=0)

                inputs = torch.from_numpy(state_t1_batch).to(self.device, non_blocking=True).float()
                with torch.no_grad():
                    outputs = self.model.eval()(inputs).cpu().numpy()
                true_batch = []
                for batch_index in range(len(mini_batch)):
                    terminal = mini_batch[batch_index][4]
                    if terminal:
                        true_batch.append(reward_t_batch[batch_index])
                    else:
                        increment = self.attrs['decay_rate_gamma'] * np.max(outputs[batch_index])
                        true_batch.append(reward_t_batch[batch_index] + increment)

                inputs = torch.from_numpy(state_t_batch).to(self.device, non_blocking=True).float()
                true_batch = torch.from_numpy(np.array(true_batch)).to(self.device).float()
                action_t_batch = torch.from_numpy(action_t_batch).to(self.device).float()
                outputs = self.model.train()(inputs)
                loss = (true_batch - (outputs * action_t_batch).sum(dim=1)).square().mean()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            state_t = state_t1
            time_steps += 1
            if time_steps % 10000 == 0:
                ckpt_path = os.path.join(self.attrs['output_path'], 'dqnnet.pt')
                saved_ckpt = {'model': deepcopy(self.model).half(),
                              'optimizer': self.optimizer.state_dict()}
                torch.save(saved_ckpt, ckpt_path)
            print('TIMESTEP:', time_steps, 'EPSILON:', epsilon,
                  'ACTION:', action_index, 'REWARD:', reward_t)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='./model',
                        help='path to save outputs')
    parser.add_argument('--media_path', type=str, default='./media',
                        help='media path')
    parser.add_argument('--learning_rate', type=float, default=1e-6,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='torch device')
    parser.add_argument('--num_actions', type=int, default=2,
                        help='number of actions')
    parser.add_argument('--decay_rate_gamma', type=float, default=0.99,
                        help='decay rate gamma')
    parser.add_argument('--observe_steps', type=int, default=100000,
                        help='steps of observe stage')
    parser.add_argument('--explore_steps', type=int, default=2000000,
                        help='steps of observe stage')
    parser.add_argument('--initial_epsilon', type=float, default=0.1,
                        help='initial epsilon')
    parser.add_argument('--final_epsilon', type=float, default=0.001,
                        help='final epsilon')
    parser.add_argument('--replay_memory_size', type=int, default=50000,
                        help='size of replay memory')
    parser.add_argument('--frame_per_action', type=int, default=1,
                        help='frame per action')
    opt = parser.parse_args()
    Trainer(**vars(opt)).train_network()


if __name__ == '__main__':
    main()
