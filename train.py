import argparse
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from flappybird import FlappyBird


class DQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(DQN, self).__init__()
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
        model = DQN(4, kargs['num_actions'])
        self.model = model.to(self.device).train()
        self.optimizer = optim.Adam(model.parameters(), lr=kargs['learning_rate'])
        self.game = FlappyBird(kargs['media_path'])
        self.kargs = kargs

    @staticmethod
    def convert_image(image, output_size=(80, 80)):
        image = cv2.resize(image, output_size)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray = cv2.threshold(image_gray, 1, 255, cv2.THRESH_BINARY)[1]
        image_gray = image_gray.reshape(output_size[0], output_size[1], 1)
        return image_gray.transpose((2, 0, 1))[None, :, :, :]

    def train_network(self):
        do_nothing = np.zeros(self.kargs['num_actions'])
        do_nothing[0] = 1
        image_raw, reward_t, terminal = self.game.frame_step(do_nothing)
        image_gray = self.convert_image(image_raw)
        state_t = np.tile(image_gray, (1, 4, 1, 1))
        replay_memory = deque()
        epsilon = self.kargs['initial_epsilon']
        time_steps = 0

        total_steps = self.kargs['observe_steps'] + self.kargs['explore_steps']
        while time_steps < total_steps:
            inputs = torch.from_numpy(state_t).to(self.device, non_blocking=True).float()
            with torch.no_grad():
                outputs = self.model(inputs).cpu().numpy()
            action_t = np.zeros(self.kargs['num_actions'])
            if time_steps % self.kargs['frame_per_action']:
                if random.random() <= epsilon:
                    action_index = random.randrange(self.kargs['num_actions'])
                    action_t[action_index] = 1
                else:
                    action_index = np.argmax(outputs, axis=1)
                    action_t[action_index] = 1
            else:
                action_t[0] = 1

            if epsilon > self.kargs['final_epsilon'] \
                    and time_steps > self.kargs['observe_steps']:
                epsilon_range = self.kargs['initial_epsilon'] - self.kargs['final_epsilon']
                epsilon -= epsilon_range / self.kargs['explore_steps']

            image_raw, reward_t, terminal = self.game.frame_step(action_t)
            image_gray = self.convert_image(image_raw)
            new_state_t = np.concatenate([image_gray, state_t[:, :3, :, :]], axis=1)
            replay_memory.append((state_t, action_t, reward_t, new_state_t, terminal))
            if len(replay_memory) > self.kargs['replay_memory_size']:
                replay_memory.popleft()

            state_t = new_state_t
            time_steps += 1


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
    parser.add_argument('--initial_epsilon', type=float, default=0.0001,
                        help='initial epsilon')
    parser.add_argument('--final_epsilon', type=float, default=0.0001,
                        help='final epsilon')
    parser.add_argument('--replay_memory_size', type=int, default=50000,
                        help='size of replay memory')
    parser.add_argument('--frame_per_action', type=int, default=1,
                        help='frame per action')
    opt = parser.parse_args()

    Trainer(output_path=opt.output_path,
            media_path=opt.media_path,
            learning_rate=opt.learning_rate,
            batch_size=opt.batch_size,
            device=opt.device,
            num_actions=opt.num_actions,
            decay_rate_gamma=opt.decay_rate_gamma,
            observe_steps=opt.observe_steps,
            explore_steps=opt.explore_steps,
            initial_epsilon=opt.initial_epsilon,
            final_epsilon=opt.final_epsilon,
            replay_memory_size=opt.replay_memory_size,
            frame_per_action=opt.frame_per_action).train_network()


if __name__ == '__main__':
    main()
