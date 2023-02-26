import argparse
import os
import torch
import torch.nn as nn


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
        self.model = DQN(4, kargs['num_actions'])
        self.kargs = kargs

    def train_network(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='./model',
                        help='path to save outputs')
    parser.add_argument('--media_path', type=str, default='./media',
                        help='media path')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight decay')
    parser.add_argument('--learning_rate', type=float, default=0.1,
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
            weight_decay=opt.weight_decay,
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
