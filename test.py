import argparse
import numpy as np
import torch
from dqnnet import DQNNet
from flappybird import FlappyBird
from train import preprocess


class Tester(object):
    def __init__(self, **kargs):
        self.device = torch.device(kargs['device'])
        model = DQNNet(4, kargs['num_actions'])
        self.model = model.to(self.device).eval()
        assert kargs['weights'], 'weights not set'
        state_dict = torch.load(kargs['weights'], map_location=self.device)
        model.load_state_dict(state_dict)
        self.game = FlappyBird(kargs['media_path'], mute=False)
        self.rt = kargs

    def test_network(self):
        state_t = self.get_initial_state()
        time_steps = 0
        while 'playing flappybird':
            action_t = np.zeros(self.rt['num_actions'])
            if time_steps % self.rt['frame_per_action'] == 0:
                inputs = torch.from_numpy(state_t).to(self.device, non_blocking=True).float()
                with torch.no_grad():
                    outputs = self.model(inputs).cpu().numpy()
                action_index = np.argmax(outputs[0])
            else:
                action_index = 0
            action_t[action_index] = 1
            image_raw, reward_t, terminal = self.game.frame_step(action_t)
            if terminal:
                state_t = self.get_initial_state()
            else:
                state_t = np.concatenate([preprocess(image_raw), state_t[:, :3, :, :]], axis=1)
            time_steps += 1
            print('TIMESTEP:', time_steps, 
                  'ACTION:', action_index, 'REWARD:', reward_t)

    def get_initial_state(self):
        do_nothing = np.zeros(self.rt['num_actions'])
        do_nothing[0] = 1
        image_raw = self.game.frame_step(do_nothing)[0]
        state_t = np.tile(preprocess(image_raw), (1, 4, 1, 1))
        return state_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--media_path', type=str, default='./media',
                        help='media path')
    parser.add_argument('--weights', type=str, default='',
                        help='path to model weights')
    parser.add_argument('--device', type=str, default='cpu',
                        help='torch device')
    parser.add_argument('--num_actions', type=int, default=2,
                        help='number of actions')
    parser.add_argument('--frame_per_action', type=int, default=1,
                        help='frame per action')
    opt = parser.parse_args()
    Tester(**vars(opt)).test_network()


if __name__ == '__main__':
    main()
