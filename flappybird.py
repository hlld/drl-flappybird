import os
import sys
import random
import pygame
from itertools import cycle


class FlappyBird(object):
    def __init__(self,
                 data_root,
                 screen_height=512,
                 screen_width=288,
                 pipe_gap_size=100,
                 fps=30,
                 mute=True):
        self._data_root = data_root
        self._screen_height = screen_height
        self._screen_width = screen_width
        self._pipe_gap_size = pipe_gap_size
        self._fps = fps
        self._mute = mute
        self._rt = {}
        self._init_game()
        self._reset_env()

    def _init_game(self):
        pygame.init()
        self._rt['fps_lock'] = pygame.time.Clock()
        self._rt['screen'] = \
            pygame.display.set_mode([self._screen_width, self._screen_height])
        pygame.display.set_caption('Flappy Bird')
        self._rt['images'] = {}
        self._rt['sounds'] = {}
        self._rt['hit_masks'] = {}

        sprite_path = os.path.join(self._data_root, 'sprites')
        numbers = []
        for index in range(10):
            file_path = os.path.join(sprite_path, f'{index}.png')
            numbers.append(pygame.image.load(file_path).convert_alpha())
        self._rt['images']['numbers'] = numbers

        self._rt['images']['base'] = \
            pygame.image.load(os.path.join(sprite_path, 'base.png')).convert_alpha()
        self._rt['images']['background'] = \
            pygame.image.load(os.path.join(sprite_path, 'background-black.png')).convert()

        players = []
        for player_status in ['upflap', 'midflap', 'downflap']:
            file_path = os.path.join(sprite_path, f'redbird-{player_status}.png')
            players.append(pygame.image.load(file_path).convert_alpha())
        self._rt['images']['player'] = players

        pipe_path = os.path.join(sprite_path, 'pipe-green.png')
        self._rt['images']['pipe'] = [
            pygame.transform.rotate(
                pygame.image.load(pipe_path).convert_alpha(), 180),
            pygame.image.load(pipe_path).convert_alpha(),
        ]

        audio_path = os.path.join(self._data_root, 'audios')
        sound_suffix = '.wav' if 'win' in sys.platform else '.ogg'
        for sound_name in ['die', 'hit', 'point', 'swoosh', 'wing']:
            file_path = os.path.join(audio_path, sound_name + sound_suffix)
            self._rt['sounds'][sound_name] = pygame.mixer.Sound(file_path)

        self._rt['hit_masks']['pipe'] = [
            self.get_hit_mask(self._rt['images']['pipe'][0]),
            self.get_hit_mask(self._rt['images']['pipe'][1]),
        ]
        self._rt['hit_masks']['player'] = [
            self.get_hit_mask(self._rt['images']['player'][0]),
            self.get_hit_mask(self._rt['images']['player'][1]),
            self.get_hit_mask(self._rt['images']['player'][2]),
        ]

        self._rt['player_w'] = self._rt['images']['player'][0].get_width()
        self._rt['player_h'] = self._rt['images']['player'][0].get_height()
        self._rt['pipe_w'] = self._rt['images']['pipe'][0].get_width()
        self._rt['pipe_h'] = self._rt['images']['pipe'][0].get_height()
        self._rt['background_w'] = self._rt['images']['background'].get_width()
        self._rt['player_index_gen'] = cycle([0, 1, 2, 1])

    def _reset_env(self):
        self._rt['player_index'] = 0
        self._rt['loop_iter'] = 0
        self._rt['score'] = 0
        self._rt['player_x'] = int(self._screen_width * 0.2)
        self._rt['player_y'] = int((self._screen_height - self._rt['player_h']) / 2)
        self._rt['base_y'] = self._screen_height * 0.79
        self._rt['base_x'] = 0
        self._rt['base_shift'] = \
            self._rt['images']['base'].get_width() - self._rt['background_w']

        new_pipe1 = self.get_random_pipe()
        new_pipe2 = self.get_random_pipe()
        self._rt['upper_pipes'] = [
            {'x': self._screen_width, 'y': new_pipe1[0]['y']},
            {'x': self._screen_width * 1.5, 'y': new_pipe2[0]['y']},
        ]
        self._rt['lower_pipes'] = [
            {'x': self._screen_width, 'y': new_pipe1[1]['y']},
            {'x': self._screen_width * 1.5, 'y': new_pipe2[1]['y']},
        ]

        self._rt['pipe_velocity_x'] = -4
        self._rt['player_velocity_y'] = 0
        self._rt['player_max_velocity_y'] = 10
        self._rt['player_min_velocity_y'] = -8
        self._rt['player_acceleration_y'] = 1
        self._rt['player_flap_acceleration'] = -9
        self._rt['player_flapped'] = False

    @staticmethod
    def get_hit_mask(image):
        mask = []
        for x in range(image.get_width()):
            mask.append([])
            for y in range(image.get_height()):
                mask[x].append(bool(image.get_at((x, y))[3]))
        return mask

    @staticmethod
    def check_collision(rect1, rect2, hit_mask1, hit_mask2):
        rect = rect1.clip(rect2)
        if rect.width == 0 or rect.height == 0:
            return False

        x1, y1 = rect.x - rect1.x, rect.y - rect1.y
        x2, y2 = rect.x - rect2.x, rect.y - rect2.y

        for x in range(rect.width):
            for y in range(rect.height):
                if hit_mask1[x1 + x][y1 + y] and hit_mask2[x2 + x][y2 + y]:
                    return True
        return False

    def get_random_pipe(self):
        gap_ys = [20, 30, 40, 50, 60, 70, 80, 90]
        index = random.randint(0, len(gap_ys) - 1)
        gap_y = gap_ys[index]
        gap_y += int(self._rt['base_y'] * 0.2)
        pipe_x = self._screen_width + 10
        return [
            {'x': pipe_x, 'y': gap_y - self._rt['pipe_h']},
            {'x': pipe_x, 'y': gap_y + self._pipe_gap_size},
        ]

    def check_crash(self):
        player_w = self._rt['images']['player'][0].get_width()
        player_h = self._rt['images']['player'][0].get_height()
        player_hit_mask = \
            self._rt['hit_masks']['player'][self._rt['player_index']]
        upper_hit_mask = self._rt['hit_masks']['pipe'][0]
        lower_hit_mask = self._rt['hit_masks']['pipe'][1]

        if self._rt['player_y'] + player_h >= self._rt['base_y'] - 1:
            return True
        else:
            player_rect = \
                pygame.Rect(self._rt['player_x'], self._rt['player_y'], player_w, player_h)
            for upper_pipe, lower_pipe in \
                    zip(self._rt['upper_pipes'], self._rt['lower_pipes']):
                upper_pipe_rect = pygame.Rect(upper_pipe['x'], upper_pipe['y'],
                                              self._rt['pipe_w'], self._rt['pipe_h'])
                lower_pipe_rect = pygame.Rect(lower_pipe['x'], lower_pipe['y'],
                                              self._rt['pipe_w'], self._rt['pipe_h'])
                upper_collide = self.check_collision(player_rect, upper_pipe_rect,
                                                     player_hit_mask, upper_hit_mask)
                lower_collide = self.check_collision(player_rect, lower_pipe_rect,
                                                     player_hit_mask, lower_hit_mask)
                if upper_collide or lower_collide:
                    return True
        return False

    def show_score(self, score):
        numbers = self._rt['images']['numbers']
        screen = self._rt['screen']

        score_digits = [int(x) for x in list(str(score))]
        total_width = 0
        for digit in score_digits:
            total_width += numbers[digit].get_width()

        x_offset = (self._screen_width - total_width) / 2
        for digit in score_digits:
            screen.blit(numbers[digit], (x_offset, self._screen_height * 0.1))
            x_offset += numbers[digit].get_width()

    def frame_step(self, actions):
        pygame.event.pump()
        reward = 0.1
        terminal = False
        screen = self._rt['screen']
        images = self._rt['images']
        sounds = self._rt['sounds']

        if sum(actions) != 1:
            raise ValueError('Multiple actions')
        if actions[1] == 1:
            if self._rt['player_y'] > -2 * self._rt['player_h']:
                self._rt['player_velocity_y'] = self._rt['player_flap_acceleration']
                self._rt['player_flapped'] = True
                if not self._mute:
                    sounds['wing'].play()

        player_mid_pos = self._rt['player_x'] + self._rt['player_w'] / 2
        for pipe in self._rt['upper_pipes']:
            pipe_mid_pos = pipe['x'] + self._rt['pipe_w'] / 2
            if pipe_mid_pos <= player_mid_pos < pipe_mid_pos + 4:
                self._rt['score'] += 1
                if not self._mute:
                    sounds['point'].play()
                reward = 1

        if (self._rt['loop_iter'] + 1) % 3 == 0:
            self._rt['player_index'] = next(self._rt['player_index_gen'])
        self._rt['loop_iter'] = (self._rt['loop_iter'] + 1) % 30
        self._rt['base_x'] = -((-self._rt['base_x'] + 100) % self._rt['base_shift'])

        if self._rt['player_velocity_y'] < self._rt['player_max_velocity_y'] \
                and not self._rt['player_flapped']:
            self._rt['player_velocity_y'] += self._rt['player_acceleration_y']
        if self._rt['player_flapped']:
            self._rt['player_flapped'] = False
        self._rt['player_y'] += min(self._rt['player_velocity_y'],
                                    self._rt['base_y'] - self._rt['player_y'] - self._rt['player_h'])
        if self._rt['player_y'] < 0:
            self._rt['player_y'] = 0

        for upper_pipe, lower_pipe in zip(self._rt['upper_pipes'], self._rt['lower_pipes']):
            upper_pipe['x'] += self._rt['pipe_velocity_x']
            lower_pipe['x'] += self._rt['pipe_velocity_x']

        if 0 < self._rt['upper_pipes'][0]['x'] < 5:
            new_pipe = self.get_random_pipe()
            self._rt['upper_pipes'].append(new_pipe[0])
            self._rt['lower_pipes'].append(new_pipe[1])

        if self._rt['upper_pipes'][0]['x'] < -self._rt['pipe_w']:
            self._rt['upper_pipes'].pop(0)
            self._rt['lower_pipes'].pop(0)

        is_crash = self.check_crash()
        if is_crash:
            if not self._mute:
                sounds['hit'].play()
                sounds['die'].play()
            terminal = True
            self._reset_env()
            reward = -1

        screen.blit(images['background'], (0, 0))

        for upper_pipe, lower_pipe in zip(self._rt['upper_pipes'], self._rt['lower_pipes']):
            screen.blit(images['pipe'][0], (upper_pipe['x'], upper_pipe['y']))
            screen.blit(images['pipe'][1], (lower_pipe['x'], lower_pipe['y']))

        screen.blit(images['base'], (self._rt['base_x'], self._rt['base_y']))
        self.show_score(self._rt['score'])
        screen.blit(images['player'][self._rt['player_index']],
                    (self._rt['player_x'], self._rt['player_y']))

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()

        self._rt['fps_lock'].tick(self._fps)
        return image_data, reward, terminal
