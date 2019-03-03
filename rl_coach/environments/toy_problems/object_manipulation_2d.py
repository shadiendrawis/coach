#
# Copyright (c) 2017 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import copy
from typing import Union
import random
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image

from rl_coach.base_parameters import VisualizationParameters
from rl_coach.spaces import DiscreteActionSpace, PlanarMapsObservationSpace, VectorObservationSpace, StateSpace
from rl_coach.environments.environment import Environment, LevelSelection
from rl_coach.environments.environment import EnvironmentParameters
from rl_coach.filters.filter import NoInputFilter, NoOutputFilter


# Parameters
class ObjectManipulation2DEnvironmentParameters(EnvironmentParameters):
    def __init__(self):
        super().__init__()
        self.default_input_filter = NoInputFilter()
        self.default_output_filter = NoOutputFilter()

    @property
    def path(self):
        return 'rl_coach.environments.toy_problems.object_manipulation_2d:ObjectManipulation2DEnvironment'


class ObjectManipulation2DEnvironment(Environment):
    def __init__(self,
                 level: LevelSelection,
                 frame_skip: int,
                 visualization_parameters: VisualizationParameters,
                 seed: Union[None, int] = None,
                 human_control: bool=False,
                 custom_reward_threshold: Union[int, float]=None,
                 width: int = 84,
                 height: int = 84,
                 num_rotations: int = 24,
                 episode_length: int = 50000,
                 **kwargs):
        super().__init__(level, seed, frame_skip, human_control, custom_reward_threshold, visualization_parameters)

        # seed
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        self.object = Object2D(width, height, num_rotations)
        self.last_result = self.object.reset()

        self.state_space = StateSpace({})

        # image observations
        self.state_space['observation'] = PlanarMapsObservationSpace(shape=np.array([width, height, 3]), low=0, high=255)

        # measurements observations
        measurements_space_size = 4
        measurements_names = ['x-position', 'y-position', 'rotation', 'num_covered_states']
        self.state_space['measurements'] = VectorObservationSpace(shape=measurements_space_size,
                                                                  measurements_names=measurements_names)

        # actions
        self.num_actions = 6
        self.action_space = DiscreteActionSpace(self.num_actions)

        self.steps = 0
        self.episode_length = episode_length

        self.width = width
        self.height = height
        self.num_rotations = num_rotations
        self.bin_size = 1
        self.covered_states = np.zeros((int(self.width / self.bin_size),
                                        int(self.height / self.bin_size),
                                        self.num_rotations))
        self.num_covered_states = 0

        # render
        if self.is_rendered:
            image = np.squeeze(self.object.render())
            self.renderer.create_screen(image.shape[1], image.shape[0])

        # initialize the state by getting a new state from the environment
        self.reset_internal_state(True)

    def _update_state(self):
        self.state = dict()

        self.state['observation'] = self.last_result['observation']
        self.state['measurements'] = np.concatenate((self.last_result['measurements'], [self.num_covered_states]))

        self.reward = 0

        # self.done = self.steps == self.episode_length
        self.done = self.steps == 100

    def _take_action(self, action):
        self.steps += 1
        self.last_result = self.object.step(action)

        state = self.last_result['measurements']
        bin_x = int(state[1] / self.bin_size)
        bin_y = int(state[0] / self.bin_size)
        if self.covered_states[bin_x, bin_y, state[2]] == 0:
            self.covered_states[bin_x, bin_y, state[2]] = 1
            self.num_covered_states += 1

    def _restart_environment_episode(self, force_environment_reset=False):
        self.steps = 0
        # self.last_result = self.object.reset()

    def get_rendered_image(self):
        raw_img = copy.copy(np.squeeze(self.last_result['observation']))
        cover_image = np.kron(np.sum(self.covered_states, 2), np.ones((self.bin_size, self.bin_size)))
        cover_image = np.round((cover_image / self.num_rotations)*255)
        raw_img[:, :, 1] = np.maximum(raw_img[:, :, 1], cover_image)
        return raw_img


class Object2D:
    def __init__(self, width: int, height: int, num_rotations: int, init_state: Union[None, tuple] = None):
        self.width = width
        self.height = height
        self.num_rotations = num_rotations
        if init_state is None:
            self.state = self.sample_random_state()
        else:
            self.state = init_state
        self.vertex_pos = self.create_initial_shape()

    @staticmethod
    def create_initial_shape():
        # xy = np.array([[-10, -6], [-2, 9], [3, 6], [5, -8]])
        len = 10
        xy = np.array([[-len, -len], [-len, len], [len, len], [len, -len]])
        return xy

    def sample_random_state(self):
        return np.array([random.uniform(5, self.width-6),
                         random.uniform(5, self.height-6),
                         random.uniform(0, self.num_rotations)],
                        dtype=np.int32)

    def step(self, action):
        if action == 0:
            self.state[0] = min(self.width - 1, self.state[0] + 1)
        elif action == 1:
            self.state[0] = max(0, self.state[0] - 1)
        elif action == 2:
            self.state[1] = min(self.height - 1, self.state[1] + 1)
        elif action == 3:
            self.state[1] = max(0, self.state[1] - 1)
        elif action == 4:
            self.state[2] = (self.state[2] + 1) % self.num_rotations
        elif action == 5:
            self.state[2] = (self.state[2] - 1) % self.num_rotations
        result = dict()
        result['observation'] = self.render()
        result['measurements'] = self.state
        return result

    def reset(self):
        self.state = self.sample_random_state()
        result = dict()
        result['observation'] = self.render()
        result['measurements'] = self.state
        return result

    def render(self):
        theta = (self.state[2] / self.num_rotations) * 2 * np.pi
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        points = np.array([np.round(np.matmul(rotation_matrix, [v[0], v[1]])).astype(np.int32)
                           for v in self.vertex_pos])
        points += [self.state[0], self.state[1]]

        image = Image.new("RGB", (self.width, self.height))
        draw = ImageDraw.Draw(image)
        points = tuple([(p[0], p[1]) for p in points])
        draw.polygon(points, fill=(127, 127, 127))
        draw.line((points[0], points[1]), fill=(255, 0, 0), width=1)
        draw.line((points[1], points[2]), fill=(0, 255, 0), width=1)
        draw.line((points[2], points[3]), fill=(0, 0, 255), width=1)
        draw.line((points[3], points[0]), fill=(255, 255, 255), width=1)
        image = np.asarray(image)
        return image
