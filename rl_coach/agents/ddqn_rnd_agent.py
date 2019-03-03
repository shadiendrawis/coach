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

from typing import Union
import copy
import matplotlib.pyplot as plt

import numpy as np

from rl_coach.agents.dqn_agent import DQNAlgorithmParameters, DQNNetworkParameters
from rl_coach.agents.value_optimization_agent import ValueOptimizationAgent
from rl_coach.agents.agent import AgentParameters
from rl_coach.memories.non_episodic.experience_replay import ExperienceReplayParameters
from rl_coach.base_parameters import NetworkParameters
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.architectures.middleware_parameters import FCMiddlewareParameters
from rl_coach.architectures.head_parameters import RNDHeadParameters
from rl_coach.exploration_policies.e_greedy import EGreedyParameters
from rl_coach.schedules import LinearSchedule
from rl_coach.core_types import ActionInfo, Batch
from rl_coach.environments.toy_problems.object_manipulation_2d import Object2D


class DDQNRNDAgentParameters(AgentParameters):
    def __init__(self):
        super().__init__(algorithm=DQNAlgorithmParameters(),
                         exploration=EGreedyParameters(),
                         memory=ExperienceReplayParameters(),
                         networks={"main": DQNNetworkParameters(),
                                   "predictor": RNDNetworkParameters(),
                                   "constant": RNDNetworkParameters()})
        self.exploration.epsilon_schedule = LinearSchedule(1.0, 0.15, 15000)
    @property
    def path(self):
        return 'rl_coach.agents.ddqn_rnd_agent:DDQNRNDAgent'


class RNDNetworkParameters(NetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_embedders_parameters = {'observation': InputEmbedderParameters()}
        self.middleware_parameters = FCMiddlewareParameters(activation_function='none')
        self.heads_parameters = [RNDHeadParameters()]
        self.optimizer_type = 'Adam'
        self.clip_gradients = None
        self.create_target_network = False


# Double DQN - https://arxiv.org/abs/1509.06461
class DDQNRNDAgent(ValueOptimizationAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)
        self.rewards = []
        self.num_steps = 0
        if self.ap.visualization.render:
            self.plot = plt.imshow(np.zeros((84, 84)))
            self.cbar = plt.colorbar(self.plot)
            plt.title('Prediction Error')

    def learn_from_batch(self, batch):
        network_keys = self.ap.network_wrappers['main'].input_embedders_parameters.keys()

        dataset = copy.deepcopy(self.memory.transitions)
        dataset = Batch(dataset)
        dataset.shuffle()
        if self.num_steps % 1024 == 0:
            for i in range(int(dataset.size / self.ap.network_wrappers['predictor'].batch_size)):
                start = i * self.ap.network_wrappers['predictor'].batch_size
                end = (i + 1) * self.ap.network_wrappers['predictor'].batch_size

                const_embedding = self.networks['constant'].online_network.predict(
                    {k: v[start:end] for k, v in dataset.next_states(network_keys).items()})

                _ = self.networks['predictor'].train_and_sync_networks(
                    copy.copy({k: v[start:end] for k, v in dataset.next_states(network_keys).items()}),
                    [const_embedding]
                )

        embedding = self.networks['constant'].online_network.predict(batch.next_states(network_keys))
        prediction = self.networks['predictor'].online_network.predict(batch.next_states(network_keys))
        prediction_error = np.mean((embedding - prediction) ** 2, axis=1)
        # self.rewards += list(prediction_error)
        # intrinsic_rewards = (prediction_error - np.mean(prediction_error)) / (np.std(prediction_error) + 1e-15)
        intrinsic_rewards = np.zeros_like(prediction_error)
        intrinsic_rewards[np.argmax(prediction_error)] = 1

        selected_actions = np.argmax(self.networks['main'].online_network.predict(batch.next_states(network_keys)), 1)
        q_st_plus_1, TD_targets = self.networks['main'].parallel_prediction([
            (self.networks['main'].target_network, batch.next_states(network_keys)),
            (self.networks['main'].online_network, batch.states(network_keys))
        ])

        # initialize with the current prediction so that we will
        #  only update the action that we have actually done in this transition
        TD_errors = []
        for i in range(self.ap.network_wrappers['main'].batch_size):
            new_target = intrinsic_rewards[i] + \
                         self.ap.algorithm.discount * q_st_plus_1[i][selected_actions[i]]
            TD_errors.append(np.abs(new_target - TD_targets[i, batch.actions()[i]]))
            TD_targets[i, batch.actions()[i]] = new_target

        # update errors in prioritized replay buffer
        importance_weights = self.update_transition_priorities_and_get_weights(TD_errors, batch)

        result = self.networks['main'].train_and_sync_networks(batch.states(network_keys), TD_targets,
                                                               importance_weights=importance_weights)
        total_loss, losses, unclipped_grads = result[:3]

        return total_loss, losses, unclipped_grads
        # return 0, 0, 0

    def choose_action(self, curr_state):
        if self.num_steps % 1024 == 0 and self.ap.visualization.render:
            self.show_pred_err_img()
        self.num_steps += 1

        num_covered_states = curr_state['measurements'][3]
        self.agent_logger.create_signal_value('Number of Covered States', num_covered_states)

        actions_q_values = self.get_prediction(curr_state)

        # choose action according to the exploration policy and the current phase (evaluating or training the agent)
        actions_q_values = actions_q_values.squeeze()
        action = self.exploration_policy.get_action(actions_q_values)
        while not self._is_valid_action(action, curr_state['measurements']):
            action = self.exploration_policy.get_action(actions_q_values)
        # action, action_probabilities = self._choose_greedy_action(curr_state['measurements'])
        # while not self._is_valid_action(action, curr_state['measurements']):
        #     action, action_probabilities = self._choose_greedy_action(curr_state['measurements'])
        self._validate_action(self.exploration_policy, action)

        if actions_q_values is not None:
            # this is for bootstrapped dqn
            if type(actions_q_values) == list and len(actions_q_values) > 0:
                actions_q_values = self.exploration_policy.last_action_values

            # store the q values statistics for logging
            self.q_values.add_sample(actions_q_values)
            for i, q_value in enumerate(actions_q_values):
                self.q_value_for_action[i].add_sample(q_value)

            action_info = ActionInfo(action=action,
                                     action_value=actions_q_values[action],
                                     max_action_value=np.max(actions_q_values))
        else:
            action_info = ActionInfo(action=action)

        return action_info

    @staticmethod
    def _is_valid_action(action, state):
        if action == 0:
            ret = state[0] != 79
        elif action == 1:
            ret = state[0] != 5
        elif action == 2:
            ret = state[1] != 79
        elif action == 3:
            ret = state[1] != 5
        else:
            ret = True
        return ret

    def _choose_greedy_action(self, state):
        prediction_errors = []
        for action in range(6):
            obj = Object2D(84, 84, 24, state[:3])
            obj.step(action)
            img = np.expand_dims(obj.render(), 0)
            emb = self.networks['constant'].online_network.predict({'observation': img})
            pred = self.networks['predictor'].online_network.predict({'observation': img})
            emb = np.squeeze(emb)
            pred = np.squeeze(pred)
            pred_err = np.sum((emb - pred) ** 2)
            prediction_errors.append(pred_err)
        action_probabilities = prediction_errors / np.sum(prediction_errors)
        action = np.random.choice(list(range(6)), p=action_probabilities)
        return action, action_probabilities

    def show_pred_err_img(self):
        img = np.zeros((84, 84))
        for i in range(84):
            obs = []
            for j in range(84):
                k = np.random.choice(list(range(24)))
                obj = Object2D(84, 84, 24, (i, j, k))
                obs.append(obj.render())
            obs = np.stack(obs, axis=0)
            emb = self.networks['constant'].online_network.predict({'observation': obs})
            pred = self.networks['predictor'].online_network.predict({'observation': obs})
            pred_err = np.mean((emb - pred) ** 2, axis=1)
            img[:, i] = pred_err
        plt.imshow(img)
        self.cbar.set_clim(vmin=np.min(img), vmax=np.max(img))
        cbar_ticks = np.linspace(np.min(img), np.max(img), num=11, endpoint=True)
        self.cbar.set_ticks(cbar_ticks)
        self.cbar.draw_all()
        plt.pause(.1)
        plt.draw()
