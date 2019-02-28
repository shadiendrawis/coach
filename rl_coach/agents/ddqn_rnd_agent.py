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

import numpy as np

from rl_coach.agents.dqn_agent import DQNAlgorithmParameters, DQNNetworkParameters
from rl_coach.agents.value_optimization_agent import ValueOptimizationAgent
from rl_coach.agents.agent import AgentParameters
from rl_coach.memories.non_episodic.experience_replay import ExperienceReplayParameters
from rl_coach.base_parameters import NetworkParameters
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.architectures.middleware_parameters import FCMiddlewareParameters
from rl_coach.architectures.head_parameters import RNDHeadParameters
from rl_coach.exploration_policies.categorical import CategoricalParameters
from rl_coach.core_types import ActionInfo
from rl_coach.environments.toy_problems.object_manipulation_2d import Object2D


class DDQNRNDAgentParameters(AgentParameters):
    def __init__(self):
        super().__init__(algorithm=DQNAlgorithmParameters(),
                         exploration=CategoricalParameters(),
                         memory=ExperienceReplayParameters(),
                         networks={"main": DQNNetworkParameters(), "rnd": RNDNetworkParameters()})

    @property
    def path(self):
        return 'rl_coach.agents.ddqn_rnd_agent:DDQNRNDAgent'


class RNDNetworkParameters(NetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_embedders_parameters = {'observation': InputEmbedderParameters()}
        self.middleware_parameters = FCMiddlewareParameters(activation_function='none')
        self.heads_parameters = [RNDHeadParameters(), RNDHeadParameters()]
        self.optimizer_type = 'Adam'
        self.clip_gradients = None
        self.use_separate_networks_per_head = True
        self.create_target_network = True


# Double DQN - https://arxiv.org/abs/1509.06461
class DDQNRNDAgent(ValueOptimizationAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)
        self.prediction_error_list = []

    def learn_from_batch(self, batch):
        network_keys = self.ap.network_wrappers['main'].input_embedders_parameters.keys()

        state_embedding = self.networks['rnd'].online_network.predict(
            {k: v for k, v in batch.states(network_keys).items()})

        result = self.networks['rnd'].train_and_sync_networks(
                copy.copy({k: v for k, v in batch.states(network_keys).items()}),
                [state_embedding[0], state_embedding[0]]
            )

        embedding, prediction = self.networks['rnd'].target_network.predict(batch.next_states(network_keys))
        prediction_error = np.mean((embedding - prediction) ** 2, axis=1)
        self.prediction_error_list += list(prediction_error)
        intrinsic_rewards = (prediction_error - np.mean(self.prediction_error_list)) / (
            np.std(self.prediction_error_list) + 1e-8)

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
        num_covered_states = curr_state['measurements'][3]
        self.agent_logger.create_signal_value('Number of Covered States', num_covered_states)

        actions_q_values = self.get_prediction(curr_state)

        # choose action according to the exploration policy and the current phase (evaluating or training the agent)
        actions_q_values = actions_q_values.squeeze()
        e_q = np.exp(actions_q_values - np.max(actions_q_values))
        action_probabilities = e_q / e_q.sum()
        # action_probabilities = np.ones_like(action_probabilities) / action_probabilities.shape[0]
        action = self.exploration_policy.get_action(action_probabilities)
        while not self._is_valid_action(action, curr_state['measurements']):
            action = self.exploration_policy.get_action(action_probabilities)
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
            ret = state[0] != 83
        elif action == 1:
            ret = state[0] != 0
        elif action == 2:
            ret = state[1] != 83
        elif action == 3:
            ret = state[1] != 0
        else:
            ret = True
        return ret

    def _choose_greedy_action(self, state):
        prediction_errors = []
        for action in range(6):
            obj = Object2D(84, 84, 16, state[:3])
            obj.step(action)
            img = np.expand_dims(obj.render(), 0)
            emb, pred = self.networks['rnd'].online_network.predict({'observation': img})
            emb = np.squeeze(emb)
            pred = np.squeeze(pred)
            pred_err = np.sum((emb - pred) ** 2)
            prediction_errors.append(pred_err)
        action_probabilities = prediction_errors / np.sum(prediction_errors)
        action = np.random.choice(list(range(6)), p=action_probabilities)
        return action, action_probabilities
