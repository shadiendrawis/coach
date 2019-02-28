from rl_coach.agents.ddqn_rnd_agent import DDQNRNDAgentParameters
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.toy_problems.object_manipulation_2d import ObjectManipulation2DEnvironmentParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.architectures.head_parameters import DuelingQHeadParameters
from rl_coach.memories.non_episodic.prioritized_experience_replay import PrioritizedExperienceReplayParameters
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.architectures.middleware_parameters import FCMiddlewareParameters
from rl_coach.architectures.layers import Conv2d, Dense

####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(50000)
schedule_params.steps_between_evaluation_periods = EnvironmentSteps(50000)
schedule_params.evaluation_steps = EnvironmentEpisodes(0)
schedule_params.heatup_steps = EnvironmentSteps(0)

#########
# Agent #
#########
agent_params = DDQNRNDAgentParameters()

# DQN params
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(1000)
agent_params.algorithm.discount = 0.999
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(4)

# NN configuration
agent_params.network_wrappers['main'].learning_rate = 0.00025
agent_params.network_wrappers['rnd'].learning_rate = 0.00025
agent_params.network_wrappers['main'].batch_size = 32
agent_params.network_wrappers['main'].replace_mse_with_huber_loss = False
input_embedder_params = InputEmbedderParameters(scheme=[Conv2d(32, 3, 1),
                                                        Conv2d(32, 1, 2),
                                                        Conv2d(32, 3, 1),
                                                        Conv2d(32, 1, 2),
                                                        Conv2d(64, 3, 1),
                                                        Conv2d(64, 1, 2)
                                                        ])
agent_params.network_wrappers['main'].input_embedders_parameters = {'observation': input_embedder_params}
agent_params.network_wrappers['rnd'].input_embedders_parameters = {'observation': input_embedder_params}
agent_params.network_wrappers['main'].middleware_parameters = FCMiddlewareParameters(scheme=[Dense(128)])
agent_params.network_wrappers['rnd'].middleware_parameters = FCMiddlewareParameters(activation_function='none',
                                                                                    scheme=[Dense(128)])
agent_params.network_wrappers['main'].heads_parameters = [DuelingQHeadParameters()]
# ER
# agent_params.memory = PrioritizedExperienceReplayParameters()
agent_params.memory.max_size = (MemoryGranularity.Transitions, 5000)

################
#  Environment #
################
env_params = ObjectManipulation2DEnvironmentParameters()

########
# Test #
########
preset_validation_params = PresetValidationParameters()

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters(),
                                    preset_validation_params=preset_validation_params)
