import os

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import ray
from ray.tune import Tuner
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from schafkopfrl.environment.multi_agent_env import SchafkopfMultiAgentEnv
from schafkopfrl.policy.lstmrlmodule import LSTMRLModule
from schafkopfrl.policy.rulebased_policy import RuleBasedRLModule

from schafkopfrl.evaluation import TournamentEvaluation

def main():

    storage_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ray_results")
    

    ray.init(num_cpus=4)
    # register the environment
    register_env("SchafkopfMultiAgentEnv", lambda config: SchafkopfMultiAgentEnv(config))

    config = (
        PPOConfig()
        .environment("SchafkopfMultiAgentEnv")
        .multi_agent(
            policies={
                "lstm_policy",
                "rulebased_policy"
            },
            policies_to_train=["lstm_policy"],
            policy_mapping_fn=lambda agent_id, episode, **kwargs: "lstm_policy",
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "lstm_policy": RLModuleSpec(
                        module_class=LSTMRLModule,
                        model_config={
                            "fcnet_hiddens": [64, 64],
                            "lstm_hidden_size": 128,
                            "lstm_num_layers": 1
                        },
                    ),
                    "rulebased_policy": RLModuleSpec(module_class=RuleBasedRLModule)
                }
            )
        )
        .env_runners(num_env_runners=3)
        .training(
            lr=0.01,
            gamma=0.9,
            kl_coeff=0.3,
            train_batch_size_per_learner=1024,
            grad_clip=0.2
            # entropy_coeff=0.01,
        )
        # .resources(num_gpus=1)
        # .learners(num_learners=1)
        # .learners(num_learners=1, num_gpus_per_learner=1)
        .evaluation(
            evaluation_interval=3,  # evaluate every N training iterations
            custom_evaluation_function=TournamentEvaluation("lstm_policy", 30).rulebased_tournament_eval_fn
        )
        # .callbacks(DebugCallbacks)
    )

    tuner = Tuner(
        trainable="PPO",
        param_space=config.to_dict(),
        run_config=ray.air.RunConfig(
            storage_path=storage_path,
            stop={"training_iteration": 500},
            checkpoint_config=ray.air.CheckpointConfig(checkpoint_at_end=True),
        ),
    )

    tuner.fit()

    #tune.run(
    #    "PPO",
    #    config=config.to_dict(),
    #    storage_path="/ray_results/",
    #    stop={"training_iteration": 500},
    #    checkpoint_at_end=True
    #)

    
if __name__ == "__main__":
    main()