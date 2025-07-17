import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils import framework
from ray.tune.registry import register_env
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.models import ModelCatalog
from ray.rllib.core.models.catalog import Catalog

from environment.linear_env import LinearSchafkopfMultiAgentEnv
from policy.linear_model import LinearModel
from policy.linearrlmodule import LinearRLModule

def main():
    ray.init(local_mode=True)
    # register the environment
    register_env("LinearSchafkopfMultiAgentEnv", lambda config: LinearSchafkopfMultiAgentEnv(config))
    ModelCatalog.register_custom_model("linear_model", LinearModel)

    config = (
        PPOConfig()
        .environment("LinearSchafkopfMultiAgentEnv")
        .multi_agent(
            # policies=policies,
            policies={
                "linear_policy": PolicySpec(),
            },
            policy_mapping_fn=lambda agent_id, episode, **kwargs: "linear_policy",
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "linear_policy": RLModuleSpec(
                        module_class=LinearRLModule,
                        model_config={
                            "custom_model": "linear_model",
                            "fcnet_hiddens": [32, 32],
                        },
                    )
                }
            )
        )
        #.env_runners(num_env_runners=2)
        .training(
            lr=0.0002,
            train_batch_size_per_learner=2000,
            #num_epochs=10,
        )
    )

    #config.api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
    # ppo = config.build_algo()
    # run the training
    # print(ppo.train())
    tune.run(
        "PPO",
        config=config.to_dict(),
        storage_path="/ray_results/",
        stop={"training_iteration": 50},
        checkpoint_at_end=True,
    )
    
if __name__ == "__main__":
    main()