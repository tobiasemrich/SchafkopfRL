import ray
from ray.tune import Tuner
from ray.rllib.algorithms.bc import BC, BCConfig
from ray.tune.registry import register_env
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from gymnasium.spaces import Discrete, Dict, Box
import numpy as np

from schafkopfrl.environment.multi_agent_env import SchafkopfMultiAgentEnv
from schafkopfrl.policy.lstmrlmodule import LSTMRLModule


def main():

    ray.init(
        local_mode=True
    )
    # register the environment
    register_env("SchafkopfMultiAgentEnv", lambda config: SchafkopfMultiAgentEnv(config))

    ds = ray.data.read_json("/home/git/SchafkopfRL/expert_data_small.jsonl", lines=True)
    ds.schema()
    ds = ds.repartition(8)      # optional: tune parallelism
    #ds = ds.cache()             # materialize / keep it in the object store

    config = (
        BCConfig()
        .environment(
            "SchafkopfMultiAgentEnv",
            observation_space=Dict({
                "player_hand": Box(low=0, high=1, shape=(32,), dtype=np.int32),
                "action_history": Box(low=-1, high=42, shape=(8, 2), dtype=np.int32),
                "action_history_len": Box(low=0, high=8, shape=(1,), dtype=np.int32),
                "action_mask": Box(low=0, high=1, shape=(43,), dtype=np.int32),
            }),
            action_space=Discrete(43),
        )
        .offline_data(
            input_=ds,
            input_read_sample_batches=True,
            dataset_num_iters_per_learner=1,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=LSTMRLModule,
                model_config={
                    "fcnet_hiddens": [64, 64],
                    "lstm_hidden_size": 128,
                    "lstm_num_layers": 1
                },
            )
        )
        .training(
            lr=0.01,
            train_batch_size_per_learner=1024,
            beta=0.0,
            grad_clip=0.2
        )
        # .evaluation(evaluation_interval=None)  # Disable evaluation for offline BC
    )

    tuner = Tuner(
        trainable="BC",
        param_space=config.to_dict(),
        run_config=ray.air.RunConfig(
            storage_path="/ray_results/",
            stop={"training_iteration": 10},  # Reduced for testing
            checkpoint_config=ray.air.CheckpointConfig(checkpoint_at_end=True),
        ),
    )

    tuner.fit()

    
if __name__ == "__main__":
    main()