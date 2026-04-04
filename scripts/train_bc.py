import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os
import numpy as np
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.bc import BCConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune import Tuner, TuneConfig
from ray.air import RunConfig
from gymnasium.spaces import Discrete, Dict, Box

from schafkopfrl.environment.multi_agent_env import SchafkopfMultiAgentEnv
from schafkopfrl.policy.lstmrlmodule import LSTMRLModule
from schafkopfrl.evaluation import TournamentEvaluation


def main() -> None:
    """Configure and launch Behavioral Cloning training from expert data."""
    ray.init(num_cpus=6, num_gpus=1)
    register_env("SchafkopfMultiAgentEnv", lambda config: SchafkopfMultiAgentEnv(config))

    data_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "shards")
    num_shards = 77
    input_data_paths = []
    for shard_idx in range(num_shards):
        shard_path = os.path.join(data_path, f'expert_data_shard_{shard_idx:03d}.jsonl')
        input_data_paths.append(shard_path)

    storage_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ray_results")

    config: BCConfig = (
        BCConfig()
        .environment(
            "SchafkopfMultiAgentEnv"
        )
        .offline_data(
            input_=input_data_paths,
            input_read_method="read_json",
            input_read_sample_batches=False,
            dataset_num_iters_per_learner=10,
            materialize_data=False,
            materialize_mapped_data=False
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=LSTMRLModule,
                model_config={
                    "fcnet_hiddens": [128, 128],
                    "lstm_hidden_size": 128,
                    "lstm_num_layers": 2
                },
            )
        )
        .training(
            lr=0.0005,
            train_batch_size_per_learner=32000,
            grad_clip=0.2,
            num_sgd_iter=4
        )
        .learners(num_learners=1, num_gpus_per_learner=1)
        .evaluation(
            evaluation_interval=20,
            evaluation_num_env_runners=1,
            custom_evaluation_function=TournamentEvaluation("default_policy", 200).rulebased_tournament_eval_fn
        )
    )

    tuner: Tuner = Tuner(
        "BC",
        param_space=config.to_dict(),
        run_config=RunConfig(
            storage_path=storage_path,
            name="bc_run",
            stop={"training_iteration": 100000},
        ),
        tune_config=TuneConfig(num_samples=1),
    )
    results = tuner.fit()
    print(results.get_best_result().metrics)


if __name__ == "__main__":
    main()
