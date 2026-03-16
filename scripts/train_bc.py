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
    ray.init(num_cpus=4, num_gpus=1)
    register_env("SchafkopfMultiAgentEnv", lambda config: SchafkopfMultiAgentEnv(config))

    data_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "expert_data.jsonl")
    storage_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ray_results")

    config: BCConfig = (
        BCConfig()
        .environment(
            "SchafkopfMultiAgentEnv",
            observation_space=Dict({
                "player_hand": Box(low=0, high=1, shape=(32,), dtype=np.float32),
                "action_history": Box(low=-1, high=42, shape=(88,), dtype=np.float32),
                "action_history_len": Box(low=0, high=44, shape=(1,), dtype=np.float32),
                "action_mask": Box(low=0, high=1, shape=(43,), dtype=np.float32),
            }),
            action_space=Discrete(43),
        )
        .offline_data(
            input_=[data_path],
            input_read_method="read_json",
            input_read_sample_batches=False,
            dataset_num_iters_per_learner=10,
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
            train_batch_size_per_learner=128,
            grad_clip=0.2,
        )
        .learners(num_learners=0, num_gpus_per_learner=1)
        .evaluation(
            evaluation_interval=3,
            custom_evaluation_function=TournamentEvaluation("default_policy", 30).rulebased_tournament_eval_fn
        )
    )

    tuner: Tuner = Tuner(
        "BC",
        param_space=config.to_dict(),
        run_config=RunConfig(
            storage_path=storage_path,
            name="bc_run",
            stop={"training_iteration": 10},
        ),
        tune_config=TuneConfig(num_samples=1),
    )
    results = tuner.fit()
    print(results.get_best_result().metrics)


if __name__ == "__main__":
    main()
