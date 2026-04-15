from typing import Any, Tuple
from gymnasium.spaces import Discrete
from ray.rllib.algorithms import Algorithm
from ray.rllib.core import Columns
from ray.rllib.env.env_runner_group import EnvRunnerGroup
from ray.rllib.utils.typing import ResultDict

from schafkopfrl.environment.multi_agent_env import SchafkopfMultiAgentEnv
import torch
from torch.distributions import Categorical
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.metrics import EVALUATION_RESULTS, ENV_RUNNER_RESULTS
from schafkopfrl.policy.lstmrlmodule import LSTMRLModule
from schafkopfrl.policy.rulebased_policy import RuleBasedRLModule
import json
import numpy as np
from schafkopfrl.policy.transformerrlmodule import TransformerRLModule

class TournamentEvaluation:
    """Evaluate an RL policy by running a tournament against a rule-based policy.

    Parameters
    ----------
    rl_module_name : str
        Name of the RL module to evaluate.
    n_rounds : int
        Number of games per evaluation round.
    model_config : dict, optional
        Model configuration for LSTMRLModule (used in BC case).
    module_class : type, optional
        RL module class to use (used in BC case).
    """
    def __init__(self, rl_module_name: str, n_rounds: int, model_config: dict | None = None, module_class: type | None = None) -> None:
        self.rl_module_name: str = rl_module_name
        self.n_episodes: int = n_rounds
        self.model_config: dict = model_config 
        self.module_class = module_class

    def rulebased_tournament_eval_fn(self, algorithm: Algorithm, eval_workers: EnvRunnerGroup) -> Tuple[ResultDict, int, int]:
        """Run a tournament of the trained policy against a rule-based opponent.

        The trained policy plays as players 1 and 3, while the rule-based
        policy plays as players 0 and 2.

        Parameters
        ----------
        algorithm : Algorithm
            The RLlib algorithm providing the trained module.
        eval_workers : EnvRunnerGroup
            Evaluation workers (unused, required by RLlib API).

        Returns
        -------
        tuple[ResultDict, int, int]
            ``(metrics_dict, episodes_this_iter, timesteps_this_iter)``.
        """

        env: SchafkopfMultiAgentEnv = SchafkopfMultiAgentEnv()
        # env_runner is None for offline algorithms (e.g. BC); fall back to learner
        if algorithm.env_runner is not None and algorithm.env_runner.module is not None:
            policy: Any = algorithm.env_runner.module._rl_modules[self.rl_module_name]
        else: #more complicated for the case of behavioural cloning
            state_dicts = algorithm.learner_group.foreach_learner(
                lambda l: l.module[self.rl_module_name].get_state()
            )
            rl_module_state = list(state_dicts)[0].get()
            policy = self.module_class(
                observation_space=env.observation_space,
                action_space=env.action_space,
                model_config=self.model_config
            )
            policy.set_state(rl_module_state)
            policy.eval() # Set to evaluation mode
            device = torch.device("cpu")
            policy.to(device)
            
        rulebased_policy: RuleBasedRLModule = RuleBasedRLModule()  

        total_rewards: dict[str, float] = {"policy": 0.0, "rulebased": 0.0}

        # Determine device from model parameters
        try:
            device: torch.device = next(policy.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        for i in range(self.n_episodes):
            obs, info = env.reset(seed=i)
            done: bool = False
            rewards: dict[str, float] = {"player_0": 0.0, "player_1": 0.0, "player_2": 0.0, "player_3": 0.0}
            while not done:
                actions = {}
                for player_id, pobs in obs.items():
                    if player_id in ["player_1", "player_3"]:
                        tensor_obs: dict[str, torch.Tensor] = self.convert_obs_dict_to_tensor(pobs, device=str(device))

                        # Inference with the RLModule
                        logits = policy.forward_inference({"obs": tensor_obs})[Columns.ACTION_DIST_INPUTS]
                        dist = Categorical(logits=logits)
                        actions[player_id] = torch.tensor([dist.sample().item()])
                    else:
                        actions[player_id] = rulebased_policy.forward_inference(info[player_id])
                obs, rew, terminateds, _, info = env.step(actions)
                done = terminateds["__all__"]
                for k in rewards:
                    rewards[k] += rew.get(k, 0.0)

            total_rewards["policy"] += rewards["player_1"] + rewards["player_3"]
            total_rewards["rulebased"] += rewards["player_0"] + rewards["player_2"]
        

        print(env.env.render())
        mean_reward: float = total_rewards["policy"] / self.n_episodes / 2
        print("avg_reward_against_rulebased_policy", mean_reward)
        algorithm.metrics.log_value(
            "avg_reward_against_rulebased_policy",
            mean_reward,
            window=1,
        )
        # Seed the evaluation key so RLlib's peek() doesn't raise KeyError
        if not algorithm.metrics._key_in_stats(
            (EVALUATION_RESULTS, ENV_RUNNER_RESULTS)
        ):
            algorithm.metrics._set_key(
                (EVALUATION_RESULTS, ENV_RUNNER_RESULTS), {}
            )
        return {"avg_reward_against_rulebased_policy": mean_reward}, self.n_episodes, self.n_episodes

    def convert_obs_dict_to_tensor(self, obs_dict: dict[str, Any], device: str = "cpu") -> dict[str, torch.Tensor]:
        """Convert a numpy observation dict to batched PyTorch tensors.

        Parameters
        ----------
        obs_dict : dict[str, Any]
            Observation dict with numpy arrays.
        device : str, optional
            Target device, by default ``"cpu"``.

        Returns
        -------
        dict[str, torch.Tensor]
            Tensors with an added batch dimension.
        """
        return {k: torch.tensor(v, dtype=torch.int32, device=device).unsqueeze(0) for k, v in obs_dict.items()}


class ValidationAccuracyEvaluation:
    """Evaluate policy prediction accuracy on a validation shard dataset.

    Parameters
    ----------
    rl_module_name : str
        Name of the RL module to evaluate.
    validation_shard_path : str
        Path to the validation JSONL shard file.
    max_samples : int, optional
        Maximum number of samples to evaluate (None for all).
    model_config : dict, optional
        Model configuration for LSTMRLModule (used in BC case).
    module_class : type, optional
        RL module class to use (used in BC case).
    """
    def __init__(self, rl_module_name: str, validation_shard_path: str, max_samples: int | None = None, model_config: dict | None = None, module_class: type | None = None) -> None:
        self.rl_module_name: str = rl_module_name
        self.validation_shard_path: str = validation_shard_path
        self.max_samples: int | None = max_samples
        self.model_config: dict = model_config
        self.module_class = module_class

    def accuracy_eval_fn(self, algorithm: Algorithm, eval_workers: EnvRunnerGroup) -> Tuple[ResultDict, int, int]:
        """Evaluate prediction accuracy on a validation dataset.

        Parameters
        ----------
        algorithm : Algorithm
            The RLlib algorithm providing the trained module.
        eval_workers : EnvRunnerGroup
            Evaluation workers (unused, required by RLlib API).

        Returns
        -------
        tuple[ResultDict, int, int]
            ``(metrics_dict, samples_evaluated, samples_evaluated)``.
        """
        env: SchafkopfMultiAgentEnv = SchafkopfMultiAgentEnv()

        # Load the policy from the algorithm
        if algorithm.env_runner is not None and algorithm.env_runner.module is not None:
            policy: Any = algorithm.env_runner.module._rl_modules[self.rl_module_name]
        else:  # Behavioral cloning case
            state_dicts = algorithm.learner_group.foreach_learner(
                lambda l: l.module[self.rl_module_name].get_state()
            )
            rl_module_state = list(state_dicts)[0].get()
            policy = self.module_class(
                observation_space=env.observation_space,
                action_space=env.action_space,
                model_config=self.model_config
            )
            policy.set_state(rl_module_state)
            policy.eval()  # Set to evaluation mode

        # Determine device
        try:
            device: torch.device = next(policy.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        correct_predictions: int = 0
        total_samples: int = 0

        # Load and evaluate validation shard
        with open(self.validation_shard_path, 'r') as f:
            for line_idx, line in enumerate(f):
                if self.max_samples is not None and line_idx >= self.max_samples:
                    break

                # Parse JSON line
                sample: dict[str, Any] = json.loads(line)
                obs: dict[str, Any] = sample["obs"]
                ground_truth_action: int = sample["actions"]

                # Convert observation to tensor
                tensor_obs: dict[str, torch.Tensor] = self._convert_obs_dict_to_tensor(obs, device=str(device))

                # Run inference
                with torch.no_grad():
                    logits = policy.forward_inference({"obs": tensor_obs})[Columns.ACTION_DIST_INPUTS]
                    predicted_action: int = torch.argmax(logits, dim=-1).item()

                # Compare prediction with ground truth
                if predicted_action == ground_truth_action:
                    correct_predictions += 1

                total_samples += 1

        # Calculate and log accuracy
        accuracy: float = correct_predictions / total_samples if total_samples > 0 else 0.0
        print(f"Validation accuracy: {accuracy:.4f} ({correct_predictions}/{total_samples})")

        algorithm.metrics.log_value(
            "validation_accuracy",
            accuracy,
            window=1,
        )

        # Seed the evaluation key so RLlib's peek() doesn't raise KeyError
        if not algorithm.metrics._key_in_stats(
            (EVALUATION_RESULTS, ENV_RUNNER_RESULTS)
        ):
            algorithm.metrics._set_key(
                (EVALUATION_RESULTS, ENV_RUNNER_RESULTS), {}
            )

        return {"validation_accuracy": accuracy}, total_samples, total_samples

    def _convert_obs_dict_to_tensor(self, obs_dict: dict[str, Any], device: str = "cpu") -> dict[str, torch.Tensor]:
        """Convert an observation dict to batched PyTorch tensors.

        Parameters
        ----------
        obs_dict : dict[str, Any]
            Observation dict with numpy arrays or lists.
        device : str, optional
            Target device, by default ``"cpu"``.

        Returns
        -------
        dict[str, torch.Tensor]
            Tensors with an added batch dimension.
        """
        tensor_dict = {}
        for k, v in obs_dict.items():
            if isinstance(v, (list, np.ndarray)):
                v = np.array(v)
            tensor_dict[k] = torch.tensor(v, dtype=torch.int32, device=device).unsqueeze(0)
        return tensor_dict


class CombinedEvaluation:
    """Combine multiple evaluation functions into one RLlib custom evaluation."""

    def __init__(self, tournament_eval: TournamentEvaluation, accuracy_eval: ValidationAccuracyEvaluation) -> None:
        self.tournament_eval = tournament_eval
        self.accuracy_eval = accuracy_eval

    def combined_eval_fn(self, algorithm: Algorithm, eval_workers: EnvRunnerGroup) -> Tuple[ResultDict, int, int]:
        """Run both tournament and validation-accuracy evaluation functions."""
        tournament_metrics, tournament_episodes, tournament_steps = self.tournament_eval.rulebased_tournament_eval_fn(
            algorithm, eval_workers
        )
        accuracy_metrics, accuracy_samples, accuracy_steps = self.accuracy_eval.accuracy_eval_fn(
            algorithm, eval_workers
        )

        combined_metrics = {**tournament_metrics, **accuracy_metrics}
        return combined_metrics, max(tournament_episodes, accuracy_samples), max(tournament_steps, accuracy_steps)

