from ray.rllib.algorithms.callbacks import DefaultCallbacks

class DebugCallbacks(DefaultCallbacks):
    
    def on_train_result(self, *, algorithm: "Algorithm", metrics_logger = None, result: dict, **kwargs) -> None:
        policy_id = "linear_policy"
        # print("on train result id", id(algorithm.get_module("linear_policy")))
        #for name, param in algorithm.get_module("linear_policy").named_parameters():
        #    if param.requires_grad:
        #        print(f"[AFTER] {name}: {param.data.norm().item()}")

    def on_episode_end(self, *, episode, prev_episode_chunks = None, env_runner = None, metrics_logger = None, env = None, env_index: int, rl_module = None, worker = None, base_env = None, policies = None, **kwargs) -> None:
        from scipy.special import softmax
        # print("Wenz prob player 0", softmax(episode.agent_episodes["player_0"].extra_model_outputs[Columns.ACTION_DIST_INPUTS].data[0])[4])
        print("Wenz prob player 3", softmax(episode.agent_episodes["player_3"].extra_model_outputs[Columns.ACTION_DIST_INPUTS].data[0])[4])