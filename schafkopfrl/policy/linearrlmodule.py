from ray.rllib.core import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_distributions import TorchCategorical
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI


class LinearRLModule(RLModule, ValueFunctionAPI):
    def __init__(
        self,
        observation_space,
        action_space,
        model_config,
        framework="torch",
        inference_only=False,
        learner_only=False,
        **kwargs
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config,
            framework=framework,
            inference_only=inference_only,
            learner_only=learner_only,
        )

        # Your custom model
        self.model = ModelCatalog.get_model_v2(
            obs_space=observation_space,
            action_space=action_space,
            num_outputs=action_space.n,
            model_config=model_config,
            framework=framework,
            name="linear_model",
        )

    @override(RLModule)
    def setup(self):
        self.framework = "torch"

    @override(RLModule)
    def _forward_inference(self, batch, **kwargs):
        logits, _ = self.model(batch)
        return {Columns.ACTION_DIST_INPUTS: logits}

    @override(RLModule)
    def _forward_exploration(self, batch, **kwargs):
        # For stochastic exploration (same as inference for PPO)
        return self.forward_inference(batch)

    @override(RLModule)
    def _forward_train(self, batch, **kwargs):
        return self.forward_inference(batch)

    @override(ValueFunctionAPI)
    def compute_values(self, batch, **kwargs):
        self.model(batch)
        # Get the value predictions
        value = self.model.value_function()
        return value

    def get_state(self, **kwargs):
        return self.model.state_dict()

    def set_state(self, state, **kwargs):
        self.model.load_state_dict(state)

    def get_exploration_action_dist_cls(self):
        # For discrete action spaces with torch, use TorchCategorical
        return TorchCategorical

    def get_train_action_dist_cls(self):
        # For discrete action spaces with torch, use TorchCategorical
        return TorchCategorical
