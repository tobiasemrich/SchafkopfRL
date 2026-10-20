# TorchRL / PettingZoo Transition
The environment is a PettingZoo AEC env (`schafkopfrl/environment/pettingzoo_env.py`), wrapped for TorchRL in
`schafkopfrl/environment/torchrl_env.py`. Training uses TorchRL (collector, GAE, `ClipPPOLoss`) instead of RLlib:

```
python scripts/train_ppo.py --encoder lstm --num-envs 4      # self-play PPO, evaluated against the rule-based policy
python scripts/create_immitation_dataset.py                  # expert shards from transcripts
python scripts/train_bc.py --encoder transformer             # behavioural cloning (last shard = validation)
python scripts/tournament_play.py                            # PIMC vs rule-based
```

# Repo Structure

```
SchafkopfRL/
├── schafkopfrl/            # core package
│   ├── environment         # schafkopf environment
│   ├── policy              # trainable and non-trainable policies
├── scripts/                # scripts to start training, process data, evaluate, ...
├── sauspiel_interface/     # crawler and interface to interact with sauspiel
```

# BC Experiments

| model | model config | agains rule-based (max) | accuracy (max) | comment |
| --- | ------------ | ----------------------- | -------------- | --- |
| LSTM | [128, 128], 128, 2 |  6.95 | 0.853 | batch: 32000
| Transformer | [128, 128], 128, 2, 2 | 5.00 | 83.9 | batch: 8000


## Next Steps
- [x] Rework Schafkopf_env to be compatible with TorchRL / PettingZoo
- [x] LSTM Agent
- [x] PIMC agent
- [x] Tournament
- [x] Immitation agent
- [next] HP PIMC Agent
- [] push to main
- [] train all agents
- [] reconsider ego-representation in state
- [] transformer based agent

