# RLLib Transition
Currently switching to RLLib for
- better abstractions, more open-source, less self-written code
- environment paralellization => scalable training
- plethora of algorithms
- easier to understand + extend

# Repo Structure

SchafkopfRL/
├── schafkopfrl/            # core package
│   ├── environment         # schafkopf environment
│   ├── policy              # trainable and non-trainable policies
├── scripts/                # scripts to start training, process data, evaluate, ...
├── sauspiel_interface/     # crawler and interface to interact with sauspiel



## Next Steps
- [x] Rework Schafkopf_env to be compatible with RLLib
- [x] LSTM Agent
- [x] PIMC agent
- [x] Tournament
- [x] Immitation agent
- [next] HP PIMC Agent
- [] push to main
- [] train all agents
- [] reconsider ego-representation in state
- [] transformer based agent

