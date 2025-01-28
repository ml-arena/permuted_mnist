from gymnasium.envs.registration import register

register(
    id='MetaMNIST-v0',
    entry_point='metamnist.env.metamnist:MetaMNISTEnv',
    max_episode_steps=100,
)