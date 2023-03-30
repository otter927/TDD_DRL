from gym.envs.registration import register

register(
    id='myenvtdd-v0',
    entry_point='myenv.envTDD:MyEnvTDD'
)

register(
    id='cartpoleenv-v0',
    entry_point='myenv.cartpole:CartPoleEnv'
)
