from tensorforce.agents import DoubleDQN
from tensorforce import Agent, Environment


import gym


env = Environment.create(environment='gym', level='Elegans-v0', max_episode_timesteps=1000, arena_size=400,
                        reward_func_name='goto', use_double_step_state=True)
network_spec = [dict(type='dense', size=64, activation='leaky-relu'),
                dict(type='dense', size=128, activation='tanh'),
                dict(type='dense', size=64, activation='leaky-relu')]
agent_spec = 'double_dqn'
memory = 1024; batch_size = 256; lr = 5e-4; freq = 64
agent = Agent.create(
    agent=agent_spec,
    states=env.states(),
    actions=env.actions(),
    memory=memory,
    batch_size=batch_size,
    exploration=0.05,
    # learning_rate=dict(type='exponential', unit='episodes', num_steps=50, initial_value=lr, decay_rate=0.8),
    learning_rate=lr,
    # discount=1.,
    discount=0.95,
    update_frequency=freq,
    network=network_spec,
    config={'seed': SEED}
    )

reward_his = deque([])
angle_his = deque([])
action_his = deque([])
for episode in range(START_EPISODE+1, END_EPISODE+1):
    timestep = 1
    while timestep < MAX_TIMESTEP+1:
        if timestep == 1:
            rewards = []
            states = []
            actions = []
            state = env.reset()
            # agent.reset()
        # Get action
        action = agent.act(state)
        # Execute action in the environment
        state, is_terminal, reward = env.execute(action)
        rewards.append(reward)
        states.append(state)
        actions.append(action)
        timestep += 1
        agent.observe(reward=reward, terminal=bool(is_terminal))
        # Pass observation to the agent
        if is_terminal:
            agent.update()
            print('Finished episode {ep} after {ts} timesteps (reward: {reward})'.format(
                ep=episode, ts=timestep, reward=np.sum(rewards)))
            reward_his.append(np.sum(rewards))
            angle_his.extend([state[3]*6.3-3.15 for state in states])
            action_his.extend(actions)
            break
    if episode%100 == 0:
        agent.save(directory=PATH_TO_MODEL, append='episodes')
