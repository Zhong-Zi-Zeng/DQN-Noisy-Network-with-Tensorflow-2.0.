import gym
from DQN import Agent
import matplotlib.pyplot as plt

EPISODE = 500

def main(agent):
    reward_list = []
    for i in range(EPISODE):
        done = False
        state = env.reset()
        total_reward = 0
        print('now episode is: {}'.format(i))

        while not done:
            # env.render()
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.remember(state,action,reward,next_state,done)
            agent.learn()
            total_reward += reward

            state = next_state

        reward_list.append(total_reward)
    return reward_list

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    use_noisy_agent = Agent(alpha=0.0005,
                  gamma=0.99,
                  n_actions=2,
                  epsilon=None,
                  batch_size=256,
                  epsilon_end=None,
                  epsilon_dec=None,
                  mem_size=10000,
                  iteration=20,
                  input_shape=4,
                  use_noisy=True)

    not_use_noisy_agent = Agent(alpha=0.0005,
                  gamma=0.99,
                  n_actions=2,
                  epsilon=0.7,
                  batch_size=32,
                  epsilon_end=0.01,
                  epsilon_dec=0.95,
                  mem_size=10000,
                  iteration=100,
                  input_shape=4,
                  use_noisy=False)

    use_noisy_agent = main(use_noisy_agent)
    not_use_noisy_agent = main(not_use_noisy_agent)

    plt.plot(use_noisy_agent, label='use_noisy')
    plt.plot(not_use_noisy_agent, label='not_use_noisy')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid()
    plt.show()

