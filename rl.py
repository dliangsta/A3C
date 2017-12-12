import gym
import numpy as np
import pickle
import os

def main():
    if os.path.isfile('agent.pkl'):
        agent = pickle.load(open('agent.pkl','rb'))
        print('loaded')
    else:
        agent = Agent()
    
    agent.train()

class Agent():

    def __init__(self):
        self.env = gym.make('FrozenLake8x8-v0')
        self.Q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        self.learning_rate = .8
        self.discount_rate = .99
        self.iteration = 0
        self.mean_rewards = []

    def train(self):
        for self.iteration in range(self.iteration, 1000000):
            # Train agent.
            self.play_episode(True)
            if self.iteration % 1000 == 0:
                # Copy table.
                # Run for 100 episodes.
                rewards = []
                for game in range(1000):
                    reward = self.play_episode(False)
                    rewards.append(reward)

                self.mean_rewards.append(np.mean(rewards))
                if len(self.mean_rewards) >= 30:
                    print('iteration: %9d, avg: %.3f, recent_avg: %.9f, recent_var: %.9f, learning_rate: %.9f' % (self.iteration, np.mean(rewards), np.mean(self.mean_rewards[-30:]), np.var(self.mean_rewards[-30:]), self.learning_rate))
                else:
                    print('iteration: %9d, avg: %.3f, learning_rate: %.9f' % (self.iteration, np.mean(rewards) ,self.learning_rate))
                if np.mean(rewards) >= .99:
                    print('Done!')

                self.save()

    def play_episode(self, train):
        # Reset environment and get first new observation
        state = self.env.reset()
        reward = 0
        done = False
        # The Q_table-Table learning algorithm
        for step in range(1000):

            if train:
                # Choose an action by greedily (with noise) picking from Q table
                action = np.argmax(self.Q_table[state, :] + np.random.randn(1, self.env.action_space.n) * 1000. / (self.iteration + 1))
            else:
                # Choose best action from Q table
                action = np.argmax(self.Q_table[state, :])

            # Get new state and reward from environment
            state_next, r, done, _ = self.env.step(action)

            if train:
                # Update Q_table-Table with new knowledge
                self.Q_table[state, action] = self.Q_table[state, action] + self.learning_rate * (r + self.discount_rate * np.max(self.Q_table[state_next, :]) - self.Q_table[state, action])
        
            reward += r
            state = state_next
        
            if done == True:
                break

        # Save.
        if not train:
            pickle.dump(self, open('agent.pkl','wb'))

        self.learning_rate = max(.08, .99999 * self.learning_rate)
        return reward

    def save(self):
        pickle.dump(self, open('agent.pkl','wb'))
    

if __name__ == '__main__':
    main()