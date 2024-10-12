"""
Dans ce TP, nous allons implémenter un agent qui apprend à jouer au jeu Taxi-v3
de OpenAI Gym. Le but du jeu est de déposer un passager à une destination
spécifique en un minimum de temps. Le jeu est composé d'une grille de 5x5 cases
et le taxi peut se déplacer dans les 4 directions (haut, bas, gauche, droite).
Le taxi peut prendre un passager sur une case spécifique et le déposer à une
destination spécifique. Le jeu est terminé lorsque le passager est déposé à la
destination. Le jeu est aussi terminé si le taxi prend plus de 200 actions.

Vous devez implémenter un agent qui apprend à jouer à ce jeu en utilisant
les algorithmes Q-Learning et SARSA.

Pour chaque algorithme, vous devez réaliser une vidéo pour montrer que votre modèle fonctionne.
Vous devez aussi comparer l'efficacité des deux algorithmes en termes de temps
d'apprentissage et de performance.

A la fin, vous devez rendre un rapport qui explique vos choix d'implémentation
et vos résultats (max 1 page).
"""

import typing as t
import gymnasium as gym
import numpy as np
from qlearning import QLearningAgent
from qlearning_eps_scheduling import QLearningAgentEpsScheduling
from sarsa import SarsaAgent
from gymnasium.wrappers import RecordVideo
import os
import matplotlib.pyplot as plt


env = gym.make("Taxi-v3", render_mode="rgb_array")
n_actions = env.action_space.n  # type: ignore
def create_env(record: bool = False, video_dir: str = "videos", episode_trigger: int = 0) -> gym.Env:
    """
    Creates the Taxi-v3 environment with optional video recording.
    
    Args:
        record (bool): Whether to record video.
        video_dir (str): Directory where videos will be saved.
        episode_trigger (int): Every 'episode_trigger' episodes will be recorded.
    
    Returns:
        gym.Env: The configured environment.
    """
    env = gym.make("Taxi-v3", render_mode="rgb_array")
    
    if record:
        # Ensure the video directory exists
        os.makedirs(video_dir, exist_ok=True)
        
        # Use RecordVideo wrapper
        env = RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda episode_id: episode_id == episode_trigger,
            name_prefix="taxi-agent"
        )
    
    return env
env = create_env(record=True, video_dir="videos_q", episode_trigger=999)  # Record the 1000th episode


#################################################
# 1. Play with QLearningAgent
#################################################

agent = QLearningAgent(
    learning_rate=0.1, epsilon=1.0, gamma=0.99, legal_actions=list(range(n_actions))
)


def play_and_train(env: gym.Env, agent: QLearningAgent, t_max=int(1e4)) -> float:
    """
    This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total rewardb
    """
    total_reward: t.SupportsFloat = 0.0
    s, _ = env.reset()

    for _ in range(t_max):
        # Get agent to pick action given state s
        a = agent.get_action(s)

        next_s, r, done, _, _ = env.step(a)

        # Train agent for state s
        # BEGIN SOLUTION
        agent.update(s, a, r, next_s)
        total_reward += r
        s = next_s
        if done:
            break
        # END SOLUTION

    return total_reward


rewards_q = []
best_mean_reward = float('-inf')
no_improvement_count = 0
for i in range(10000):
    rewards_q.append(play_and_train(env, agent))
    if i % 100 == 0:
        mean_reward = np.mean(rewards_q[-100:])
        print(f"Episode {i}, mean reward: {mean_reward:.2f}, epsilon: {agent.epsilon:.4f}")
        
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        if mean_reward > 0:
            print("Performance satisfaisante atteinte!")
            break
        
        if no_improvement_count >= 50:  # Si pas d'amélioration pendant 5000 épisodes
            print("Réinitialisation de l'agent en raison d'un manque d'amélioration")
            agent.reset()
            no_improvement_count = 0


assert np.mean(rewards_q[-100:]) > 0.0

plt.plot(rewards_q)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('QLearningAgent')
plt.show()
env.close()
#################################################
# 2. Play with QLearningAgentEpsScheduling
#################################################

env = create_env(record=True, video_dir="videos_q_eps", episode_trigger=999)  # Record the 1000th episode
agent = QLearningAgentEpsScheduling(
    learning_rate=0.1,  # Augmenter le taux d'apprentissage
    epsilon=1.0,
    gamma=0.99,
    legal_actions=list(range(n_actions)),
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay_steps=100000  # Augmenter le nombre d'étapes pour la décroissance
)

rewards_q_eps = []
best_mean_reward = float('-inf')
no_improvement_count = 0
for i in range(10000):
    rewards_q_eps.append(play_and_train(env, agent))
    if i % 100 == 0:
        mean_reward = np.mean(rewards_q_eps[-100:])
        print(f"Episode {i}, mean reward: {mean_reward:.2f}, epsilon: {agent.epsilon:.4f}")

        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if mean_reward > 0:
            print("Performance satisfaisante atteinte!")
            break
        
        if no_improvement_count >= 50:  # Si pas d'amélioration pendant 5000 épisodes
            print("Réinitialisation de l'agent en raison d'un manque d'amélioration")
            agent.reset()
            no_improvement_count = 0


assert np.mean(rewards_q_eps[-100:]) > 0.0

# TODO: créer des vidéos de l'agent en action
env.close()
plt.plot(rewards_q_eps)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('QLearningAgentEpsScheduling')
plt.show()
####################
# 3. Play with SARSA
####################


env = create_env(record=True, video_dir="videos_sarsa", episode_trigger=999)  # Record the 1000th episode

agent = SarsaAgent(learning_rate=0.1, gamma=0.99, legal_actions=list(range(n_actions)))

rewards_sarsa = []
for i in range(10000):
    rewards_sarsa.append(play_and_train(env, agent))
    if i % 100 == 0:
        mean_reward = np.mean(rewards_sarsa[-100:])
        print(f"Episode {i}, mean reward: {mean_reward:.2f}")

        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if mean_reward > 0:
            print("Performance satisfaisante atteinte!")
            break
        
        if no_improvement_count >= 50:  # Si pas d'amélioration pendant 5000 épisodes
            print("Réinitialisation de l'agent en raison d'un manque d'amélioration")
            agent.reset()
            no_improvement_count = 0

assert np.mean(rewards_sarsa[-100:]) > 0.0

env.close() 
plt.plot(rewards_sarsa)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('SarsaAgent')
plt.show()

def plot_performance(rewards_q, rewards_q_eps, rewards_sarsa):
    plt.figure(figsize=(12, 8))
    

    # Calcul de la moyenn mobile pour lisser les courbes 
    def moving_average(rewards, window_size):
        return np.convolve(rewards, np.ones((window_size,))/window_size, mode='valid')

    episodes = 100
    smoothed_rewards_q = moving_average(rewards_q, episodes)
    smoothed_rewards_q_eps = moving_average(rewards_q_eps, episodes)
    smoothed_rewards_sarsa = moving_average(rewards_sarsa, episodes)

    plt.plot(smoothed_rewards_q, label='QLearningAgent', color='blue')
    plt.plot(smoothed_rewards_q_eps, label='QLearningAgentEpsScheduling', color='green')
    plt.plot(smoothed_rewards_sarsa, label='SarsaAgent', color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Smoothed Rewards')
    plt.title('Comparaison des performances des agents')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_performance(rewards_q, rewards_q_eps, rewards_sarsa)