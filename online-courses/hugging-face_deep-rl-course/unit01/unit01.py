# Hugging Face page: https://huggingface.co/learn/deep-rl-course/unit1/hands-on
    # Suggested to train on Google Colab (GPUs)

import gymnasium as gym

from huggingface_sb3 import load_from_hub, package_to_hub
from huggingface_hub import notebook_login # To log to our Hugging Face account to be able to upload models to the Hub.

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

def main():
    # %% Explore environment
    # We create our environment
    env = make_vec_env('LunarLander-v2', n_envs=16)
    env.reset()
    print("_______________OBSERVATION SPACE_______________ ")
    print("Observation Space Shape", env.observation_space.shape)
    print("Sample observation", env.observation_space.sample()) # Get a random observation

    print("\n_______________ACTION SPACE_______________")
    print("Action Space Shape", env.action_space.shape)
    print("Action Space Sample", env.action_space.sample()) # Take a random action

    # %% Train agent
    # DONE: Define a PPO MlpPolicy architecture
    # We use MultiLayerPerceptron (MLPPolicy) because the input is a vector,
    # if we had frames as input we would use CnnPolicy
    print("\n_______________Training_______________")
    model = PPO(
        policy = 'MlpPolicy',
        env = env,
        n_steps = 1024,
        batch_size = 64,
        n_epochs = 4,
        gamma = 0.999,
        gae_lambda = 0.98,
        ent_coef = 0.01,
        verbose=1
    )

    # DONE: Train it for 1,000,000 timesteps
    model.learn(total_timesteps=int(1e6))

    # DONE: Specify file name for model and save the model to file
    model_name = "ppo-LunarLander-v2"
    model.save(model_name)

    # %% Evaluate the agent
    # DONE: Evaluate the agent
    # Create a new environment for evaluation
    print("\n_______________Evaluating_______________")
    eval_env = Monitor(gym.make("LunarLander-v2", render_mode="rgb_array"))

    # Evaluate the model with 10 evaluation episodes and deterministic=True
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=50)

    # Print the results
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

if __name__ == "__main__":
    main()