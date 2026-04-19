import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

import pickle
import numpy as np
import gymnasium as gym
import traceback

try:
    from stable_baselines3 import PPO
    from stable_baselines3.ppo import MlpPolicy
    from imitation.algorithms.adversarial.gail import GAIL
    from imitation.rewards.reward_nets import BasicRewardNet
    from imitation.util.networks import RunningNorm
    from imitation.util.util import make_vec_env
except Exception as e:
    print(f"Error importing SB3 or Imitation: {e}")
    traceback.print_exc()

# Import the custom environment
from robot_env import Op3GymEnv

def main():
    print("Starting GAIL Training Pipeline...")

    # Load trajectories
    traj_path = "normalized_expert_trajectories.pkl"
    try:
        with open(traj_path, "rb") as f:
            expert_trajs = pickle.load(f)
        print(f"Loaded {len(expert_trajs)} expert trajectories.")
    except Exception as e:
        print(f"Failed to load expert trajectories from {traj_path}: {e}")
        return

    # To create a vectorized Gym environment for parallel training
    # Standard practice with SB3 and imitation libraries
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    num_envs = 1
    venv = DummyVecEnv([lambda: Op3GymEnv()])
    print("Vectorized robot environment created.")

    # Generator: PPO agent that controls the robot joints
    learner = PPO(
        env=venv,
        policy=MlpPolicy,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        gamma=0.99,
        n_epochs=10,
    )

    # Discriminator reward network 
    reward_net = BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
    )

    # Setup GAIL Algorithm
    # GAIL takes the Generator (PPO agent) and Discriminator and trains them adversarially
    gail_trainer = GAIL(
        demonstrations=expert_trajs,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=4,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
    )

    print("Training GAIL... ")
    
    # Run the trainer. For demo purposes, we will train for 10_000 steps.
    total_timesteps = 10_000
    try:
        gail_trainer.train(total_timesteps=total_timesteps)
        print("GAIL Training complete!")
        
        # Save the Generator policy
        save_path = "op3_gail_policy.zip"
        learner.save(save_path)
        print(f"Robot policy saved to {save_path}")

    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    main()
