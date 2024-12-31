import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import os
from pynput import keyboard

from envs.fixed_base_robot_env import FixedBaseRobotEnv

# Define the TensorBoard log directory
log_dir = "./models_and_logs/PPO_1/"
os.makedirs(log_dir, exist_ok=True)

# Create environment without rendering
env = FixedBaseRobotEnv(render_mode="human", findings_log_path="./models_and_logs/Perform_logs", logging=False)

# Create the model
# model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
# model = PPO.load(os.path.join(log_dir, "ppo_fixed_base_robot"), env=env, tensorboard_log=log_dir)
model = SAC.load("/home/konstantinkoenig/Desktop/Arbeitsplatz Ubuntu/Code/reinforcement_learning_projects/two_link_robot/models_and_logs/legacy/SAC_extra_layer_legacy/models_and_logs/SAC_extra_layer/best_model.zip", env=env, tensorboard_log=log_dir)
# model = PPO.load(os.path.join(log_dir, "best_model"), env=env, tensorboard_log=log_dir)

print(model.policy)

# Reset the environment
obs, info = env.reset()

# For keyboard input
reset_requested = False

def on_press(key):
    global reset_requested
    try:
        if key.char == 'r':
            reset_requested = True
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press)  # Start the listener
listener.start()


# Run the trained agent
for _ in range(5000):
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    # env.render()
    if terminated or truncated or reset_requested:
        obs, info = env.reset()
        reset_requested = False


# Close the environment
env.close()
