from stable_baselines3 import PPO
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.fixed_base_robot_env import FixedBaseRobotEnv

# Define the TensorBoard log directory
log_dir = "./models_and_logs/PPO_old/"
os.makedirs(log_dir, exist_ok=True)

# Create environment without rendering
env = FixedBaseRobotEnv(render_mode="human")

# Create the model
# model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
model = PPO.load(os.path.join(log_dir, "ppo_fixed_base_robot_revised_rewards_100M"), env=env, tensorboard_log=log_dir)
# model = PPO.load(os.path.join(log_dir, "best_model"), env=env, tensorboard_log=log_dir)


# # Reset the environment
# obs, info = env.reset()

# # For keyboard input
# reset_requested = False

# def on_press(key):
#     global reset_requested
#     try:
#         if key.char == 'r':
#             reset_requested = True
#     except AttributeError:
#         pass

# listener = keyboard.Listener(on_press=on_press)  # Start the listener
# listener.start()


# # Run the trained agent
# for _ in range(5000):
#     action, _states = model.predict(obs)
#     obs, reward, terminated, truncated, info = env.step(action)
#     # env.render()
#     if terminated or truncated or reset_requested:
#         obs, info = env.reset()
#         reset_requested = False

# Print model architecture
print(model.policy)

# Close the environment
env.close()
