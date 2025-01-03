import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import os
import sys
import datetime
import traceback
import torch as th

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from envs.fixed_base_robot_env import FixedBaseRobotEnv


# Define the TensorBoard log directory
model_dir = "./models_and_logs/SAC_extra_layer"
os.makedirs(model_dir, exist_ok=True)

# Create environment without rendering
env = FixedBaseRobotEnv(findings_log_path="models_and_logs/SAC_extra_layer/findings.txt") # not render_mode = "human" 

# Configure custom network architecture
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256]))

# Create the model
model = SAC('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=model_dir)
# model = SAC.load(os.path.join(model_dir, "ppo_fixed_base_robot"), env=env, tensorboard_log=model_dir)

# Set seed for reproducable results
model.set_random_seed(seed=1)

print(model.policy)

# Define an evaluation callback
eval_callback = EvalCallback(env, best_model_save_path=model_dir,
                            log_path=model_dir, eval_freq=1000,
                            deterministic=True, render=False)

# Set training steps here
TRAINING_STEPS_IN_MIO = 200

# Number of Previous trainings steps for correct na:
TRAINING_STEPS_OFFSET = 0

for i in range(TRAINING_STEPS_IN_MIO):

    # Train the model for 1Mio steps with the callback
    model.learn(total_timesteps=1000000, callback=eval_callback, reset_num_timesteps=False, tb_log_name="tb_logs")

    # Make an entry to the log
    current_time = datetime.datetime.now()
    env.log_message('\n')
    env.log_message("***** Evaluation after " + str(i +TRAINING_STEPS_OFFSET + 1) + "M steps *****")
    env.log_message("timestamp: " + str(current_time))
    env.log_findings_eval_dict()
    env.log_message('\n')

    # Save the model every 5 Mio Steps and write the performance to the log file
    if (i + TRAINING_STEPS_OFFSET + 1) % 5 == 0:
        model_filename = "ppo_fixed_base_robot_" + str(i + TRAINING_STEPS_OFFSET + 1) + "M"
        model.save(os.path.join(model_dir, model_filename))

# Evaluate the model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
print(f"Mean reward: {mean_reward} +/- {std_reward}")    


# Close the environment
env.close()

