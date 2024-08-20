import gymnasium as gym
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import os
import sys
import datetime
import torch as th
from torch.optim import Adam

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from envs.fixed_base_robot_env import FixedBaseRobotEnv
from testing_and_diagnosis.print_model_info import *

# Define the directory to save the models
model_dir = os.path.join(os.path.dirname(__file__)) 
os.makedirs(model_dir, exist_ok=True)

# Create environment without rendering
env = FixedBaseRobotEnv(findings_log_path=model_dir+"/findings.txt", logging=True) # not render_mode = "human" 

# Training Algorithm
train_algorithm = "PPO"   ### change here   <-------------
# Model Architecture
model_architecture = [512, 256]    ### change here  <-------------
# activation function
activation_function = th.nn.Tanh    ### change heres     <-------------
# learning rate
learning_rate = 0.0003   ### change heres    <-------------
# Set the clip range
clip_range = 0.2     ### change heres   <-------------
# Set the discount factor
gamma = 0.99    ### change heres    <-------------
# Set weight decay for L2 regularization
weight_decay = 0.0 #1e-4    ### change heres  <-------------
# Set entropy coefficient for entropy regularization
entropy_coefficient = 0.0 #0.01  ### change heres  <-------------

# Configure custom network architecture
# policy_kwargs = dict(activation_fn=activation_function,
#                      net_arch=dict(pi=model_architecture, vf=model_architecture))

# Define custom optimizer with L2 regularization
optimizer_kwargs = dict(weight_decay=weight_decay, eps=1e-5)
# Include the optimizer details in policy_kwargs
policy_kwargs = dict(
    activation_fn=activation_function,
    net_arch=dict(pi=model_architecture, vf=model_architecture),
    optimizer_class=Adam,
    optimizer_kwargs=optimizer_kwargs
)

# Create the model     ### change heres      <-------------
model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=model_dir, learning_rate=learning_rate, clip_range=clip_range, gamma=gamma, ent_coef=entropy_coefficient)   ### change heres
# model = PPO.load(os.path.join(model_dir, "ppo_fixed_base_robot"), env=env, tensorboard_log=model_dir)

# Set seed for reproducable results
model.set_random_seed(seed=1)

# print model info
printModelInfo(model)

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

    # Save the model every 1 Mio Steps and write the performance to the log file
    # if (i + TRAINING_STEPS_OFFSET + 1) % 5 == 0:
    model_filename = "two_link_robot_" + train_algorithm + str(model_architecture) + "_" + str(i + TRAINING_STEPS_OFFSET + 1) + "M"
    model.save(os.path.join(model_dir, model_filename))

# Evaluate the model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
print(f"Mean reward: {mean_reward} +/- {std_reward}")    


# Close the environment
env.close()

