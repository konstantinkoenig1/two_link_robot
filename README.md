# README

## Project Description

This project implements a two-link robot controlled by a NN trained with a deep reinforcement learning algorithm. 

The end-effector must reach a target in a randomly generated location in the least possible time. When reaching the target, the end-effector speed must be sufficiently low to avoid breaking the target.

## How to run the demonstration?

### Step 1: Recreate conda environment
Navigate to root of directory "two_link_robot" and execute this command:

conda env create -f environment.yml

### Step 2: Activate conda environment

conda activate myenv_two_link_robot

### Step 3: Run
To demonstrate the best saved model, run: 

python3 perform.py

The model has already been trained.

## Author
Konstantin König, Dec 31st 2024
konstantin.koenig@aol.com