import copy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os
import datetime

from rendering.rendering import Renderer

# Add  parent directory to the system path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from definitions import *


## 
class FixedBaseRobotEnv(gym.Env):
    """Custom Environment that follows gym interface. inheriting from the gym.Env class
    Methods: __init__(), reset(), step(); render(); close();
    Attributes: self.action_space;  self.configuration, self.render_mode;
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100} 

    def __init__(self, render_mode=None, findings_log_path="findings.txt", logging = False):
        # Define action and observation space
        # They must be gymnasium.spaces objects
    
        # Stable baselines bug, use next code snippet instead
        # # Define action_space: The manipulator can set the joint torques of both joints continuously in defined interval.
        # self.action_space = spaces.Box(low=-MAX_TORQUE, high=MAX_TORQUE, shape=(1,2), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,2), dtype=np.float32)

        # # Define action_space: The manipulator can set the joint torques of both joints continuously in defined interval.
        # self.action_space = spaces.Dict({"tau_1": spaces.Box(low=-MAX_TORQUE, high=MAX_TORQUE), "tau_2": spaces.Box(low=-MAX_TORQUE, high=MAX_TORQUE)})


        # Observation space:
        '''
        observation_space: 8x1 array with 
        0) x component of direction of target   min: -VIEWPORT_SIDE/2   max: VIEWPORT_SIDE/2
        1) y component of direction of target   min: 0                  max: VIEWPORT_SIDE/2
        2) velocity of end-effector in x        min: -MAX_VEL           max: MAX_VEL
        3) velocity of end-effector in y        min: -MAX_VEL           max: MAX_VEL
        4) joint 1 angle                       min: -MAX_ANGLE         max: MAX_ANGLE
        5) joint 2 angle                       min: -MAX_ANGLE         max: MAX_ANGLE
        6) joint 1 angular velocity            min: -OMEGA_MAX         max: OMEGA_MAX
        7) joint 2 angular velocity            min: -OMEGA_MAX         max: OMEGA_MAX
        '''

        
        
        self.observation_space = spaces.Box(low=np.array([-VIEWPORT_SIDE, # 0 
                                                        -VIEWPORT_SIDE/2, # 1
                                                        -MAX_VEL, # 2
                                                        -MAX_VEL, # 3
                                                        0, # 4
                                                        0, # 5
                                                        -OMEGA_MAX, # 6
                                                        -OMEGA_MAX,]),  # 7
                                            high=np.array([VIEWPORT_SIDE, # 0 # Scaling with MAX_TORQUE later. Bug workaround
                                                        VIEWPORT_SIDE/2, # 1 # Scaling with MAX_TORQUE later. Bug workaround
                                                        MAX_VEL, # 2
                                                        MAX_VEL, # 3
                                                        MAX_ANGLE, # 4
                                                        MAX_ANGLE, # 5
                                                        OMEGA_MAX, # 6
                                                        OMEGA_MAX,]),  # 9
                                                        dtype=np.float32)
        
        # Observation space:
        '''
        configuration: 16x1 array with 
        0) action: joint 1 torque               min: -MAX_TORQUE        max: MAX_TORQUE 
        1) action: joint 2 torque               min: -MAX_TORQUE        max: MAX_TORQUE
        2) x component of target                min: -VIEWPORT_SIDE/2   max: VIEWPORT_SIDE/2
        3) y component of target                min: 0                  max: VIEWPORT_SIDE/2
        4) position of end-effector in x        min: -VIEWPORT_SIDE/2   max: VIEWPORT_SIDE/2
        5) position of end-effector in y        min: 0                  max: VIEWPORT_SIDE/2
        6) velocity of end-effector in x        min: -MAX_VEL           max: MAX_VEL
        7) velocity of end-effector in y        min: -MAX_VEL           max: MAX_VEL
        8) acceleration of end-effector in x    min: -MAX_ACC           max: MAX_ACC
        9) acceleration of end-effector in y    min: -MAX_ACC           max: MAX_ACC
        10) joint 1 angle                       min: -MAX_ANGLE         max: MAX_ANGLE
        11) joint 2 angle                       min: -MAX_ANGLE         max: MAX_ANGLE
        12) joint 1 angular velocity            min: -OMEGA_MAX         max: OMEGA_MAX
        13) joint 2 angular velocity            min: -OMEGA_MAX         max: OMEGA_MAX
        14) joint 1 angular acceleration        min: -OMEGA_DOT_MAX     max: OMEGA_DOT_MAX
        15) joint 2 angular acceleration        min: -OMEGA_DOT_MAX     max: OMEGA_DOT_MAX
        '''

        
        
        self.configuration = spaces.Box(low=np.array([-1, # 0  # Scaling with MAX_TORQUE later. Bug workaround
                                                        -1, # 1 # Scaling with MAX_TORQUE later. Bug workaround
                                                        -VIEWPORT_SIDE/2, # 2
                                                        0, # 3
                                                        -VIEWPORT_SIDE/2, # 4
                                                        0, # 5
                                                        -MAX_VEL, # 6
                                                        -MAX_VEL, # 7
                                                        -MAX_ACC, # 8
                                                        -MAX_ACC, # 9
                                                        0, # 10
                                                        0, # 11
                                                        -OMEGA_MAX, # 12
                                                        -OMEGA_MAX, # 13
                                                        -OMEGA_DOT_MAX, # 14
                                                        -OMEGA_DOT_MAX,]),  # 15
                                            high=np.array([1, # 0 # Scaling with MAX_TORQUE later. Bug workaround
                                                        1, # 1 # Scaling with MAX_TORQUE later. Bug workaround
                                                        VIEWPORT_SIDE/2, # 2
                                                        VIEWPORT_SIDE/2, # 3
                                                        VIEWPORT_SIDE/2, # 4
                                                        VIEWPORT_SIDE/2, # 5
                                                        MAX_VEL, # 6
                                                        MAX_VEL, # 7
                                                        MAX_ACC, # 8
                                                        MAX_ACC, # 9
                                                        MAX_ANGLE, # 10
                                                        MAX_ANGLE, # 11
                                                        OMEGA_MAX, # 12
                                                        OMEGA_MAX, # 13
                                                        OMEGA_DOT_MAX, # 14
                                                        OMEGA_DOT_MAX,]),  # 15
                                                        dtype=np.float32)
        
        # Store the log path as an attribute of the instance
        self.findings_log_path = findings_log_path


        # For logging
        self.logging = logging
        self.create_findings_log_file()
        self.findings_log_list, self.findings_evaluation_dict = self.create_findings_list()
        self.episodes_since_instantiation = 0

        
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if self.render_mode == "human":
            self.renderer = Renderer(self.metadata)

    
    def create_findings_log_file(self):
        if self.logging:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.findings_log_path), exist_ok=True)
            # Create the file (or clear it if it already exists)
            with open(self.findings_log_path, 'w') as file:
                current_time = datetime.datetime.now()
                file.write('Log File Created on ' + str(current_time) + '\n')

    def log_message(self, message):
            if self.logging:
                # Write a message to the file
                with open(self.findings_log_path, 'a') as file:
                    file.write(message + '\n')
    
    def log_findings_eval_dict(self):
        if self.logging:
            self.eval_findings_variable()
            self.log_message("Evaluation of last 5000 episodes:")
        # Iterate over the dictionary and log each key-value pair
            for key, value in self.findings_evaluation_dict.items():
                message = f"{key}: {value}"
                self.log_message(message)


    def create_findings_list(self):
        findings_log_list = ["null"] * 5000
        findings_evaluation_dict = {
            "crashed_into_ground": 0.0,
            "crashed_into_target": 0.0,
            "target_reached": 0.0,
            "max_length_reached": 0.0
        }
        return findings_log_list, findings_evaluation_dict
    
    def update_findings_list(self, result):
        # Adding the element "ergebnis_a" to the end of the list
        self.findings_log_list.append(result)
        # Deleting the first element at the beginning of the list
        self.findings_log_list.pop(0)
    
    def eval_findings_variable(self):
        total_count = len(self.findings_log_list)
        self.findings_evaluation_dict = {
            "crashed_into_ground": self.findings_log_list.count("crashed_into_ground") / total_count,
            "crashed_into_target": self.findings_log_list.count("crashed_into_target") / total_count,
            "target_reached": self.findings_log_list.count("target_reached") / total_count,
            "max_length_reached": self.findings_log_list.count("max_length_reached") / total_count,
        }


    def reset(self, seed=None):
        # Reset the position of the manipulator. Set theta dot and theta ddot to zero.
        self.steps = 0
        self.timesteps = 0
        self.episodes_since_instantiation = 1


        # Initialize current state as 0 vector.
        self.configuration = np.zeros((16, 1)) # initialize, temporarily set invalid state
        
        # Target
        self.configuration[2:4] = self.generateObjectGoal() # set target 
        target = self.configuration[2:4]
        
        # Theta (Original configuration)
        self.configuration[10:12] = np.array([[np.pi/2], [0]]) 
        
        # update task space accordingly
        self.updateTaskSpace()

        # Direction vector
        dir_ee_target = self.getTargetVector(target)

        # Assemble observation from configuration data
        observation = np.vstack((dir_ee_target, self.configuration[6:8], self.configuration[10:12], self.configuration[12:14]))
        
        # return configuration as observation
        observation = observation.flatten() # Returns a copy, collapsed into one dimension
        

        # Empty info
        info = {}

        if self.render_mode == "human":
            time_in_ms = self.timesteps*TIMESTEP_LENGTH*1000
            self.renderer.render_frame(self.configuration, time_in_ms)

        return observation, info

    def generateObjectGoal(self):
        # returns cartesian coordinates of (object and) goal, (both) 2x1 arrays for x and y component respecively
        
        while True:
            # randomly generate manipulator configuration to ensure goal in workspcae
            theta_1 = np.random.uniform(0, MAX_ANGLE) 
            theta_2 = np.random.uniform(0, MAX_ANGLE) 

            # plug values into p_ee0 = T0_2 p_ee2
            goal1 = l_i * (np.cos(theta_1) * np.cos(theta_2) - np.sin(theta_1) * np.sin(theta_2)) + l_i * np.cos(theta_1)
            goal2 = l_i * (np.cos(theta_1) * np.sin(theta_2) + np.cos(theta_2) * np.sin(theta_1)) + l_i * np.sin(theta_1)
            goal = np.array([[goal1], [goal2]])

            # Check if the y-coordinate is positive
            if goal[1, 0] > 0:
                # return obj, goal # TODO: generate obj as well
                return goal
            
    def getTargetVector(self, target):
        p_ee = self.configuration[4:6]
        dir_ee_target = target - p_ee

        return dir_ee_target

    
    def updateJointSpace(self, dt):
        ## completes Joint Space Config for next timestep based on current Joint Space Config and Joint Torques ##

        # get Torque and Joint Angles and Velocity for curent timestep
        tau = self.configuration[0:2] # Pointer
        theta = self.configuration[10:12] # Pointer
        theta_dot = self.configuration[12:14] # Pointer
        theta_ddot = self.configuration[14:16] # Pointer

        theta_1 = theta[0][0] # Pointer
        theta_2 = theta[1][0] # Pointer
        theta_dot_1 = theta_dot[0][0] # Pointer
        theta_dot_2 = theta_dot[1][0] # Pointer

        # Calculate the common denominator
        denominator = l_i**2 * m_i * (np.cos(theta_2)**2 - 2)

        # Calculate the elements of the inverse matrix M_inv
        M_inv11 = -1 / denominator
        M_inv12 = (np.cos(theta_2) + 1) / denominator
        M_inv21 = M_inv12  # Same as M_inv12
        M_inv22 = -(2 * np.cos(theta_2) + 3) / denominator

        # Creating the matrix M_inv
        M_inv = np.array([[M_inv11, M_inv12],
                        [M_inv21, M_inv22]]) 
        
        
        # Calculating coriolis and centrifugal terms V
        V1 = -l_i**2 * m_i * theta_dot_2 * np.sin(theta_2) * (2 * theta_dot_1 + theta_dot_2) 
        V2 = l_i**2 * m_i * theta_dot_1**2 * np.sin(theta_2)

        V = np.array([[V1],
                    [V2]]) 
        
        # Calculating the elements of the gravitational vector G
        G1 = g * l_i * m_i * (np.cos(theta_1 + theta_2) + 2 * np.cos(theta_1))
        G2 = g * l_i * m_i * np.cos(theta_1 + theta_2)

        G = np.array([[G1],
                    [G2]]) 
        
        
        # The motion of the manipulator is simulated by the dynamic equation for acceleration
        theta_ddot = np.matmul(M_inv, (tau - V - G)) # Reassignment


        # Numerically integrate forward to find theta_dot and theta
        theta_dot = theta_dot + theta_ddot * dt # Reassignment
        theta = theta + theta_dot * dt + 1/2 * theta_ddot * dt**2 # Reassignment

        # write jointspace to current state
        self.configuration[10:12] = copy.deepcopy(theta)
        self.configuration[12:14] = copy.deepcopy(theta_dot)
        self.configuration[14:16] = copy.deepcopy(theta_ddot)


    def updateTaskSpace(self):
        # completes Task space from given joint space configuration
        
        theta = self.configuration[10:12] # Pointer
        theta_dot = self.configuration[12:14] # Pointer
        theta_ddot = self.configuration[14:16] # Pointer
        theta_1 = theta[0][0] # Pointer
        theta_2 = theta[1][0] # Pointer
        theta_dot_1 = theta_dot[0][0] # Pointer
        theta_dot_2 = theta_dot[1][0] # Pointer

        # Jacobian Matrix J
        J11 = -l_i * np.sin(theta_1 + theta_2) - l_i * np.sin(theta_1)
        J12 = -l_i * np.sin(theta_1 + theta_2)
        J21 = l_i * np.cos(theta_1 + theta_2) + l_i * np.cos(theta_1)
        J22 = l_i * np.cos(theta_1 + theta_2)

        J = np.array([[J11, J12],
                    [J21, J22]]) 
        
        # Time derivative of the Jacobian
        # J_dot11 = -l_i * theta_dot_1 * np.cos(theta_1) - l_i * np.cos(theta_1 + theta_2) * (theta_dot_1 + theta_dot_2)
        # J_dot12 = -l_i * np.cos(theta_1 + theta_2) * (theta_dot_1 + theta_dot_2)
        # J_dot21 = -l_i * theta_dot_1 * np.sin(theta_1) - l_i * np.sin(theta_1 + theta_2) * (theta_dot_1 + theta_dot_2)
        # J_dot22 = -l_i * np.sin(theta_1 + theta_2) * (theta_dot_1 + theta_dot_2)

        # Creating the matrix J_dot
        # J_dot = np.array([[J_dot11, J_dot12],
                        # [J_dot21, J_dot22]]) 

        # calculate position of the end-effector 
        p_ee_x = l_i * (np.cos(theta_1) * np.cos(theta_2) - np.sin(theta_1) * np.sin(theta_2)) + l_i * np.cos(theta_1)
        p_ee_y = l_i * (np.cos(theta_1) * np.sin(theta_2) + np.cos(theta_2) * np.sin(theta_1)) + l_i * np.sin(theta_1)
        p_ee = np.array([[p_ee_x], [p_ee_y]])

        # calculate velocity of end-effector
        vel_ee = np.dot(J, theta_dot)

        # calculate acceleration of end-effector
        # acc_ee = np.dot(J_dot, theta_dot) + np.dot(J, theta_ddot)

        # self.configuration = np.vstack((action, target, p_ee, vel_ee, acc_ee, theta, theta_dot, theta_ddot))
        self.configuration[4:6] = copy.deepcopy(p_ee) 
        self.configuration[6:8] = copy.deepcopy(vel_ee) 
        # self.configuration[8:10] = copy.deepcopy(acc_ee)
    
    def clipConfigToObservationSpace(self):

        '''
        configuration: 16x1 array with 
        0) action: joint 1 torque               min: -MAX_TORQUE        max: MAX_TORQUE             DO NOT ALTER
        1) action: joint 2 torque               min: -MAX_TORQUE        max: MAX_TORQUE             DO NOT ALTER
        2) x component of target                min: -VIEWPORT_SIDE/2   max: VIEWPORT_SIDE/2        DO NOT ALTER
        3) y component of target                min: 0                  max: VIEWPORT_SIDE/2        DO NOT ALTER
        4) position of end-effector in x        min: -VIEWPORT_SIDE/2   max: VIEWPORT_SIDE/2        DO NOT ALTER
        5) position of end-effector in y        min: 0                  max: VIEWPORT_SIDE/2        DO NOT ALTER
        6) velocity of end-effector in x        min: -MAX_VEL           max: MAX_VEL                DO NOT ALTER
        7) velocity of end-effector in y        min: -MAX_VEL           max: MAX_VEL                DO NOT ALTER
        8) acceleration of end-effector in x    min: -MAX_ACC           max: MAX_ACC                DO NOT ALTER
        9) acceleration of end-effector in y    min: -MAX_ACC           max: MAX_ACC                DO NOT ALTER
        10) joint 1 angle                       min: -MAX_ANGLE         max: MAX_ANGLE              WRAP
        11) joint 2 angle                       min: -MAX_ANGLE         max: MAX_ANGLE              WRAP
        12) joint 1 angular velocity            min: -OMEGA_MAX         max: OMEGA_MAX              CLIP
        13) joint 2 angular velocity            min: -OMEGA_MAX         max: OMEGA_MAX              CLIP
        14) joint 1 angular acceleration        min: -OMEGA_DOT_MAX     max: OMEGA_DOT_MAX          CLIP
        15) joint 2 angular acceleration        min: -OMEGA_DOT_MAX     max: OMEGA_DOT_MAX          CLIP
        '''

        # Wrap Theta
        self.configuration[10:12] = (self.configuration[10:12])%(MAX_ANGLE)

        # Clip Angular Velocity
        self.configuration[12:14] = np.clip(self.configuration[12:14], np.array([[-OMEGA_MAX],[-OMEGA_MAX]]), np.array([[OMEGA_MAX],[OMEGA_MAX]]))

        # Clip Angular Acceleration
        self.configuration[14:16] = np.clip(self.configuration[14:16], np.array([[-OMEGA_DOT_MAX],[-OMEGA_DOT_MAX]]), np.array([[OMEGA_DOT_MAX],[OMEGA_DOT_MAX]]))

        

    def getDefaultReward(self):
        # returns the default reward in [-0.5, 0] when neither crashed nor target reached as linear function of distance to target.
        
        # calculate euclidean distance to target
        p_ee = self.configuration[4:6] # Pointer
        target = self.configuration[2:4] # Pointer
        dist = np.linalg.norm(target - p_ee)

        # calculate reward
        # reward = -dist/(4*l_i)
        reward = CRASHED_GROUND_REWARD/MAX_STEPS_PER_EPISODE * dist/VIEWPORT_SIDE
        
        return reward
    
    
    def evalStep(self):
        # returns true if goal is reached at suficiently low speed, false otherwise

        
        target = self.configuration[2:4] # Pointer
        p_ee = self.configuration[4:6] # Pointer
        vel_ee = self.configuration[6:8] # Pointer
        theta = self.configuration[10:12] # Pointers

        # calculate y component of link 1
        joint_1_y = l_i * np.sin(theta[0][0]) # if both have positive y-component return false
        
        # Evaluate if Robot Crashed into ground (if both have positive y-component return false)
        if joint_1_y < 0 or p_ee[1][0] < 0:
            if self.render_mode == "human":
                print("Robot crashed into ground")
                
            return "crashedIntoGround"
        
        
        # Evaluate if Robot Reached or Crashed into Target
        if np.linalg.norm(p_ee - target) <= TOL_DIST:
            # End-effector near target
            if np.linalg.norm(vel_ee) <= TOL_VEL:
                if self.render_mode == "human":
                    print("Target Reached")
                    
                return "targetReached" # Reached target without crash 
            else:
                if self.render_mode == "human":
                    print("Robot end-effector speed too high when reaching target")
                    
                return "targetCrashed" # Crashed into target
        else:
            return "ongoing" # Did not reach target
        

    
    def step(self, action):

        # Reshape action
        action = action.reshape((2, 1))
        action = action * MAX_TORQUE
        
        # Take the action
        self.configuration[0:2] = copy.deepcopy(action) # Set torque to joints 1 and 2 

        # Calculate effect of the taken action
        for _ in range(TIMESTEPS_PER_ACTION): 
            self.updateJointSpace(TIMESTEP_LENGTH) # Robotics: Actions be taken every 50 milliseconds.
            self.clipConfigToObservationSpace()
            self.timesteps += 1
            
            if self.render_mode == "human":
                time_in_ms = self.timesteps*TIMESTEP_LENGTH*1000
                self.updateTaskSpace()
                self.renderer.render_frame(self.configuration, time_in_ms)
        
        self.updateTaskSpace()

        # Assemble 
        target = self.configuration[2:4]

        dir_ee_target = self.getTargetVector(target)
        
        observation = np.vstack((dir_ee_target, self.configuration[6:8],self.configuration[10:12], self.configuration[12:14]))

        # Return observation
        observation = observation.flatten() # Return copy of configuration collapsed to 1D

        stepResult = self.evalStep()

        # Return reward and check if terminated
        if stepResult == "ongoing": # Neither crashed nor target reached 
            
            reward = self.getDefaultReward()
            terminated = False
            info = {"state": "ongoing"}

        elif stepResult ==  "crashedIntoGround": # Crash into ground
            reward = CRASHED_GROUND_REWARD
            terminated = True
            info = {"state": "crashed_into_ground"}
            self.update_findings_list(info["state"])
        
        elif stepResult == "targetCrashed":  # Goal reached
            reward = CRASHED_TARGET_REWARD
            terminated = True
            info = {"state": "crashed_into_target"}
            self.update_findings_list(info["state"])
            
        else: # Target Reached
            reward = REACHED_TARGET_REWARD
            terminated = True
            info = {"state": "target_reached"}
            self.update_findings_list(info["state"])
            
        # Episodes are truncated after 10s = 1000 timesteps = 200 decisions
        if self.steps >= MAX_STEPS_PER_EPISODE:
            truncated = True
            info = {"state": "max_length_reached"}
            self.update_findings_list(info["state"])
        else:
            truncated = False


        self.steps += 1

        if self.render_mode == "human":
            time_in_ms = self.timesteps*TIMESTEP_LENGTH*1000
            self.renderer.render_frame(self.configuration, time_in_ms)

        return observation, reward, terminated, truncated, info

    
