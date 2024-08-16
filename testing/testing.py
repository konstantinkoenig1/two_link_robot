# Load an instance of fixed_base_robot_env
import sys
import os
import copy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from definitions import *

from testing_funcitions import *
from envs.fixed_base_robot_env import FixedBaseRobotEnv

# Unit tests for all methods in environment

# Test __init__ method
myEnv = FixedBaseRobotEnv()
# print(dir(myEnv))
print("myEnv.observation_space:")
print(myEnv.observation_space)
print("myEnv.action_space:")
print(myEnv.action_space)
print("myEnv.render_mode:")
print(myEnv.render_mode)
print("myEnv.action_space.sample():")
print(myEnv.action_space.sample())


# Test reset() method
# myEnv = FixedBaseRobotEnv()
# observation, info = myEnv.reset()
# printCurrentState(myEnv)

# Test updateTaskSpace() method: p_ee
# myEnv = FixedBaseRobotEnv()
# myEnv.reset()
# desired_theta = np.array([[0], [np.pi/2]])
# myEnv.current_state[10:12] = copy.deepcopy(desired_theta)
# myEnv.updateTaskSpace()
# printCurrentState(myEnv)
# p_ee test successful for [np.pi/2],[0] | [0],[0] | [np.pi],[0] | [0], [np.pi]  |  [0], [np.pi/2]

# Test updateTaskSpace() method: vel_ee
# myEnv = FixedBaseRobotEnv()
# myEnv.reset()
# desired_theta = np.array([[2], [-np.pi/4]])
# desired_theta_dot = np.array([[-2],[3]])
# myEnv.current_state[10:12] = copy.deepcopy(desired_theta)
# myEnv.current_state[12:14] = copy.deepcopy(desired_theta_dot)
# myEnv.updateTaskSpace()
# printCurrentState(myEnv)
# vel_ee test successful for theta [np.pi/2],[0] theta_dot [2*np.pi],[2*np.pi]  |  theta [2], [-np.pi/4] theta_dot   [-2],[3]

# Test updateTaskSpace() method: acc_ee
# myEnv = FixedBaseRobotEnv()
# myEnv.reset()
# desired_theta = np.array([[np.pi/2],[0]])
# desired_theta_dot = np.array([[0],[0]])
# desired_theta_ddot = np.array([[0],[0]])
# myEnv.current_state[10:12] = copy.deepcopy(desired_theta)
# myEnv.current_state[12:14] = copy.deepcopy(desired_theta_dot)
# myEnv.current_state[14:16] = copy.deepcopy(desired_theta_ddot)
# myEnv.updateTaskSpace()
# printCurrentState(myEnv)
# acc_ee test successful for theta [np.pi/2],[0] theta_dot [2*np.pi],[2*np.pi] ddot [0],[0] |  theta [2], [-np.pi/4] theta_dot [-2],[3] ddot [10],[-20]

# updateTaskSpace method test successful


# test updateJointSpace
#-> Input: get Torque and Joint Angles and Velocity for curent timestep
#-> tau = self.current_state[0:2] # Pointer
#-> theta = self.current_state[10:12] # Pointer
#-> theta_dot = self.current_state[12:14] # Pointer
#-> theta_ddot = self.current_state[14:16] # Pointer


# Test theta_ddot, # Test theta_dot, # Test theta
# myEnv = FixedBaseRobotEnv()
# myEnv.reset()
# desired_theta = np.array([[np.pi-0.05*np.pi], [-0.05*np.pi]])
# desired_theta_dot = np.array([[0], [0]])
# desired_tau = np.array([[-250], [-78]])
# myEnv.current_state[10:12] = copy.deepcopy(desired_theta)
# myEnv.current_state[12:14] = copy.deepcopy(desired_theta_dot)
# myEnv.current_state[0:2] = copy.deepcopy(desired_tau)
# print("before:")
# printCurrentState(myEnv)
# myEnv.updateJointSpace(TIMESTEP_LENGTH)
# myEnv.updateTaskSpace()
# print("after:")
# printCurrentState(myEnv)

# ''' Success for: input: 
# theta: [np.pi/2],[0] | theta_dot: [0],[0]  | tau: [20],[20]
# theta: [0.8*np.pi], [-0.2*np.pi]  | theta_dot: [-2*np.pi], [3*np.pi]   |    tau: [-15], [12]  
# '''



# Test clipping
# myEnv = FixedBaseRobotEnv()
# myEnv.reset()
# myEnv.updateTaskSpace()
# myEnv.current_state[10:12] = np.array([[12*np.pi], [-3*np.pi]])  # theta
# myEnv.current_state[12:14] = np.array([[1.4], [-1.5]])   # theta_dot
# myEnv.current_state[14:16] = np.array([[1.4], [-1.5] ])   # theta_dot
# myEnv.current_state[10:12] = (myEnv.current_state[10:12]+np.pi)%(2*np.pi) - np.pi
# myEnv.clipObservationSpace()
# printCurrentState(myEnv)

'''test successfull for theta [12*np.pi], [-3*np.pi] '''
'''test successfull for theta_dot [1000], [1000]  |   [-1000], [1000]  |  [1000], [-1000]    |  [-1000], [-1000] |   [1.4], [-1.5] '''
'''test successfull for theta_ddot [1000], [1000]  |   [-1000], [1000]  |  [1000], [-1000]    |  [-1000], [-1000] |   [1.4], [-1.5] '''
# my_vec = np.array([[0],[0]]) + np.array([[np.pi],[np.pi]])
# print(my_vec)

# print((12*np.pi + np.pi)%(2*np.pi)-np.pi)

# # Investigate length difference bug: Measure distance between {0}, {1} and {1}, {2} for config [pi/2][0]
# myEnv = FixedBaseRobotEnv()
# myEnv.reset()
# desired_theta = np.array([[np.pi/2], [0]])
# myEnv.current_state[10:12] = copy.deepcopy(desired_theta)
# myEnv.updateTaskSpace()

# printCurrentState(myEnv)

# joint_1 = np.array([[l_i*np.cos(desired_theta[0][0])],[l_i*np.sin(desired_theta[0][0])]])
# p_ee = myEnv.current_state[4:6]

# # 1.: in actual cartesian space
# print("length link 1 cartesian space:")
# print(np.linalg.norm(joint_1))

# print("length link 2 cartesian space:")
# print(np.linalg.norm(joint_1-p_ee))

# # 2.: in pixels in rendering
# # Get position of joint 1
# WINDOW_WIDTH = 2100
# WINDOW_HEIGHT = 1200
# WHITE = (255, 255, 255)
# BLUE = (26, 26, 87)
# BLACK = (0, 0, 0)
# RED = (255, 0, 0) 
# GREEN = (0, 110, 20)
# SCALE = (WINDOW_WIDTH-100)/2




# # Frame Zero

# FRAME_0_ORG = np.array([[(WINDOW_WIDTH-100)/2 + 50], [(WINDOW_HEIGHT-100)+50]])
# OFFSET = FRAME_0_ORG
# OFFSET_X = OFFSET[0][0]
# OFFSET_Y = OFFSET[1][0]
# print("Frame 0 origin")
# print(OFFSET)

# # Get information from current_state
# theta_1 = copy.deepcopy(myEnv.current_state[10]) 

# joint_1 = np.array([[l_i * np.cos(theta_1)],[l_i * np.sin(theta_1)]])
# # joint_1[1][0] = - joint_1[1][0] # y-Axis inverted in Pygame window
# joint_1 = joint_1*SCALE
# joint_1 = joint_1.flatten()
# # joint_1 = joint_1.astype(int) + OFFSET
# joint_1 = np.array([[OFFSET_X + joint_1[0]],[OFFSET_Y-joint_1[1]]])


# # Get position of end effector
# p_ee = copy.deepcopy(myEnv.current_state[4:6])
# # p_ee[1][0] = - p_ee[1][0] # y-Axis inverted in Pygame window
# p_ee = p_ee * SCALE 
# p_ee = p_ee.flatten()
# # p_ee = p_ee.astype(int) + OFFSET
# p_ee = np.array([[OFFSET_X + p_ee[0]],[OFFSET_Y-p_ee[1]]])


# print("joint 1 render space:")
# print(joint_1)

# print("p_ee render space:")
# print(p_ee)

# print("nominal link length render space")
# print(0.5*SCALE)










# myEnv = FixedBaseRobotEnv()
# myEnv.reset()
# print("************* Current State: *********************")
# printCurrentState(myEnv)


# reset the environment



# check position of end-effector and 1st joint


