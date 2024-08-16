import numpy as np

## DEFINE CONSTANTS ##

# Natural Constants 

g = 9.81    # N/kg

# Specifications of agent 

LINK_LENGTH = 0.5   # in m. 1/4 * viewport sidelength 
LINK_WIDTH = 1/8 * LINK_LENGTH
RHO_STEEL = 7850    # Density of steel in kg/m^3
I_zz = (1/24576) * RHO_STEEL * (LINK_LENGTH**3) * (LINK_LENGTH**2)      # z-component of inertia tensor
INERTIA_TENSOR = np.array([[0, 0, 0],
                   [0, 0, 0],
                   [0, 0, I_zz]])
LINK_MASS = RHO_STEEL * LINK_LENGTH * LINK_WIDTH**2

# Action Space 

MAX_TORQUE = 250 # N


# Observation Space  # Double realistic assumptions to avoid clipping

VIEWPORT_SIDE = 2   # Field of 2x2 meters
MAX_VEL = 10    # 6 # in m/s # Avoid clipping by choosing high velocities that can hardly be reached with given joint torques.
MAX_ACC = 8       # 4 # 2x max acc for max torque applied to joint 1 only.
MAX_ANGLE = 2*np.pi   # np.pi # in rad. Joint angle anticlockwise 
OMEGA_MAX = 3*np.pi # in rad/s. That's 1.5 revolution per second. That's enough.
OMEGA_DOT_MAX = 3*np.pi    # That's necessary to meet max torque application. Accelerating to above ang velocity in 1s

# Reward Design 
    # Idea: negative reward with r = -(speed-MAX_SPEED)*x if threshold passed.
CRASHED_GROUND_REWARD = -10
CRASHED_TARGET_REWARD = 0
REACHED_TARGET_REWARD = 0.5


# Parameters for computation 

TIMESTEP_LENGTH = 1/100                 # This is exactly 0.01 seconds. Physics timestep length.
TIMESTEPS_PER_ACTION = 5                # One action every 5 timesteps. That's 20 actions per second or one action every 50ms
EP_LENGTH = 10                          # Truncate episode after 10s
MAX_STEPS_PER_EPISODE = EP_LENGTH/(TIMESTEP_LENGTH*TIMESTEPS_PER_ACTION)
TOL_DIST = 0.05                         # Max distance from target to end episode
TOL_VEL = 0.25 #0.02                          # Max tolerated velocity near target


# INTRODUCE SYNONYMS FOR VARIABLES TO SHORTEN FORMULAS

l_i = LINK_LENGTH
m_i = LINK_MASS


# For logging
FINDINGS_LOG_PATH = "models_and_logs/PPO_1/findings.txt"
