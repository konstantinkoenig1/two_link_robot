import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

def printCurrentState(myClass):

    annotations = np.array([["0) action: joint 1 torque:            "],
                            ["1) action: joint 2 torque:            "],
                            ["2) x component of target:             "],
                            ["3) y component of target:             "],
                            ["4) position of end-effector in x:     "],
                            ["5) position of end-effector in y:     "],
                            ["6) velocity of end-effector in x:     "],
                            ["7) velocity of end-effector in y:     "],
                            ["8) acceleration of end-effector in x: "],
                            ["9) acceleration of end-effector in y: "],
                            ["10) joint 1 angle:                    "],
                            ["11) joint 2 angle:                    "],
                            ["12) joint 1 angular velocity:         "],
                            ["13) joint 2 angular velocity:         "],
                            ["14) joint 1 angular acceleration:     "],
                            ["15) joint 2 angular acceleration:     "]])                                          
    '''
        current_state: 16x1 array with 
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
    

    print(np.hstack((annotations, myClass.current_state)))




