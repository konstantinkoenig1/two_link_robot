
import copy
import sys
import os

# Add  parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from definitions import *

import pygame
from pygame.locals import *





WINDOW_WIDTH = 2100
WINDOW_HEIGHT = 1200
WHITE = (255, 255, 255)
BLUE = (26, 26, 87)
BLACK = (0, 0, 0)
RED = (255, 0, 0) 
GREEN = (0, 110, 20)
SCALE = (WINDOW_WIDTH-100)/2



# Frame Zero

FRAME_0_ORG = np.array([[(WINDOW_WIDTH-100)/2 + 50], [(WINDOW_HEIGHT-100)+50]])
OFFSET = FRAME_0_ORG
OFFSET_X = OFFSET[0][0]
OFFSET_Y = OFFSET[1][0]



class Renderer():

    def __init__(self, metadata):
        self.window = None
        self.clock = None
        self.metadata = metadata
        print("Rendering initialized")
    
    
    def render_frame(self, current_state, time):

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption("Fixed Base Robot")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        canvas.fill(WHITE)

        # Get information from current_state
        theta_1 = copy.deepcopy(current_state[10]) 

        # # Get position of joint 1
        # joint_1 = np.array([[l_i * np.cos(theta_1)],[l_i * np.sin(theta_1)]])
        # joint_1[1][0] = - joint_1[1][0] # y-Axis inverted in Pygame window
        # joint_1 = joint_1*SCALE
        # joint_1 = joint_1.astype(int) + OFFSET


        # # Get position of end effector
        # p_ee = copy.deepcopy(current_state[4:6])
        # p_ee[1][0] = - p_ee[1][0] # y-Axis inverted in Pygame window
        # p_ee = p_ee * SCALE 
        # p_ee = p_ee.astype(int) + OFFSET

        # Get position of target
        target = copy.deepcopy(current_state[2:4]) 
        target[1][0] = - target[1][0] # y-Axis inverted in Pygame window
        target = target * SCALE
        target = target.astype(int) + OFFSET

        joint_1 = np.array([[l_i * np.cos(theta_1)],[l_i * np.sin(theta_1)]])
        joint_1 = joint_1*SCALE
        # joint_1[1][0] = - joint_1[1][0] # y-Axis inverted in Pygame window
        joint_1 = joint_1.flatten()
        # joint_1 = joint_1.astype(int) + OFFSET
        joint_1 = np.array([[OFFSET_X + joint_1[0]],[OFFSET_Y-joint_1[1]]])


        # Get position of end effector
        p_ee = copy.deepcopy(current_state[4:6])
        p_ee = p_ee * SCALE 
        # p_ee[1][0] = - p_ee[1][0] # y-Axis inverted in Pygame window
        p_ee = p_ee.flatten()
        # p_ee = p_ee.astype(int) + OFFSET
        p_ee = np.array([[OFFSET_X + p_ee[0]],[OFFSET_Y-p_ee[1]]])

        # Draw Boundary of task space
        boundary = Rect(50, 50, WINDOW_WIDTH-100, WINDOW_HEIGHT-100)
        pygame.draw.rect(canvas, RED, boundary, 4)

        # Target
        pygame.draw.circle(canvas, GREEN, (int(target[0][0]) , int(target[1][0])), 20)
        pygame.draw.circle(canvas, GREEN, (int(target[0][0]) , int(target[1][0])), 40, width=10)


        # Draw Link 1 to the canvas

        pygame.draw.line(
                canvas,
                BLUE,
                (FRAME_0_ORG[0][0], FRAME_0_ORG[1][0]),
                (int(joint_1[0][0]) , int(joint_1[1][0])),
                width=20 
                )
        
        # # Link 2
        pygame.draw.line(
                canvas,
                BLUE,
                (int(joint_1[0][0]) , int(joint_1[1][0])),
                (int(p_ee[0][0]) , int(p_ee[1][0])),
                width=20 
                )
        # Draw base of manipulator
        base_1 = Rect(FRAME_0_ORG[0][0] - 30, FRAME_0_ORG[1][0], 60,10)
        pygame.draw.rect(canvas, BLACK, base_1)
        pygame.draw.circle(canvas, BLACK, (FRAME_0_ORG[0][0], FRAME_0_ORG[1][0]), 30, draw_top_right=True, draw_top_left=True, draw_bottom_left=False, draw_bottom_right=False)

        # # Joint 2
        pygame.draw.circle(canvas, BLACK, (int(joint_1[0][0]) , int(joint_1[1][0])), 30)

        # # end-effector
        pygame.draw.circle(canvas, BLACK, (int(p_ee[0][0]) , int(p_ee[1][0])), 30)

        # # Display Text
        timeInfo = "time: "
        timeInfo += str(int(time))
        timeInfo += " ms"
        font = pygame.font.Font('freesansbold.ttf', 32)
        text = font.render(timeInfo, True, BLACK)
        

    
        # The following line copies our drawings from `canvas` to the visible window
        self.window.blit(canvas, canvas.get_rect())
        self.window.blit(text, (WINDOW_WIDTH-400,100))
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(self.metadata["render_fps"])
        
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
