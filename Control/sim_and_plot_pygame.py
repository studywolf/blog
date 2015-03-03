'''
Copyright (C) 2015 Travis DeWolf

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np
import pygame
import pygame.locals
import sys

class ArmPart:
    """
    A class for storing relevant arm segment information.
    """

    def __init__(self, pic, 
                 flip_image=False, 
                 r_scale=1.0):

        self.base = pygame.image.load(pic)
        
        self.length = self.base.get_rect()[2]
        self.width = self.base.get_rect()[3]

        rect = self.base.get_rect()
        self.r = np.sqrt((rect[2] / 2. * r_scale)**2 + (rect[3] / 2.)**2) 

        if flip_image == True:
            self.base = pygame.transform.flip(self.base, 0, 1)

        self.rotation = 0

    def rotate(self):
        """
        Rotates and re-centers the arm segment.
        """
        # rotate our image 
        image = pygame.transform.rotate(self.base, np.degrees(self.rotation))
        # set it up so that we're rotating around the center point
        rect = image.get_rect()
        # reset the center
        rect.center = np.zeros(2)

        return image, rect


class Runner:
    """
    A class for drawing the arm simulation using PyGame
    """
    def __init__(self, title='', dt=1e-4, control_steps=10, 
                       display_steps=100, t_target=1.0, 
                       box=[-1,1,-1,1], rotate=0.0,
                       control_type='', trajectory=None,
                       infinite_trail=False, mouse_control=False):
        self.dt = dt
        self.control_steps = control_steps
        self.display_steps = display_steps
        self.target_steps = int(t_target/float(dt*display_steps))
        self.trajectory = trajectory

        self.box = box 
        self.control_type = control_type 
        self.infinite_trail = infinite_trail
        self.mouse_control = mouse_control
        self.rotate = rotate
        self.title = title

        self.sim_step = 0
        self.trail_index = 0
        self.pen_lifted = False

        self.width = 642
        self.height = 600
        self.base_offset = np.array([self.width / 2.0, self.height*.9])

        # self.figure_width = 4.75
        # self.figure_height = 4.75
        

    def run(self, arm, control_shell, video=None, video_time=None):

        self.arm = arm
        self.shell = control_shell

        arm1 = ArmPart('img/three_link/svgupperarm2.png', 
                        flip_image=True, 
                        r_scale = .5)
        arm2 = ArmPart('img/three_link/svgforearm2.png', 
                        flip_image=True, 
                        r_scale = .6)
        arm3 = ArmPart('img/three_link/svghand2.png', 
                        flip_image=True, 
                        r_scale= 1)

        background = pygame.image.load('img/whiteboard.jpg')

        pygame.init()
        
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.title)

        # constants
        fps = 20 # frames per second
        # colors
        white = (255, 255, 255)
        red = (255, 0, 0)
        black = (0, 0, 0)
        arm_color = (75, 75, 75)
        line_color = (50, 50, 50, 200)

        # scaling_term = np.array([self.height / self.figure_width,
        #                          self.height / self.figure_height])
        scaling_term = np.ones(2) * 105
        # upperarm constants 
        upperarm_length = self.arm.L[0] * scaling_term[0]
        upperarm_width = .15 * scaling_term[1]
        r_upperarm = np.sqrt((upperarm_length / 2.0) ** 2 + \
                             (upperarm_width / 2.0) ** 2)
        # forearm constants 
        forearm_length = self.arm.L[1] * scaling_term[0]
        forearm_width = .15 * scaling_term[1]
        r_forearm = np.sqrt((forearm_length / 2.0) ** 2 + \
                            (forearm_width / 2.0) ** 2)
        # hand constants 
        hand_length = self.arm.L[2] * scaling_term[0]
        hand_width = .15 * scaling_term[1]
        r_hand = np.sqrt((hand_length / 2.0) ** 2 + \
                         (hand_width / 2.0) ** 2)

        fpsClock = pygame.time.Clock()

        # constants for magnify plotting
        magnify_scale = 1.75
        magnify_window_size = np.array([200, 200])
        first_target = np.array([321, 330])
        magnify_offset = first_target * magnify_scale - magnify_window_size / 2

        # setup pen trail and appending functions
        self.trail_data = []
        def pen_down1():
            self.pen_lifted = False
            x,y = self.arm.position()
            x = int( x[-1] * scaling_term[0] + self.base_offset[0]) 
            y = int(-y[-1] * scaling_term[1] + self.base_offset[1])
            self.trail_data.append([[x,y],[x,y]])

            self.trail_data[self.trail_index].append(points[3])
            pen_down = pen_down2
        def pen_down2():
            self.trail_data[self.trail_index].append(points[3])
        pen_down = pen_down1


        # enter simulation / plotting loop
        while True: 

            self.display.fill(white)

            self.target = self.shell.controller.target * np.array([1, -1]) * \
                            scaling_term + self.base_offset
           
            # before drawing
            for j in range(self.display_steps):            
                # update control signal
                if self.sim_step % self.control_steps == 0 or \
                    'tau' not in locals():
                        tau = self.shell.control(self.arm)
                # apply control signal and simulate
                self.arm.apply_torque(u=tau, dt=self.dt)

                self.sim_step +=1

            # get (x,y) positions of the joints
            x,y = self.arm.position()
            points = [(int(a * scaling_term[0] + self.base_offset[0]), 
                       int(-b * scaling_term[1] + self.base_offset[1])) 
                       for a,b in zip(x,y)]

            arm1.rotation = self.arm.q[0] + np.pi
            arm2.rotation = self.arm.q[1] + arm1.rotation
            arm3.rotation = self.arm.q[2] + arm2.rotation
            arm1_image, arm1_rect = arm1.rotate()
            arm2_image, arm2_rect = arm2.rotate()
            arm3_image, arm3_rect = arm3.rotate()

            # center the joint of the arm at the offset location
            arm1_rect.center += np.asarray(points[0])
            arm1_rect.center += np.array([
                                    -np.cos(arm1.rotation) * arm1.r,
                                    np.sin(arm1.rotation) * arm1.r])

            # center the joint of the arm at the offset location
            arm2_rect.center += np.asarray(points[1])
            arm2_rect.center += np.array([
                                    -np.cos(arm2.rotation) * arm2.r,
                                    np.sin(arm2.rotation) * arm2.r])

            # center the joint of the arm at the offset location
            arm3_rect.center += np.asarray(points[2])
            arm3_rect.center += np.array([
                                    -np.cos(arm3.rotation) * (arm3.length / 2.0 - 10),
                                    np.sin(arm3.rotation) * (arm3.length / 2.0 - 10)])

            # transparent upperarm line
            rotation_upperarm = arm1.rotation - np.pi
            line_upperarm = pygame.Surface((upperarm_length, upperarm_width),
                    pygame.SRCALPHA, 32)
            line_upperarm.fill(line_color)
            line_upperarm = pygame.transform.rotate(line_upperarm, np.degrees(rotation_upperarm))
            # because when rotated the image gets padded we need to offset from center
            rect_upperarm = line_upperarm.get_rect()
            line_upperarm_x = points[0][0] - rect_upperarm.width / 2.0 + np.cos(rotation_upperarm) * r_upperarm
            line_upperarm_y = points[0][1] - rect_upperarm.height / 2.0 - np.sin(rotation_upperarm) * r_upperarm

            # transparent forearm line
            rotation_forearm = arm2.rotation - np.pi
            line_forearm = pygame.Surface((forearm_length, forearm_width),
                    pygame.SRCALPHA, 32)
            line_forearm.fill(line_color)
            line_forearm = pygame.transform.rotate(line_forearm, np.degrees(rotation_forearm))
            # because when rotated the image gets padded we need to offset from center
            rect_forearm = line_forearm.get_rect()
            line_forearm_x = points[1][0] - rect_forearm.width / 2.0 + np.cos(rotation_forearm) * r_forearm
            line_forearm_y = points[1][1] - rect_forearm.height / 2.0 - np.sin(rotation_forearm) * r_forearm

            # transparent hand line
            rotation_hand = arm3.rotation - np.pi
            line_hand = pygame.Surface((hand_length, hand_width),
                    pygame.SRCALPHA, 32)
            line_hand.fill(line_color)
            line_hand = pygame.transform.rotate(line_hand, np.degrees(rotation_hand))
            # because when rotated the image gets padded we need to offset from center
            rect_hand = line_hand.get_rect()
            line_hand_x = points[2][0] - rect_hand.width / 2.0 + np.cos(rotation_hand) * r_hand
            line_hand_y = points[2][1] - rect_hand.height / 2.0 - np.sin(rotation_hand) * r_hand

            # update trail

            if self.shell.pen_down is True:
                pen_down()
            elif self.shell.pen_down is False and self.pen_lifted is False:
                pen_down = pen_down1
                self.pen_lifted = True

            # draw things! 
            self.display.blit(background, (0,0)) # draw on the background

            for trail in self.trail_data:
                pygame.draw.aalines(self.display, black, False, trail, True)

            # draw arm images
            self.display.blit(arm1_image, arm1_rect)
            self.display.blit(arm2_image, arm2_rect)
            self.display.blit(arm3_image, arm3_rect)

            # draw original arm lines 
            # pygame.draw.lines(self.display, arm_color, False, points, 18)

            # draw transparent arm lines
            self.display.blit(line_upperarm, (line_upperarm_x, line_upperarm_y))
            self.display.blit(line_forearm, (line_forearm_x, line_forearm_y))
            self.display.blit(line_hand, (line_hand_x, line_hand_y))

            # draw circles at shoulder
            pygame.draw.circle(self.display, black, points[0], 30)
            pygame.draw.circle(self.display, arm_color, points[0], 12)

            # draw circles at elbow 
            pygame.draw.circle(self.display, black, points[1], 20)
            pygame.draw.circle(self.display, arm_color, points[1], 7)

            # draw circles at wrist
            pygame.draw.circle(self.display, black, points[2], 15)
            pygame.draw.circle(self.display, arm_color, points[2], 5)

            # draw target
            pygame.draw.circle(self.display, red, [int(val) for val in self.target], 10)

            # now display magnification of drawing area
            magnify = pygame.Surface(magnify_window_size)
            magnify.blit(background, (-200,-200)) # draw on the background
            # magnify.fill(white)
            # put a border on it
            pygame.draw.rect(magnify, black, (2.5, 2.5, 195, 195), 1)
            # now we need to rescale the trajectory and targets
            # using the first target position, which I know to be the 
            # desired center of the magnify area
            for trail in self.trail_data:
                pygame.draw.aalines(magnify, black, False, 
                        np.asarray(trail) * magnify_scale - magnify_offset, True)
            pygame.draw.circle(magnify, red, 
                    np.array(self.target * magnify_scale - magnify_offset, 
                        dtype=int), 5)

            # now draw the target and hand line 
            self.display.blit(magnify, (32, 45))

            # check for quit
            for event in pygame.event.get():
                if event.type == pygame.locals.QUIT:
                    pygame.quit()
                    sys.exit()

            pygame.display.update()
            fpsClock.tick(fps)

    def show(self):
        try:
            plt.show()
        except AttributeError:
            pass
