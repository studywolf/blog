import numpy as np
import pygame
import pygame.locals

from armpart import ArmPart

black = (0, 0, 0)
white = (255, 255, 255)
arm_color = (50, 50, 50, 200) # fourth value specifies transparency

pygame.init()

width = 750
height = 750
display = pygame.display.set_mode((width, height))
fpsClock = pygame.time.Clock()

upperarm = ArmPart('upperarm.png', scale=.7)
forearm = ArmPart('forearm.png', scale=.8)
hand = ArmPart('hand.png', scale=1.0)

line_width = 15
line_upperarm = pygame.Surface((upperarm.scale, line_width), pygame.SRCALPHA, 32)
line_forearm = pygame.Surface((forearm.scale, line_width), pygame.SRCALPHA, 32)
line_hand = pygame.Surface((hand.scale, line_width), pygame.SRCALPHA, 32)

line_upperarm.fill(arm_color)
line_forearm.fill(arm_color)
line_hand.fill(arm_color)

origin = (width / 2, height / 2)

def transform(rect, base, arm_part):
    rect.center += np.asarray(base)
    rect.center += np.array([np.cos(arm_part.rotation) * arm_part.offset,
                            -np.sin(arm_part.rotation) * arm_part.offset])
def transform_lines(rect, base, arm_part):
    transform(rect, base, arm_part)
    rect.center += np.array([-rect.width / 2.0, -rect.height / 2.0])

while 1:

    display.fill(white)

    # rotate our joints
    ua_image, ua_rect = upperarm.rotate(.03) 
    fa_image, fa_rect = forearm.rotate(-.02) 
    h_image, h_rect = hand.rotate(.03) 

    # generate (x,y) positions of each of the joints
    joints_x = np.cumsum([0, 
                          upperarm.scale * np.cos(upperarm.rotation),
                          forearm.scale * np.cos(forearm.rotation),
                          hand.length * np.cos(hand.rotation)]) + origin[0]
    joints_y = np.cumsum([0, 
                          upperarm.scale * np.sin(upperarm.rotation),
                          forearm.scale * np.sin(forearm.rotation), 
                          hand.length * np.sin(hand.rotation)]) * -1 + origin[1]
    joints = [(int(x), int(y)) for x,y in zip(joints_x, joints_y)]

    transform(ua_rect, joints[0], upperarm)
    transform(fa_rect, joints[1], forearm)
    transform(h_rect, joints[2], hand)
    # transform the hand a bit more because it's weird
    h_rect.center += np.array([np.cos(hand.rotation), 
                              -np.sin(hand.rotation)]) * -10

    display.blit(ua_image, ua_rect)
    display.blit(fa_image, fa_rect)
    display.blit(h_image, h_rect)

    # rotate arm lines
    line_ua = pygame.transform.rotozoom(line_upperarm, 
                                        np.degrees(upperarm.rotation), 1)
    line_fa = pygame.transform.rotozoom(line_forearm, 
                                        np.degrees(forearm.rotation), 1)
    line_h = pygame.transform.rotozoom(line_hand, 
                                        np.degrees(hand.rotation), 1)
    # translate arm lines
    lua_rect = line_ua.get_rect()
    transform_lines(lua_rect, joints[0], upperarm)

    lfa_rect = line_fa.get_rect()
    transform_lines(lfa_rect, joints[1], forearm)

    lh_rect = line_h.get_rect()
    transform_lines(lh_rect, joints[2], hand)

    display.blit(line_ua, lua_rect)
    display.blit(line_fa, lfa_rect)
    display.blit(line_h, lh_rect)

    # draw circles at joints for pretty
    pygame.draw.circle(display, black, joints[0], 30)
    pygame.draw.circle(display, arm_color, joints[0], 12)
    pygame.draw.circle(display, black, joints[1], 20)
    pygame.draw.circle(display, arm_color, joints[1], 7)
    pygame.draw.circle(display, black, joints[2], 15)
    pygame.draw.circle(display, arm_color, joints[2], 5)

    # check for quit
    for event in pygame.event.get():
        if event.type == pygame.locals.QUIT:
            pygame.quit()
            sys.exit()

    pygame.display.update()
    fpsClock.tick(30)
