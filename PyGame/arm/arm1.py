import numpy as np
import pygame
import pygame.locals

from armpart import ArmPart

black = (0, 0, 0)
white = (255, 255, 255)

pygame.init()

width = 500
height = 500
display = pygame.display.set_mode((width, height))
fpsClock = pygame.time.Clock()

upperarm = ArmPart('upperarm.png', scale=.7)

base = (width / 2, height / 2)

while 1:

    display.fill(white)

    ua_image, ua_rect = upperarm.rotate(.01) 
    ua_rect.center += np.asarray(base)
    ua_rect.center -= np.array([np.cos(upperarm.rotation) * upperarm.offset,
                                -np.sin(upperarm.rotation) * upperarm.offset])

    display.blit(ua_image, ua_rect)
    pygame.draw.circle(display, black, base, 30)

    # check for quit
    for event in pygame.event.get():
        if event.type == pygame.locals.QUIT:
            pygame.quit()
            sys.exit()

    pygame.display.update()
    fpsClock.tick(30)
