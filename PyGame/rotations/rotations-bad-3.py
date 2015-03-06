import pygame
import pygame.locals

white = (255, 255, 255)

pygame.init()

display = pygame.display.set_mode((300, 300))
fpsClock = pygame.time.Clock()

image = pygame.image.load('pic.png')
radians = 0

while 1:

    display.fill(white)

    radians += .5

    rotated_image = pygame.transform.rotate(image, radians)
    rect = rotated_image.get_rect()
    rect.center = (150, 150)

    display.blit(rotated_image, rect)

    # check for quit
    for event in pygame.event.get():
        if event.type == pygame.locals.QUIT:
            pygame.quit()
            sys.exit()

    pygame.display.update()
    fpsClock.tick(30)
