import pygame
import pygame.locals

white = (255, 255, 255)

pygame.init()

display = pygame.display.set_mode((300, 300))
fpsClock = pygame.time.Clock()

image = pygame.image.load('pic.png')
degrees = 0

while 1:

    display.fill(white)

    degrees += 1

    rotated = pygame.transform.rotozoom(image, degrees, 1)
    rect = rotated.get_rect()
    rect.center = (150, 150)

    display.blit(rotated, rect)

    # check for quit
    for event in pygame.event.get():
        if event.type == pygame.locals.QUIT:
            pygame.quit()
            sys.exit()

    pygame.display.update()
    fpsClock.tick(30)
