import pygame
import pygame.locals

white = (255, 255, 255)

pygame.init()

display = pygame.display.set_mode((300, 300))
fpsClock = pygame.time.Clock()

image = pygame.image.load('pic.png')

while 1:

    display.fill(white)

    image = pygame.transform.rotate(image, 1)
    rect = image.get_rect()

    display.blit(image, rect)

    # check for quit
    for event in pygame.event.get():
        if event.type == pygame.locals.QUIT:
            pygame.quit()
            sys.exit()

    pygame.display.update()
    fpsClock.tick(30)
