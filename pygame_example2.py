import pygame

pygame.init()

black = (0, 0, 0)
width = 470
height = 600
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
FPS = 60
folder = 'img/'
background_img = pygame.image.load(folder+"parking_overlay.png")
backgroundimg_rect = background_img.get_rect()
tractor = pygame.image.load(folder+"tractor_sprite.png")
haystack = pygame.image.load(folder+"haystack.png")

class Player(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.transform.scale(tractor, (200, 100))
        self.image = pygame.transform.scale(haystack, (200, 100))
        self.image.set_colorkey(black)
        self.rect = self.image.get_rect()
        self.rect.x = 10
        self.rect.y = height - 10

all_sprites = pygame.sprite.Group()
player = Player()
all_sprites.add(player)
