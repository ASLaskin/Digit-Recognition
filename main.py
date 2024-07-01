import pygame
import sys
from pygame.locals import *
import numpy as np
from keras import models
import cv2

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

IMAGESAVE = False
MODEL = models.load_model('model.keras')

LABELS = {0: 'Zero', 1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine'}

# Initialize
pygame.init()
FONT = pygame.font.SysFont('arial', 18)
pygame.display.set_caption('Digit Recognition')
DISPLAYSURFACE = pygame.display.set_mode((640, 480))

BOUNDRYINC = 5

iswriting = False
number_xcord = []
number_ycord = []
image_cnt = 1

PREDICT = True
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURFACE, WHITE, (xcord, ycord), 4, 0)

            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            if number_xcord and number_ycord:
                number_xcord = sorted(number_xcord)
                number_ycord = sorted(number_ycord)

                rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDRYINC, 0), min(640, number_xcord[-1] + BOUNDRYINC)
                rect_min_y, rect_max_y = max(number_ycord[0] - BOUNDRYINC, 0), min(480, number_ycord[-1] + BOUNDRYINC)

                number_xcord = []
                number_ycord = []

                ing_arr = np.array(pygame.PixelArray(DISPLAYSURFACE))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)

                if IMAGESAVE:
                    cv2.imwrite(f'img_{image_cnt}.png', ing_arr)
                    image_cnt += 1

                if PREDICT and ing_arr.size > 0:
                    image = cv2.resize(ing_arr, (28, 28))
                    image = np.pad(image, (10, 10), 'constant', constant_values=0)
                    image = cv2.resize(image, (28, 28)) / 255.0

                    label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))])

                    textSurface = FONT.render(label, True, RED)
                    textRecObj = textSurface.get_rect()
                    textRecObj.left, textRecObj.top = rect_max_x, rect_max_y
                    DISPLAYSURFACE.blit(textSurface, textRecObj)

                pygame.display.update()

        if event.type == KEYDOWN:
            if event.unicode == "c":
                DISPLAYSURFACE.fill(BLACK)
                pygame.display.update()
    pygame.display.update()
