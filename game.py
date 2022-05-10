import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import time
pygame.init()
font = pygame.font.Font('arial.ttf', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)
GREEN = (127,255,0)

SPEED = 50

class SnakeGameAI:
    
    def __init__(self, w=400, h=400):
        self.BLOCK_SIZE = 20
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
        
    def reset(self):
        # init game state
        self.direction = Direction.DOWN
        # self.direction = Direction.RIGHT

        # self.head = Point(0, 0)
        self.head = Point(self.w//2, self.h//2)
        self.snake = [self.head, 
                    Point(self.head.x-self.BLOCK_SIZE, self.head.y),
                    Point(self.head.x-(2*self.BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
            
        
    def _place_food(self):
        x = random.randint(0, (self.w-self.BLOCK_SIZE )//self.BLOCK_SIZE )*self.BLOCK_SIZE 
        y = random.randint(0, (self.h-self.BLOCK_SIZE )//self.BLOCK_SIZE )*self.BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            if len(self.snake)>= (self.w//self.BLOCK_SIZE * self.h//self.BLOCK_SIZE):
                pass
            else:
                self._place_food()
            
        
    def play_step(self,action,shortestPath=None,n_game=None):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_game()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision():# or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward -= 10
            # time.sleep(2)
            return reward, game_over, self.score
        if len(self.snake)>= (self.w//self.BLOCK_SIZE * self.h//self.BLOCK_SIZE):
            game_over = True
            
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward += 10
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui(shortestPath,n_game)
        self.clock.tick(SPEED)
        # 6. return game over and score

        return reward, game_over, self.score
    
    def is_collision(self,pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - self.BLOCK_SIZE or pt.x < 0 or pt.y > self.h - self.BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        
        return False
    
    def quit_game(self):
        pygame.quit()
        quit()
        
    def _update_ui(self,shortestPath=None,n_game=None):
        
        self.display.fill(WHITE)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, self.BLOCK_SIZE, self.BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        
        
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, self.BLOCK_SIZE, self.BLOCK_SIZE))
        
        if shortestPath != None:
            for x,y in shortestPath:
                pygame.draw.rect(self.display, GREEN, pygame.Rect(x+4, y+4, 12, 12))

        text = font.render("Score: " + str(self.score), True, BLACK)
        self.display.blit(text, [0, 0])

        if n_game != None:
            text2 = font.render("Game: " + str(n_game), True, BLACK)
            self.display.blit(text2, [0,25])

        pygame.display.flip()
        
    def _move(self, action):

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action,[1,0,0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action,[0,1,0]):
            new_dir = clock_wise[(idx + 1) % 4]
        else:
            new_dir = clock_wise[(idx - 1) % 4]
        
        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += self.BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= self.BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += self.BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= self.BLOCK_SIZE
            
        self.head = Point(x, y)