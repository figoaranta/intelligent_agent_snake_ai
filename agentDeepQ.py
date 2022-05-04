import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_Qnet, QTrainer
from helper import plot, save_plot
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class AgentDeepQ:

    def __init__(self,input,hidden,output):
        self.learning_rate = 0.001
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_Qnet(input,hidden,output)
        self.trainer = QTrainer(self.model,LR,self.gamma)
        self.input = input
    
    def get_state(self,game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y-20)
        point_d = Point(head.x, head.y+20)
        
        point_l2 = Point(head.x - 40, head.y)
        point_r2 = Point(head.x + 40, head.y)
        point_u2 = Point(head.x, head.y-40)
        point_d2 = Point(head.x, head.y+40)

        point_l3 = Point(head.x - 60, head.y)
        point_r3 = Point(head.x + 60, head.y)
        point_u3 = Point(head.x, head.y-60)
        point_d3 = Point(head.x, head.y+60)

        point_l4 = Point(head.x - 80, head.y)
        point_r4 = Point(head.x + 80, head.y)
        point_u4 = Point(head.x, head.y-80)
        point_d4 = Point(head.x, head.y+80)

        point_l5 = Point(head.x - 100, head.y)
        point_r5 = Point(head.x + 100, head.y)
        point_u5 = Point(head.x, head.y-100)
        point_d5 = Point(head.x, head.y+100)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN


        if self.input == 23:
            state = [
                # Danger Straight
                (dir_r and game.is_collision(point_r)) or
                (dir_l and game.is_collision(point_l)) or 
                (dir_u and game.is_collision(point_u)) or
                (dir_d and game.is_collision(point_d)),

                # Danger Right
                (dir_r and game.is_collision(point_d)) or
                (dir_l and game.is_collision(point_u)) or 
                (dir_u and game.is_collision(point_r)) or
                (dir_d and game.is_collision(point_l)),

                # Danger Left
                (dir_r and game.is_collision(point_u)) or
                (dir_l and game.is_collision(point_d)) or 
                (dir_u and game.is_collision(point_l)) or
                (dir_d and game.is_collision(point_r)),

                # Danger Straight 2
                (dir_r and game.is_collision(point_r2)) or
                (dir_l and game.is_collision(point_l2)) or 
                (dir_u and game.is_collision(point_u2)) or
                (dir_d and game.is_collision(point_d2)),

                # Danger Right 2
                (dir_r and game.is_collision(point_d2)) or
                (dir_l and game.is_collision(point_u2)) or 
                (dir_u and game.is_collision(point_r2)) or
                (dir_d and game.is_collision(point_l2)),

                # Danger Left 2
                (dir_r and game.is_collision(point_u2)) or
                (dir_l and game.is_collision(point_d2)) or 
                (dir_u and game.is_collision(point_l2)) or
                (dir_d and game.is_collision(point_r2)),

                # Danger Straight 3
                (dir_r and game.is_collision(point_r3)) or
                (dir_l and game.is_collision(point_l3)) or 
                (dir_u and game.is_collision(point_u3)) or
                (dir_d and game.is_collision(point_d3)),

                # Danger Right 3
                (dir_r and game.is_collision(point_d3)) or
                (dir_l and game.is_collision(point_u3)) or 
                (dir_u and game.is_collision(point_r3)) or
                (dir_d and game.is_collision(point_l3)),

                # Danger Left 3
                (dir_r and game.is_collision(point_u3)) or
                (dir_l and game.is_collision(point_d3)) or 
                (dir_u and game.is_collision(point_l3)) or
                (dir_d and game.is_collision(point_r3)),

                # Danger Straight 4
                (dir_r and game.is_collision(point_r4)) or
                (dir_l and game.is_collision(point_l4)) or 
                (dir_u and game.is_collision(point_u4)) or
                (dir_d and game.is_collision(point_d4)),

                # Danger Right 4
                (dir_r and game.is_collision(point_d4)) or
                (dir_l and game.is_collision(point_u4)) or 
                (dir_u and game.is_collision(point_r4)) or
                (dir_d and game.is_collision(point_l4)),

                # Danger Left 4
                (dir_r and game.is_collision(point_u4)) or
                (dir_l and game.is_collision(point_d4)) or 
                (dir_u and game.is_collision(point_l4)) or
                (dir_d and game.is_collision(point_r4)),

                # Danger Straight 5
                (dir_r and game.is_collision(point_r5)) or
                (dir_l and game.is_collision(point_l5)) or 
                (dir_u and game.is_collision(point_u5)) or
                (dir_d and game.is_collision(point_d5)),

                # Danger Right 5
                (dir_r and game.is_collision(point_d5)) or
                (dir_l and game.is_collision(point_u5)) or 
                (dir_u and game.is_collision(point_r5)) or
                (dir_d and game.is_collision(point_l5)),

                # Danger Left 5
                (dir_r and game.is_collision(point_u5)) or
                (dir_l and game.is_collision(point_d5)) or 
                (dir_u and game.is_collision(point_l5)) or
                (dir_d and game.is_collision(point_r5)),

                # Move Direction
                dir_l,
                dir_r,
                dir_u,
                dir_d,

                # Food location
                game.food.x < game.head.x, # food left
                game.food.x > game.head.x, # food right
                game.food.y < game.head.y, # food up
                game.food.y > game.head.y, # food down 

                # Length of Snake
                # snake_length
            ]
        else:
            state = [
            # Danger Straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger Right
            (dir_r and game.is_collision(point_d)) or
            (dir_l and game.is_collision(point_u)) or 
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)),

            # Danger Left
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)) or 
            (dir_u and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_r)),

            # Move Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y, # food down 

            # Length of Snake
            # snake_length
        ]

        return np.array(state, dtype=int)

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))

    def train_long_memory(self):
        if (len(self.memory)) > BATCH_SIZE:
            mini_sample = random.sample(self.memory,BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        states,actions,rewards,next_states,dones = zip(*mini_sample)

        self.trainer.train_step(states,actions,rewards,next_states,dones)

    def train_short_memory(self,state,action,reward,next_state,done):
        self.trainer.train_step(state,action,reward,next_state,done)

    def get_action(self,state,preTrained=False):
        if preTrained == False:
            self.epsilon = 80 - self.n_games
        
        final_move = [0,0,0]

        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state,dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move ,None

def train(agent=None,preTrained=False):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    if agent is None:
        agent = AgentDeepQ()
    game = SnakeGameAI()
    while agent.n_games < 500:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old,preTrained)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old,final_move,reward,state_new,done)

        # remember
        agent.remember(state_old,final_move,reward,state_new,done)

        if done: 
            game.reset()
            agent.n_games += 1
            if agent.learning_rate > 0.001:
                agent.learning_rate -=  0.001
            agent.train_long_memory()
            agent.model.save()

            if score > record:
                record = score

            print('Game',agent.n_games, 'Score',score, 'Record', record)

            plot_scores.append(score)

            total_score += score
            mean_score = total_score/agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores,plot_mean_scores)
    save_plot()
        

def loadTrainedAgent():
    agent = AgentDeepQ()
    agent.model.load()
    agent.epsilon = 0
    train(agent,True)

if __name__ == '__main__':
    train()