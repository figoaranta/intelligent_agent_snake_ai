from statistics import median
from agentDeepQ import AgentDeepQ
from agentHamilton import AgentHamilton
from agentStar import AgentStar
from game import SnakeGameAI
import numpy as np

if __name__ == '__main__':
    
    # environmentGrids = [(400,400),(320,320),(240,240)]
    # agents = ["dq1","dq2","hamilton","aStar"]
    # for w,h in environmentGrids:
    #     print("Environment Grid:",w,h)
    #     game = SnakeGameAI(w,h)
    #     for agent in agents:
    #         print("Agent:",agent)
    #         n_game = 0
    #         if agent == "dq1":
    #             agent = AgentDeepQ(11,256,3)
    #             agent.model.load(model_name='dq1')
    #         elif agent == "dq2":
    #             agent = AgentDeepQ(23,529,3)
    #             agent.model.load(model_name='dq2')
    #         elif agent == "hamilton":
    #             agent = AgentHamilton(game)
    #         else:
    #             agent = AgentStar(game)
    #         scores = []
    #         while n_game<10:
    #             state_old = agent.get_state(game)
    #             action,path = agent.get_action(state_old,True)
    #             reward, game_over, score = game.play_step(action,path,n_game+1)
    #             if game_over:
    #                 n_game+=1
    #                 # print("Game:",n_game)
    #                 # print("Score:",game.score)
    #                 scores.append(game.score)
    #                 game.reset()
    #         print(scores)
    #         print("Higest Score:",max(scores))
    #         print("Standard Deviation:",np.std(scores))
    #         print("Mean:", sum(scores)/len(scores))
    #         print("Sum of all scores:",sum(scores))
    #         print("Median:", median(scores))
    #         print("Lowest Score:", min(scores))


                    

    
    
    game = SnakeGameAI(200,200)
    # Deep Q with 11 inputs and 256 hidden 
    agent = agent = AgentHamilton(game)
    
    while True:

        state_old = agent.get_state(game)

        action,path = agent.get_action(state_old,True)

        reward, game_over, score = game.play_step(action,path)

        if game_over:
            game.reset()
