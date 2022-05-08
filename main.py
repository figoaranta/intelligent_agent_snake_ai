from statistics import median
from agentDeepQ import AgentDeepQ
from agentHamilton import AgentHamilton
from agentStar import AgentStar
from game import SnakeGameAI


if __name__ == '__main__':
    width = 400
    height = 400
    game = SnakeGameAI(width,height)

    # Uncomment below for Agent Deep Q-1
    # agent = AgentDeepQ(11,256,3)
    # agent.model.load(model_name='dq1')

    # Uncomment below for Agent Deep Q-2
    # agent = AgentDeepQ(23,529,3)
    # agent.model.load(model_name='dq2')

    # Uncomment below for Agent Star
    # agent =  AgentStar(game)

    # Uncomment below for Agent Hamiltonian
    agent =  AgentHamilton(game)

    
    while True:

        state_old = agent.get_state(game)

        action,path = agent.get_action(state_old,True)

        reward, game_over, score = game.play_step(action,path)

        if game_over:
            game.reset()
