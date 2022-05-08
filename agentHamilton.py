from agentStar import generateAdjacentMatrix
from game import Direction, Point
import numpy as np
import random

class AgentHamilton:
    def __init__(self,game):
        self.block_size = 20
        self.move = 0
        self.maze = [ [0 for i in range(game.w//20)] for j in range(game.h//20) ]
        self.maze, self.hamiltonianPath = self.generateHamiltonian(self.maze)
        self.NoHamiltonian = (game.w//20 * game.h//20) %2 != 0
        if self.NoHamiltonian:print("No Hamiltonian Cycle, Proceed with random moves")
        else: print("There is Hamiltonian Cycle in the Graph")

    def generateOddWidhthHamiltonian(self,maze):
        count = 2
        for i in range (len(maze)):
            if i %2 == 0:
                for j in range(1,len(maze[i])):
                    maze[i][j] = count
                    count += 1
            else:
                for j in range(len(maze[i]),1,-1):
                    maze[i][j-1] = count
                    count += 1

        for i in range(len(maze),1,-1):
            maze[i-1][0] = count
            count+=1

        maze[0][0] = 1
        return maze


    def generateEvenWidhthHamiltonian(self,maze):
        count = 2
        for i in range (len(maze[0])):
            if i %2 == 0:
                for j in range(1,len(maze)):
                    maze[j][i] = count
                    count += 1
            else:
                for j in range(len(maze),1,-1):
                    maze[j-1][i] = count
                    count += 1

        for i in range(len(maze[0]),1,-1):
            maze[0][i-1] = count
            count+=1

        maze[0][0] = 1
        return maze


    def generateHamiltonian(self,maze):
        d = {}
        hamiltonian=None
        if len(maze[0]) %2 == 0:
            hamiltonian = self.generateEvenWidhthHamiltonian(maze)
        else:
            hamiltonian = self.generateOddWidhthHamiltonian(maze)
        for i in range(len(hamiltonian)):
            for j in range(len(hamiltonian[i])):
                d[hamiltonian[i][j]] = (i,j)
        return hamiltonian,d

    def get_state(self,game):

        head = game.snake[0]

        
        hamiltonianNode = self.maze[head.y//self.block_size][head.x//self.block_size]


        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            hamiltonianNode,
            # Move Direction
            [
                dir_l,
                dir_r,
                dir_u,
                dir_d,
            ],
            [game.w,game.h]
        ]

        return np.array(state, dtype=object)

    def get_action(self,state,preTrained=False):
        final_move = [0,0,0]

        if self.NoHamiltonian:
            idx = random.randint(0,2)
        else:
            currentNode, direction,game = state

            nextNode = currentNode+1
            if nextNode > ((game[0]//self.block_size) * (game[1]//self.block_size) ):
                nextNode = 2
            currentHamiltonianPath_y,currentHamiltonianPath_x = self.hamiltonianPath.get(currentNode)
            currentHamiltonianPath = Point(currentHamiltonianPath_x,currentHamiltonianPath_y)

            
            nextHamiltonianPath_y,nextHamiltonianPath_x = self.hamiltonianPath.get(nextNode)
            nextHamiltonianPath = Point(nextHamiltonianPath_x,nextHamiltonianPath_y)

            
            target_location = [
                nextHamiltonianPath.x < currentHamiltonianPath.x, # food left 0
                nextHamiltonianPath.x > currentHamiltonianPath.x, # food right 1
                nextHamiltonianPath.y < currentHamiltonianPath.y, # food up 2
                nextHamiltonianPath.y > currentHamiltonianPath.y, # food down 3
            ]
            if (target_location[0] and direction[0]) or (target_location[1] and direction[1]) or (target_location[2] and direction[2]) or (target_location[3] and direction[3]) : #target is in straight
                idx = 0
            elif (target_location[3] and direction[0]) or (target_location[2] and direction[1]) or (target_location[0] and direction[2]) or (target_location[1] and direction[3]): #target is in right
                idx = 2
            elif (target_location[2] and direction[0]) or (target_location[3] and direction[1]) or (target_location[1] and direction[2]) or (target_location[0] and direction[3]): #target is in up
                idx = 1
            else: 
                idx = random.randint(1,2)

        # nextHamiltonianPath = ((nextHamiltonianPath[0]*20, nextHamiltonianPath[1]*20)) if nextHamiltonianPath != None else None

        final_move[idx] = 1
        return final_move, None
