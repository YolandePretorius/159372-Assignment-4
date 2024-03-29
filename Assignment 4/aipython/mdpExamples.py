# mdpExamples.py - MDP Examples
# AIFCA Python3 code Version 0.9.1 Documentation at http://aipython.org

# Artificial Intelligence: Foundations of Computational Agents
# http://artint.info
# Copyright David L Poole and Alan K Mackworth 2017-2020.
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: http://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

from mdpProblem import MDP, GridMDP

class party(MDP): 
    """Simple 2-state, 2-Action Partying MDP Example"""
    def __init__(self, discount=0.9):
        states = {'healthy','sick'}
        actions = {'relax', 'party'}
        MDP.__init__(self, states, actions, discount)

    def R(self,s,a):
        "R(s,a)"
        return { 'healthy': {'relax': 7, 'party': 10},
                 'sick':    {'relax': 0, 'party': 2 }}[s][a]

    def P(self,s,a):
        "returns a dictionary of {s1:p1} such that P(s1 | s,a)=p1. Other probabilities are zero."
        phealthy = {  # P('healthy' | s, a)
                     'healthy': {'relax': 0.95, 'party': 0.7},
                     'sick': {'relax': 0.5, 'party': 0.1 }}[s][a]
        return {'healthy':phealthy, 'sick':1-phealthy}


class MDPtiny(GridMDP):
    def __init__(self, discount=0.9):
        actions = ['right', 'upC', 'left', 'upR']
        self.x_dim = 2   # x-dimension
        self.y_dim = 3
        states = [(x,y) for x in range(self.x_dim) for y in range(self.y_dim)]
        # for GridMDP
        self.xoff = {'right':0.25, 'upC':0, 'left':-0.25, 'upR':0}
        self.yoff = {'right':0, 'upC':-0.25, 'left':0, 'upR':0.25}
        GridMDP.__init__(self, states, actions, discount)

    def P(self,s,a):
        """return a dictionary of {s1:p1} if P(s1 | s,a)=p1. Other probabilities are zero.
        """
        (x,y) = s
        if a == 'right':
            return {(1,y):1}
        elif a == 'upC':
            return {(x,min(y+1,2)):1}
        elif a == 'left':
            if (x,y) == (0,2): return {(0,0):1}
            else: return {(0,y): 1}
        elif a == 'upR':
            if x==0:
                if y<2: return {(x,y):0.1, (x+1,y):0.1, (x,y+1):0.8}
                else:  # at (0,2)
                    return {(0,0):0.1, (1,2): 0.1, (0,2): 0.8}
            elif y < 2: # x==1
                return {(0,y):0.1, (1,y):0.1, (1,y+1):0.8}
            else: # at (1,2)
               return {(0,2):0.1, (1,2): 0.9}

    def R(self,s,a):
        (x,y) = s
        if a == 'right':
            return [0,-1][x]
        elif a == 'upC':
            return [-1,-1,-2][y]
        elif a == 'left':
            if x==0:
                return [-1, -100, 10][y]
            else: return 0
        elif a == 'upR':
            return [[-0.1, -10, 0.2],[-0.1, -0.1, -0.9]][x][y]
                # at (0,2) reward is   0.1*10+0.8*-1=0.2
                
class grid(GridMDP):
    """ x_dim * y_dim grid with rewarding states"""
    def __init__(self, discount= 0.9, x_dim=10, y_dim=10):
        self.x_dim = x_dim # size in x-direction
        self.y_dim = y_dim # size in y-direction
        actions = ['up', 'down', 'right', 'left']
        states = [(x,y) for x in range(y_dim) for y in range(y_dim)]
        self.rewarding_states = {(3,2):-10, (3,5):-5, (8,2):10, (7,7):3}
        self.fling_states = {(8,2), (7,7)}
        self.xoff = {'right':0.25, 'up':0, 'left':-0.25, 'down':0}
        self.yoff = {'right':0, 'up':0.25, 'left':0, 'down':-0.25}
        GridMDP.__init__(self, states, actions, discount)
 
    def intended_next(self,s,a):
        """returns the next state in the direction a.
        This is where the agent will end up if to goes in its intended_direction
             (which it does with probability 0.7).
        """
        (x,y) = s
        if a=='up':
            return (x,  y+1 if y+1 < self.y_dim else y)
        if a=='down':
            return (x,  y-1 if y > 0 else y)
        if a=='right':
            return (x+1 if x+1 < self.x_dim else x,y)
        if a=='left':
            return (x-1 if x > 0 else x,y)

    def P(self,s,a):
        """return a dictionary of {s1:p1} if P(s1 | s,a)=p1. Other probabilities are zero.
        Corners are tricky because different actions result in same state.
        """
        if s in self.fling_states:
            return {(0,0): 0.25, (self.x_dim-1,0):0.25, (0,self.y_dim-1):0.25, (self.x_dim-1,self.y_dim-1):0.25}
        res = dict()
        for ai in self.actions:
            s1 = self.intended_next(s,ai)
            ps1 = 0.7 if ai==a else 0.1
            if s1 in res: # occurs in corners
                res[s1] += ps1
            else:
                res[s1] = ps1           
        return res       

    def R(self,s,a):
         if s in self.rewarding_states:
             return self.rewarding_states[s]
         else:
             (x,y) = s
             rew = 0
             # rewards from crashing:
             if y==0: ## on bottom.
                 rew += -0.7 if a == 'down' else -0.1
             if y==self.y_dim-1: ## on top.
                 rew += -0.7 if a == 'up' else -0.1
             if x==0: ## on left
                 rew += -0.7 if a == 'left' else -0.1
             if x==self.x_dim-1: ## on right.
                 rew += -0.7 if a == 'right' else -0.1
             return rew
                
## Testing value iteration
# Try the following:
# pt = party(discount=0.9)
# pt.vi(1)
# pt.vi(100)
# party(discount=0.99).vi(100)
# party(discount=0.4).vi(100)

# gr = grid()
# gr.show()
# q,v,pi = gr.vi(100)
# q[(7,2)]


## Testing asynchronous value iteration
# Try the following:
# pt = party(discount=0.9)
# pt.avi(10)
# pt.vi(1000)

# gr = grid()
# q = gr.avi(100000)
# q[(7,2)]

