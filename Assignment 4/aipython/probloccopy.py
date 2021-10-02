# probLocalization.py - Localization example
# AIFCA Python3 code Version 0.9.1 Documentation at http://aipython.org

# Artificial Intelligence: Foundations of Computational Agents
# http://artint.info
# Copyright David L Poole and Alan K Mackworth 2017-2020.
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: http://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

from probHMM import HMMVEfilter, HMM
from display import Displayable
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons

class HMM_Local(HMMVEfilter):
    def __init__(self, hmm):
        HMMVEfilter.__init__(self, hmm)

    def go(self, action):
        self.hmm.trans = self.hmm.act2trans[action]
        self.advance()

local_states = list(range(16))
door_positions = {2,4,7,11}
def prob_door(loc): return 0.8 if loc in door_positions else 0.1
local_obs = {'door':[prob_door(i) for i in range(16)]}
hmm_16pos = HMM(local_states, {'door'}, local_obs, [], [1/16 for i in range(16)])
hmm_16pos.act2trans = {'right': [[0.1 if next == current
                                 else 0.8 if next == (current+1)%16
                                 else 0.074 if next == (current+2)%16
                                 else 0.002 for next in range(16)] for current in range(16)],
                        'left': [[0.1 if next == current
                                 else 0.8 if next == (current-1)%16
                                 else 0.074 if next == (current-2)%16
                                 else 0.002 for next in range(16)] for current in range(16)]}

loc_filt = HMM_Local(hmm_16pos)
# loc_filt.observe({'door':True}); loc_filt.go("right");  loc_filt.observe({'door':False});loc_filt.go("right");  loc_filt.observe({'door':True})
# loc_filt.state_dist

class ShowLocal(Displayable):
    def __init__(self,hmm):
        self.hmm = hmm
        self.loc_filt = HMM_Local(hmm)
        fig,(self.ax) = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        left_butt = Button(plt.axes([0.05,0.05,0.1,0.075]), "left")
        left_butt.on_clicked(self.left)
        right_butt = Button(plt.axes([0.25,0.05,0.1,0.075]), "right")
        right_butt.on_clicked(self.right)
        door_butt = Button(plt.axes([0.45,0.05,0.1,0.075]), "door")
        door_butt.on_clicked(self.door)
        nodoor_butt = Button(plt.axes([0.65,0.05,0.1,0.075]), "no door")
        nodoor_butt.on_clicked(self.nodoor)
        reset_butt = Button(plt.axes([0.85,0.05,0.1,0.075]), "reset")
        reset_butt.on_clicked(self.reset)
        self.bars = self.ax.bar(self.hmm.states, [1 for i in range(16)])
                #this makes sure y-axis goes to 1, graph overwritten in draw_dist
        self.draw_dist()
        plt.show()

    def draw_dist(self):
        for bar,val in zip(self.bars,(self.loc_filt.state_dist[i] for i in range(16))):
            bar.set_height(val)
        plt.draw()

    def left(self,event):
        self.loc_filt.go("left")
        self.draw_dist()
    def right(self,event):
        self.loc_filt.go("right")
        self.draw_dist()
    def door(self,event):
        self.loc_filt.observe({'door':True})
        self.draw_dist()
    def nodoor(self,event):
        self.loc_filt.observe({'door':False})
        self.draw_dist()
    def reset(self,event):
        self.loc_filt.state_dist = {i:1/16 for i in range(16)}
        self.draw_dist()

# sl = ShowLocal(hmm_16pos)

