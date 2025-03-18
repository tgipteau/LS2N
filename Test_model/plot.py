# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:48:09 2024

@author: sully
"""
import subprocess
import read



filename = 'solution.txt'

DF, T = read.solution(filename)

Time_to_plot = [0, int(len(T)/4), int(len(T)/2), int(len(T)*3/4), -1]

for t in Time_to_plot:
    read.plot2D(DF[t], T[t], "output/plot_at"+str(T[t]))
    #read.plot3D(DF[t], T[t])