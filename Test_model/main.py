# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:48:09 2024

@author: sully
"""
import model
import read
import analysis as als
params = {
    "a": 0.006,
    "b": 0.247,
    "c": 0.01,
    "r": 0.134,
    "f": 0.017,
    "h": 0.04,
    "alpha": 0.5,
    "beta": 0.5,
    "delta": 0.268,
    "tmax": 10,
    "p": 0.1,
    "I": 0.5,
    "ui": 5.838126612843206,
    "vi": 2.481203810458363,
    "wi": 2.481203810458363
}


model.sim_PDE(params)

filename = 'output/solution.txt'

DF, T = read.solution(filename)
print(T)
Time_to_plot = list(range(0, len(T), len(T)//15))

read.clean()
for t in Time_to_plot:
    read.plot2D(DF[t], T[t], "output/heatmaps/plot_at"+str(T[t]))



N = als.norm_ueq(params, DF, T)    
