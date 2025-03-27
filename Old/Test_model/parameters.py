#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 15:40:43 2025

@author: e24h297n
"""

def builder(Lambda):
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
        "tmax": 100.0,
        "p": 0.3,
        "I": 0.3
    }
    keys = list(params.keys())
    if len(Lambda) != len(keys):
        raise ValueError("La liste doit contenir autant de valeurs que de paramètres dans le dictionnaire.")
    
    # Affecter les valeurs du dictionnaire
    for i, key in enumerate(keys):
        params[key] = Lambda[i]
    
    with open("params.txt", "w") as file:
        for key, value in params.items():
            file.write(f"{key} {value}\n")