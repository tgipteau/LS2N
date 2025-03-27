#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 10:54:10 2025

@author: e24h297n
"""

import subprocess
import Test_model.read as read


def sim_PDE(params, script_name = "modele_gen.edp"):
    """
    Parameters
    ----------
    params : dict
        Contains parameters of the model.
    script_name : str, optional
        FreeFem++ file for simulation. The default is "modele_gen.edp".

    Returns
    -------
    None.

    """
    
    
    with open("input/params.txt", "w") as file:
        for key, value in params.items():
            file.write(f"{key} {value}\n")
            
    try:
        subprocess.run(['FreeFem++-CoCoa', script_name])
        print(f"Script Freefem {script_name} ok")
    except subprocess.CalledProcessError as e:
        print(f"Erreur {script_name}: {e}")
        exit()

    print("Simulation ended")

