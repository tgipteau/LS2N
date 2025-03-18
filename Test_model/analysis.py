#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 09:37:25 2025

@author: e24h297n
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sdfsdf

def norm_ueq(params, U, T):
    """
    Parameters
    ----------
    params : dict
        parameters of the model.
    U : Dataframe pandas
        Solution computed from freefem++.

    Returns
    -------
    N : list of arrays
        contains norm of the solution between persistence equilibrum.

    """
    for key, value in params.items():
        globals()[key] = value

    D = (f* alpha * delta - h*(c+f ))/( a*h)
    uplus = h*(b+ np.sqrt(D))/f
    vplus = b+ np.sqrt(D)
    wplus = alpha *(b+ np.sqrt(D))/ beta 
    Ueq = np.array([uplus, vplus, wplus])
    
    N = []
    N_0 = []
    for i in range(len(U)):
        Ut = U[i].iloc[:,-3:]
        
        N.append(np.array(np.sqrt((Ut - Ueq)**2).sum(axis=0)))
        N_0.append(np.array(np.sqrt((Ut)**2).sum(axis=0)))
        
    u_norm = [N[i][0] for i in range(len(N))]
    v_norm =  [N[i][1] for i in range(len(N))]
    w_norm = [N[i][2] for i in range(len(N))]
    
    fig, ax = plt.subplots()
    ax.plot(T,u_norm, label = r'$|u - u_{eq}|$', color = "lightgreen")
    ax.plot(T, v_norm, label = r'$|v - v_{eq}|$', color = "darkgreen")
    ax.plot(T, w_norm , label = r'$|w - w_{eq}|$', color = "black")
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.set_xlabel("time")
    ax.set_title(r'Distance from steady state $U_{eq}$ over time')
    """
    ax.axvline(960, color='r', ls='--')
    ax.text(980, -0.075, r'$\tau_n$', color='r', ha='right', va='top', rotation=0,
                transform=ax.get_xaxis_transform())
    """
    plt.show()
    fig.savefig('output/others/distance_curve_eq.png', bbox_inches='tight')
    
    u_norm0 = [N_0[i][0] for i in range(len(N))]
    v_norm0 =  [N_0[i][1] for i in range(len(N))]
    w_norm0 = [N_0[i][2] for i in range(len(N))]
    
    fig, ax = plt.subplots()
    ax.plot(T,u_norm0, label = r'$|u - u_{ex}|$', color = "lightgreen")
    ax.plot(T, v_norm0, label = r'$|v - v_{ex}|$', color = "darkgreen")
    ax.plot(T, w_norm0 , label = r'$|w - w_{ex}|$', color = "black")
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.set_xlabel("time")
    ax.set_title(r'Distance from extinction state $U_{ex}$ over time')
    """
    ax.axvline(960, color='r', ls='--')
    ax.text(980, -0.075, r'$\tau_n$', color='r', ha='right', va='top', rotation=0,
                transform=ax.get_xaxis_transform())
    """
    plt.show()
    fig.savefig('output/others/distance_curve_ex.png', bbox_inches='tight')
    
    return N



    