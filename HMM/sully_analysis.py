#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 09:37:25 2025

@author: SULLY
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.interpolate import griddata
import cma
import pickle as pic

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
    Ueq = np.array([uplus, wplus])
    
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

def l2_norm(u, dx, dy):
    return (np.nansum(np.abs(u)**2) * dx * dy)

def expo(t, Lambda, C):
    return C * np.exp(-Lambda*t)

def logi(t, L, k, t0):
    return L/(1+np.exp(-k*(t-t0)))

def J(val, T, Norm):
    J = 0
    for i in range(len(T)):

        J = J + (Norm[i]-logi(T[i], val[0], val[1], val[2]))**2

    return J

def timeseeker(params, DF, T):
    
    for key, value in params.items():
        globals()[key] = value

    D = (f* alpha * delta - h*(c+f ))/( a*h)
    uplus = h*(b+ np.sqrt(D))/f
    vplus = b+ np.sqrt(D)
    wplus = alpha *(b+ np.sqrt(D))/ beta 
    Ueq = np.array([uplus, vplus, wplus])
    
    N = norm_L2(params, DF, T)  
    N0 = N[0]
    
    Nt = N/N0
    
    cma_init = [np.float64(0.9711015065004944),
     np.float64(-0.013860312936169763),
     np.float64(383.1779781162241)]
    options = {'bounds':[None, None], 'tolfun':1e-12, 'popsize': 200}
    
    escma = cma.CMAEvolutionStrategy(cma_init, 3, options).optimize(J, None,
                                                                    None, 1,
                                                                    (T,Nt))
    return escma.result[0]

def verif(params, DF, T):
    for key, value in params.items():
        globals()[key] = value

    D = (f* alpha * delta - h*(c+f ))/( a*h)
    uplus = h*(b+ np.sqrt(D))/f
    vplus = b+ np.sqrt(D)
    wplus = alpha *(b+ np.sqrt(D))/ beta 
    Ueq = np.array([uplus, vplus, wplus])
    
    N = norm_L2(params, DF, T)  
    N0 = N[0]
    
    [L, k, t0] = timeseeker(params, DF, T)
    Fx = [logi(t, L, k, t0)* N0 for t in T] 
    

    fig, ax = plt.subplots()
    ax.plot(T,N, label = r'$\| u - u_{eq} \|_{L^2}$', color = "lightgreen")
    ax.plot(T,Fx, label = r'$Ce^{-\lambda t} \| u_0 - u_{eq} \|_{L^2}$', color = "blue")
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.set_xlabel("time")
    
    return [L, k, t0]

import model
import read

def run_multiple(cond_init, tau):
    Tk = []
    Uk = []



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
        "tmax": tau,
        "p": 0,
        "I": 0,
        "ui": cond_init[0],
        "vi": cond_init[1],
        "wi": cond_init[2]
    }
    
    model.sim_PDE(params)
    
    
    filename = 'output/solution.txt'
    
    DF, T = read.solution(filename)
    #Tn= verif(params, DF, T)  
    #Tk.append(Tn)
    Uk.append(DF)
    
    #NL2 = np.array([norm_L2(params, Uk[k], T) for k in range(len(Uk))])
    #Tcar = np.array([Tk[k][-1] for k in range(len(Uk))])
    
    #TbN0 = pd.DataFrame(data = {"U0":[NL2[k][0]for k in range(len(Uk))], "Tc": Tcar})
    
    NSim = [T, Uk]
    
    
    
    with open('output/mesures_and_time.txt','wb') as mes_hyb:
        mon_pickler=pic.Pickler(mes_hyb)
        mon_pickler.dump(NSim)

    
    return NSim


def plot_norm_by_u0(params, DF, T):
    for key, value in params.items():
        globals()[key] = value

    D = (f* alpha * delta - h*(c+f ))/( a*h)
    uplus = h*(b+ np.sqrt(D))/f
    vplus = b+ np.sqrt(D)
    wplus = alpha *(b+ np.sqrt(D))/ beta 
    Ueq = np.array([uplus, vplus, wplus])
    
    color = ['royalblue','orange','green','firebrick','mediumorchid', 
                           'sienna', 'hotpink','grey', 'yellowgreen', 'darkturquoise',
                           'orangered','steelblue','mediumseagreen',
                           'darkslategray', 'cyan', 'olivedrab', 'orchid', 'peru', 'honeydew',
                           'rosybrown', 'indigo','crimson','forestgreen','navy' ]

    
    grid_x, grid_y = np.meshgrid(
        np.linspace(DF[0][0]['x'].min(), DF[0][0]['x'].max(), 200),
        np.linspace(DF[0][0]['y'].min(), DF[0][0]['y'].max(), 200)
    )
    dx = np.mean(np.diff(grid_x))  # Approximation du pas en x
    dy = np.mean(np.diff(grid_y, axis = 0))  # Approximation du pas en y
    
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[1].set_xlabel("t")
    for k in range(len(DF)):
        U = DF[k]
        
        grid_u = [griddata(
            (U[n]['x'], U[n]['y']),    # Coordonnées des nœuds
            U[n]['u(x,y)'],          # Valeurs à interpoler
            (grid_x, grid_y),      # Points de la grille régulière
            method='cubic'         # Interpolation cubique (smooth)
        ) for n in range(len(U)) ]
        dUeq = [l2_norm(grid_u[n] - uplus, dx, dy) for n in range(len(T))]
        dUex = [l2_norm(grid_u[n], dx, dy) for n in range(len(T))]
        
        Nd = [dUeq, dUex]
        leg_y = [r'$\| u - u_{eq} \|_{L^2}$', r'$\| u - u_{ex} \|_{L^2}$']
        leg_title = ["Distance during time from equilibrum state", "Distance during time from extinction state"]
        for i, ax in enumerate(axes):
            ax.plot(T, Nd[i], color = color[k])
            ax.set_ylabel(leg_y[i])
            
            ax.set_title(leg_title[i])
    
        #fig.suptitle("Distances from steadies states", fontsize=14)
    fig.savefig('output/others/distance_curves.png', bbox_inches='tight')
    
    
    
    
def norm_L2(U1, U2 = [0, 0, 0]):
    """
    Parameters
    ----------

    U1 : Dataframe pandas
        Solution computed from freefem++ at time t.
    U2 : Dataframe pandas
        Solution computed from freefem++ at time t.

    Returns
    -------
    N : list of arrays
        contains norm between U1 and U1.

    """
        
    
    grid_x, grid_y = np.meshgrid(
        np.linspace(U1['x'].min(), U1['x'].max(), 100),
        np.linspace(U1['y'].min(), U1['y'].max(), 100))
    dx = np.mean(np.diff(grid_x))  # Approximation du pas en x
    dy = np.mean(np.diff(grid_y, axis = 0))  # Approximation du pas en y
    

    
    
        
        
        
    grid_u1 = griddata(
        (U1['x'], U1['y']),    # Coordonnées des nœuds
        U1['u(x,y)'],          # Valeurs à interpoler
        (grid_x, grid_y),      # Points de la grille régulière
        method='cubic'         # Interpolation cubique (smooth)
    )
    grid_v1 = griddata(
        (U1['x'], U1['y']),    # Coordonnées des nœuds
        U1['v(x,y)'],          # Valeurs à interpoler
        (grid_x, grid_y),      # Points de la grille régulière
        method='cubic'         # Interpolation cubique (smooth)
    )
    grid_w1 = griddata(
        (U1['x'], U1['y']),    # Coordonnées des nœuds
        U1['v(x,y)'],          # Valeurs à interpoler
        (grid_x, grid_y),      # Points de la grille régulière
        method='cubic'         # Interpolation cubique (smooth)
    )
    
    UVW1 = [grid_u1, grid_v1, grid_w1]
    if isinstance(U2, pd.DataFrame):
        grid_u2 = griddata(
            (U2['x'], U2['y']),    # Coordonnées des nœuds
            U2['u(x,y)'],          # Valeurs à interpoler
            (grid_x, grid_y),      # Points de la grille régulière
            method='cubic'         # Interpolation cubique (smooth)
        )
        grid_v2 = griddata(
            (U2['x'], U2['y']),    # Coordonnées des nœuds
            U2['v(x,y)'],          # Valeurs à interpoler
            (grid_x, grid_y),      # Points de la grille régulière
            method='cubic'         # Interpolation cubique (smooth)
        )
        grid_w2 = griddata(
            (U2['x'], U2['y']),    # Coordonnées des nœuds
            U2['v(x,y)'],          # Valeurs à interpoler
            (grid_x, grid_y),      # Points de la grille régulière
            method='cubic'         # Interpolation cubique (smooth)
        )
        
        UVW2 = [grid_u2, grid_v2, grid_w2]
   
        N = [UVW1[i] - UVW2[i] for i in range(len(UVW1)) ]
        N = [l2_norm(N[i], dx, dy) for i in range(len(UVW1)) ]
        
    if isinstance(U2, list) : 
        N = [l2_norm(UVW1[i]-U2[i], dx, dy) for i in range(len(UVW1)) ]
    return N