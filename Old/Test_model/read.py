import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os, shutil

# Fonction pour lire les données du fichier FreeFem++
def solution(filename):
    '''
    Parameters
    ----------
    filename : STR
        Name of the file that contains solutions.

    Returns
    -------
    DF : list of Dataframe that contains solution at indice corresponding to times
    times : list of values of the discretised time

    '''    

    data = []
    T = []
    with open(filename, 'r') as f:
        # Ignorer la première ligne (en-tête)
        header = f.readline()
    
        # Lire les données ligne par ligne
        for line in f:
            t, x, y, u, v, w = map(float, line.split())  # Convertir chaque élément en float


            data.append((t, x, y, u, v, w))  # Ajouter (x, y, u) pour le temps t
    
    data = np.array(data)
    times = np.unique(data[:, 0])
    groupes_par_temps = [data[data[:, 0] == t][:,1:] for t in times]

    nt = len(times)
    
    X = [groupes_par_temps[i][:,0] for i in range(nt)]
    Y = [groupes_par_temps[i][:,1] for i in range(nt)]
    
    U = [groupes_par_temps[i][:,2] for i in range(nt)]
    V = [groupes_par_temps[i][:,3] for i in range(nt)]
    W = [groupes_par_temps[i][:,4] for i in range(nt)]
    
    DF = [pd.DataFrame([X[i], Y[i], U[i], V[i], W[i]], index=["x", "y", "u(x,y)", "v(x,y)", "w(x,y)"]).transpose() for i in range(nt)]

    return DF, times

def clean(folder = 'output/heatmaps'):
    
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
# Fonction pour plot les données du fichier FreeFem++

def plot2D(df, t, name):
    """
    Parameters
    ----------
    df : list of Dataframe that contains solution at indice corresponding to times
    times : list of values of the discretised time
    t : int
        index of time simulation.
    name : str
        name for the figure generated.

    Returns
    -------
    None.

    """
    
    vmin, vmax = 0, 5
    # Définir une grille régulière pour l'interpolation
    grid_x, grid_y = np.meshgrid(
        np.linspace(df['x'].min(), df['x'].max(), 200),
        np.linspace(df['y'].min(), df['y'].max(), 200)
    )
    
    # Interpolation avec scipy.griddata
    grid_u = griddata(
        (df['x'], df['y']),    # Coordonnées des nœuds
        df['u(x,y)'],          # Valeurs à interpoler
        (grid_x, grid_y),      # Points de la grille régulière
        method='cubic'         # Interpolation cubique (smooth)
    )
    
    # Affichage de la carte de chaleur
    plt.figure(figsize=(8, 6))
    heatmap = plt.contourf(grid_x, grid_y, grid_u, levels=100,  cmap='RdYlGn', vmin=vmin, vmax=vmax)
    plt.colorbar(heatmap, label='u(x, y)')
    
    # Ajouter les points originaux pour comparaison
    #plt.scatter(df['x'], df['y'], c='red', label='Points originaux', s=30)
    plt.title('u('+str(t)+', x, y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(name+".png",  dpi=300)


    #plt.grid(True, linestyle='--', alpha=0.5)
    #plt.show()

def plot3D(df, t):

    grid_x, grid_y = np.meshgrid(
        np.linspace(df['x'].min(), df['x'].max(), 100),
        np.linspace(df['y'].min(), df['y'].max(), 100)
    )
    
    # Interpolation des valeurs de 'u(x,y)'
    grid_u = griddata((df['x'], df['y']), df['u(x,y)'], (grid_x, grid_y), method='linear')
    
    # Affichage en surface
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(grid_x, grid_y, grid_u, cmap='viridis', edgecolor='none')
    
    # Ajouter des étiquettes
    ax.set_title('u('+str(t)+', x, y)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x, y)')
    
    # Ajouter une barre de couleur
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='u(x, y)')
    
    plt.show()