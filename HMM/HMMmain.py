import simu
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_dfs(from_folder="Simulations/default"):
    df = pd.read_csv(os.path.join(from_folder, "data.csv"),
                     names=["t", "x", "y", "ub", "wb", "um", "wm"], header=0, sep=" ", index_col=False)
    df = df.drop_duplicates()
    
    df_feux = pd.read_csv(os.path.join(from_folder, "fires.dat"),
                          names=['t', 'x', 'y', 'r', 'I'], header=None, sep=" ", index_col=False)
    
    return df, df_feux


def norm_ueq(U, T, uplus, wplus):
    """
    Parameters
    ----------
    params : dict
        parameters of the model.
    U : Dataframe pandas
        Solution computed from freefem++.
    T: linspace Temps
    Returns
    -------
    N : list of arrays
        contains norm of the solution between persistence equilibrum.

    """
    
    # vecteur equilibre P+
    Ueq = np.array([uplus, wplus, uplus, wplus])
    
    
    N = []
    # pour chaque temps, calculer les normes
    for i in T:
        Ut = U[U['t'] == i].iloc[:, -4:]
        N.append(np.array(np.sqrt((Ut - Ueq) ** 2).sum(axis=0)))
        
    # vecteur distance max (norm de O extinction à P+ persistance)
    extinct = np.zeros(shape=(Ut.shape[0], Ut.shape[1]))
    val_span = np.array(np.sqrt((extinct - Ueq) ** 2).sum(axis=0))
    
    return N, val_span


def plots(Ns, T, val_span):
    


    fig, ax = plt.subplots()
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.set_xlabel("time")
    ax.set_ylim(0, max(val_span))
    ax.invert_yaxis()
    ax.set_title(r'Distance from steady state $U_{eq}$ over time')

    # vecteur equilibre P+
    Ueq = np.array([uplus, wplus, uplus, wplus])
    
    # ajout des courbes
    for i, N in enumerate(Ns):
        # d'où quatre distances à P+ (enfonction du temps)
        ub_norm = [N[i][0] for i in range(len(N))]
        wb_norm = [N[i][1] for i in range(len(N))]
        um_norm = [N[i][2] for i in range(len(N))]
        wm_norm = [N[i][3] for i in range(len(N))]
        
        ax.plot(T, ub_norm, label=f'ub{i}-ueq{i}')
    
    plt.show()
    fig.savefig('output/others/distance_curve_eq.png', bbox_inches='tight')
    
    
    
if __name__ == "__main__":
    
    
    T = simu.T_linspace
    
    nb_simulations = 15
    Ns = []
    
    for i in tqdm(range(nb_simulations)):
        uplus, wplus, umoins, wmoins = simu.run_simulation()
        df, df_feux = get_dfs()
        N, val_span = norm_ueq(U=df, T=T, uplus=uplus, wplus=wplus)
        Ns.append(N)
        
    plots(Ns, T, val_span)
    