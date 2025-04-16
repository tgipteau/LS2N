import os

from computes import norm_ueq, get_dfs, plots
from simu import run_simulation, T_linspace
from tqdm import tqdm
    
if __name__ == "__main__":
    
    nb_simulations = 25
    range_sim = range(0, nb_simulations)
    Ns = []
    os.makedirs("Simulations", exist_ok=True)
    
    for i in tqdm(range_sim):
        
        sim_name = f"_{i}"
        uplus, wplus, umoins, wmoins = run_simulation(sim_name=sim_name)
        df, df_feux = get_dfs(sim_name=sim_name)
        N, val_span = norm_ueq(U=df, T=T_linspace, uplus=uplus, wplus=wplus)
        Ns.append(N)
        
    plots(Ns, T_linspace, val_span, uplus, wplus)
    