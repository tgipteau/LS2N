from computes import norm_ueq, get_dfs, plots
from simu import run_simulation, T_linspace
from tqdm import tqdm
    
if __name__ == "__main__":
    
    nb_simulations = 100
    range_sim = range(0, nb_simulations)
    Ns = []
    
    for i in tqdm(range_sim):
        
        sim_name = f"_{i}"
        uplus, wplus, umoins, wmoins = run_simulation(sim_name=sim_name)
        df, df_feux = get_dfs(sim_name=sim_name)
        N, val_span = norm_ueq(U=df, T=T_linspace, uplus=uplus, wplus=wplus)
        Ns.append(N)
        print(N)
        
    plots(Ns, T_linspace, val_span, uplus, wplus)
    