"""
Manipulation des données brutes.
"""
import os

import pandas as pd
import glob
import re


# Fonction pour extraire le temps t du nom de fichier
def extraire_t(fichier):
    match = re.search(r"u-(\d+)\.dat", fichier)
    return int(match.group(1)) if match else float('inf')


def get_df(folder_path):

    original_cwd = os.getcwd()
    os.chdir(folder_path)
    

    fichiers = glob.glob(f"u-*.dat")
    
    # Trier les fichiers par t
    fichiers = sorted(fichiers, key=extraire_t)

    # Lecture et concaténation des fichiers
    dfs = []
    for f in fichiers:
        t = extraire_t(f)
        
        # Lire le fichier en ignorant les lignes vides
        df = pd.read_csv(f, sep="\t", header=None, names=["x", "y", "ut", "ub"])
        
        # Ajouter la colonne du temps t
        df["t"] = t
        
        dfs.append(df)
    
    df_final = pd.concat(dfs, ignore_index=True)
    df_final = df_final.drop_duplicates()
 
 
    try:
        df_feux = pd.read_csv(f"fires.dat", sep="\t", header=None,
                          names=["t", "x", "y", "r", "I"])
    except FileNotFoundError:
        df_feux = pd.DataFrame()
        pass
    
    os.chdir(original_cwd)
    # on renvoie le df et le pas de temps maximal
    return df_final, df_feux

