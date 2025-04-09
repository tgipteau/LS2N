import os
import shutil
import subprocess
import sys
import time

########################################################################################
# ----------------------- Lancement de simulations par FreeFEM++
########################################################################################


if __name__ == "__main__":
    
    sim_folder = sys.argv[1]
    output_folder = os.path.join(sim_folder, "output")
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Copier le script FreeFem dans le dossier de sortie
    script_dest = os.path.join(sim_folder, os.path.basename("model.edp"))
    shutil.copy("model.edp", script_dest)
    
    original_cwd = os.getcwd()  # Enregistrer le répertoire d'origine
    os.chdir(sim_folder)
    
    print("\n------------------------")
    print("New simulation.")
    
    params = {}
    with open("params.txt", 'r') as f:
        for ligne in f:
            # Enlever les espaces et les retours à la ligne
            ligne = ligne.strip()
            
            # Ignore les lignes vides ou les commentaires
            if not ligne or ligne.startswith("#"):
                continue
            
            # Séparer la ligne en clé et valeur
            cle, valeur = ligne.split(":", 1)
            params[cle.strip()] = valeur.strip()
        
    print("...Loaded params.")
    
    print("Output will be moved to ", output_folder)
    print(os.getcwd())
    # Vérifier si FreeFem++-CoCoa est disponible, sinon lancer FreeFem++ classique
    try:
        if subprocess.call(["which", "FreeFem++-CoCoa"]) == 0:  # Vérifier la présence de FreeFem++-CoCoa
            # Lancer FreeFem++-CoCoa avec les résultats redirigés vers le dossier
            subprocess.run(
                ['FreeFem++-CoCoa', os.path.basename("model.edp"), '-glut',
                 '/Applications/FreeFem++.app/Contents/ff-4.15/bin/ffglut']
            )
            print(f"FreeFem script {"model.edp"} launched with FreeFem++-CoCoa.")
        else:
            # Lancer FreeFem++ classique sinon, avec les résultats redirigés vers le dossier
            subprocess.run(['FreeFem++', os.path.basename("model.edp")])
            print(f"FreeFem script {"model.edp"} launched with FreeFem++ (not CoCoa).")
    except subprocess.CalledProcessError as e:
        print(f"Error while trying to run script {"model.edp"}: {e}")
        exit()

    quit()

    

    
