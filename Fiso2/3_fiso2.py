import sys
import pygame_gui
import pygame
import cv2
import pandas as pd
import pygame_gui
import pygame
import numpy as np
from scipy.spatial import Delaunay
import math
import random
import cv2
import os
import re
import shutil
import subprocess
import time
import glob

########################################################################################
# ----------------------- Définition des constantes / semi-variables
########################################################################################

SIM_NAME = sys.argv[1]
SIM_FOLDER = os.path.join("Simulations", SIM_NAME)
OUTPUT_FOLDER = os.path.join(SIM_FOLDER, 'output')
PARAMS_PATH = os.path.join(SIM_FOLDER, 'params.txt')
FRAMES_FOLDER = os.path.join(SIM_FOLDER, 'Frames')

# Généralités pygame
SCREEN_WIDTH = 1680
SCREEN_HEIGHT = 937
TPS = 10  # ticks par seconde

# Paramètres rendu isométrique
ISO_SCALE = 15
ANGLE = 25
COS = math.cos(math.radians(ANGLE))
SIN = math.sin(math.radians(ANGLE))

rot = -110
cos_rot = math.cos(math.radians(rot))
sin_rot = math.sin(math.radians(rot))

ORIGIN_X = SCREEN_WIDTH // 2.3
ORIGIN_Y = SCREEN_HEIGHT

# Paramètres visuels densité des arbres
MAX_ELMTS_PER_TILE = 40
AREA_SCALE = 1  # hectares par pixel
ELMT_SIZE = 10  # taille des assets blittés en pixel

# Paramètres feux
FIRE_DURATION = 5  # durée en pas de temps
DEAD_TREE_DURATION = 15

# Max d'arbres (réel)
TREES_T_MIN = 0
TREES_T_MAX = 30

# MAX T est valué par data_util lors du chargement des données (voir fonction "build_from_solution")
MAX_T = -1
MAX_ELMTS_PER_TILE -= MAX_ELMTS_PER_TILE % 2 # checkup divisible par 2

# constantes couleurs
BROWN = (143, 50, 43)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DARK_GREEN = (35, 94, 35)
LIGHT_GREEN = (86, 179, 86)
RED = (255, 0, 0)
SNOW = (197, 205, 217)
ICE = (160, 190, 235)

########################################################################################
# ----------------------- Initialisation de Pygame et du videoWriter
########################################################################################

print("Program started.")
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption('IsoForest2')


video_path = os.path.join(SIM_FOLDER, f'{SIM_NAME}.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(video_path, fourcc, 45, (SCREEN_WIDTH, SCREEN_HEIGHT))


########################################################################################
# ----------------------- Chargement des assets et sources externes
########################################################################################

# gestion de l'affichage des paramètres
def load_text_from_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()


font = pygame.font.Font(None, 26)
text_content = load_text_from_file(PARAMS_PATH)
text_surface = font.render(text_content, True, BLACK)


def load_and_scale_image(image_path, size):
    image = pygame.image.load(image_path).convert_alpha()
    return pygame.transform.scale(image, (size, size))


images = {
    "mixte_tree": load_and_scale_image("assets/mixte.png", ELMT_SIZE),
    "boreal_tree": load_and_scale_image("assets/boreal.png", ELMT_SIZE),
    "dead_tree": load_and_scale_image("assets/deadtree.png", ELMT_SIZE),
    "burning_tree": load_and_scale_image("assets/burning_tree.png", ELMT_SIZE),
}


########################################################################################
# ----------------------- Gestion rendu isométrique
########################################################################################

# utilisé dans Tile.draw_tile. Les coordonnées de Nodes et Fires sont cartésiennes.
def projeter_en_isometrique(x, y):
    cx = x
    cy = y
    
    xr = (cx * cos_rot) - (cy * sin_rot)
    yr = (cx * sin_rot) + (cy * cos_rot)
    
    x_proj = (-xr - yr) * COS * ISO_SCALE + ORIGIN_X
    y_proj = (-xr + yr) * SIN * ISO_SCALE + ORIGIN_Y
    
    return x_proj, y_proj


########################################################################################
# ----------------------- Construction de l'UI
########################################################################################

manager = pygame_gui.UIManager((SCREEN_WIDTH, SCREEN_HEIGHT))
ui_height = 25

slider_rect = pygame.Rect((10, SCREEN_HEIGHT - ui_height),
                          (SCREEN_WIDTH - 20, ui_height))
play_button_rect = pygame.Rect((10, SCREEN_HEIGHT - ui_height - 60 - 10), (60, 60))

slider = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=slider_rect,
    start_value=0,
    value_range=(0, 0),  # value range adaptée à MAX_T dans main seulement (MAX_T pas encore valué)
    manager=manager,
)

play_button = pygame_gui.elements.UIButton(
    relative_rect=play_button_rect,
    text="autoplay",
    manager=manager,
)

########################################################################################
# ----------- Importation des résultats FreeFem++ et construction des nodes
########################################################################################


# Node : sommet de maille, avec vecteurs des valeurs en fonction du temps
class Node:
    """ Une node est d'abord un point réel / non-interpolé, donné par la sortie freefem"""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.ub = None
        self.um = None
        
        self.fires = None


# Fonction pour extraire le temps t du nom de fichier
def extraire_t(fichier):
    match = re.search(r"u-(\d+)\.dat", fichier)
    return int(match.group(1)) if match else float('inf')


def get_df():
    
    original_cwd = os.getcwd()

    # d'abord les feux, qui sont dans SIM_FOLDER
    os.chdir(SIM_FOLDER)
    try:
        df_feux = pd.read_csv(f"fires.dat", sep="\t", header=None,
                              names=["t", "x", "y", "r", "I"])
    except FileNotFoundError:
        df_feux = pd.DataFrame()
        
    # puis on va dans "output" qui contient les données "u-t.dat"
    os.chdir('output')
    
    fichiers = glob.glob(f"u-*.dat")
    
    # Trier les fichiers par t
    fichiers = sorted(fichiers, key=extraire_t)
    
    # Lecture et concaténation des fichiers
    dfs = []
    for f in fichiers:
        t = extraire_t(f)
        
        # Lire le fichier en ignorant les lignes vides
        df = pd.read_csv(f, sep="\t", header=None, names=["x", "y", "um", "ub"])
        
        # Ajouter la colonne du temps t
        df["t"] = t
        
        dfs.append(df)
    
    df_final = pd.concat(dfs, ignore_index=True)
    df_final = df_final.drop_duplicates()
    
    # retour dans le dossier de fiso2
    os.chdir(original_cwd)
    # on renvoie le df et le pas de temps maximal
    return df_final, df_feux



# build_from_solution : construction des nodes à partir du dossier de sortie FreeFem++
def build_from_solution():
    global MAX_T
    
    print(f"Building from solution folder \"{SIM_FOLDER}\"...")
    nodes__ = []
    
    # fonction get_df de Model.data_util.py
    df, df_feux = get_df()
    MAX_T = df["t"].max()
    # nota : le df_feux est passé en retour, il n'est pas utilisé avant que les mailles soient définies
    
    print(f"\tMax time is {MAX_T}")
    
    ub_min, ub_max = df["ub"].min(), df["ub"].max()
    um_min, um_max = df["um"].min(), df["um"].max()
    
    unique_xy_list = list(df[['x', 'y']].drop_duplicates().itertuples(index=False, name=None))
    
    for xy in unique_xy_list:
        x = xy[0]
        y = xy[1]
        node = Node(x, y)
        
        filtered_df = df[(df['x'] == x) & (df['y'] == y)]
        
        node.ub = filtered_df['ub'].to_numpy()
        node.um = filtered_df['um'].to_numpy()
        
        nodes__.append(node)
    
    print(f"\tBuilt {len(nodes__)} nodes.")
    print(f"\tub_min = {ub_min}")
    print(f"\tub_max = {ub_max}")
    print(f"\tum_min = {um_min}")
    print(f"\tum_max = {um_max}")
    print("\n")
    
    return nodes__, df_feux


# generate_intermediate_nodes : optionnel, pour affiner le maillage (création de nodes moyennes)
# attention, ça veut dire 4(?) fois plus de mailles à gérer !
def generate_intermediate_nodes(nodes):
    """ Génère des nœuds intermédiaires en utilisant Delaunay et les milieux des côtés des triangles. """
    
    print("Generating intermediate nodes...")
    points = np.array([(node.x, node.y) for node in nodes])
    tri = Delaunay(points)
    
    new_nodes = []
    existing_nodes = {(node.x, node.y): node for node in nodes}
    
    for simplex in tri.simplices:
        # Créer des nœuds intermédiaires sur les bords des triangles
        for i in range(3):
            p1 = nodes[simplex[i]]
            p2 = nodes[simplex[(i + 1) % 3]]
            mid_x = (p1.x + p2.x) / 2
            mid_y = (p1.y + p2.y) / 2
            
            if (mid_x, mid_y) not in existing_nodes:
                new_node = Node(mid_x, mid_y)
                # Interpolation des valeurs des nœuds voisins (u, v, w)
                new_node.ut = (p1.ut + p2.ut) / 2
                new_node.ub = (p1.ub + p2.ub) / 2
                
                existing_nodes[(mid_x, mid_y)] = new_node
                new_nodes.append(new_node)
    
    print("\tGenerated intermediate nodes.")
    print("\n")
    return new_nodes


########################################################################################
# ----------- Gestion des mailles à partir des nodes récupérées ci-dessus
########################################################################################

# triangle_area : calcul de l'aire d'un triangle (= d'une maille)
def triangle_area(x1, y1, x2, y2, x3, y3):
    """ Calcule l'aire d'un triangle à partir de ses sommets """
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)


class Fire:
    
    def __init__(self, t_start, x, y, r, I):
        self.t_start = t_start
        # coordonnées cartésiennes
        self.x = x
        self.y = y
        self.r = r
        self.I = I


# class::Tree : arbre avec sa position (isométrique) son type et affiché ou non. Méthode d'affichage incluse.
class Tree:
    
    def __init__(self, state, *pos):
        
        self.pos_x = pos[0]
        self.pos_y = pos[1]
        
        self.iso_pos = (pos[0] - ELMT_SIZE // 2, pos[1] - ELMT_SIZE)
        # états : 'hide', 'mixte', 'boreal', 'burning', 'dead'
        self.states = ['hide'] * MAX_T
    
    # affichage de l'arbre en fonction du state et du temps
    def blit(self, screen, t):
        if self.states[t] == 'boreal':
            screen.blit(images["boreal_tree"], self.iso_pos)
        elif self.states[t] == 'mixte':
            screen.blit(images["mixte_tree"], self.iso_pos)
        elif self.states[t] == 'burning':
            screen.blit(images["burning_tree"], self.iso_pos)
        elif self.states[t] == 'dead':
            screen.blit(images["dead_tree"], self.iso_pos)
        elif self.states[t] == 'hide':
            pass


# class::Tile : Une maille. Vecteurs moyens des nodes qui la compose. Contient la liste de ses instances d'arbres,
# gère ses feux, l'affichage de ses arbres et son affichage (fond, bordure).
class Tile:
    """ Représente une zone polygonale (triangle) dans l'écran avec ses nœuds et ses propriétés."""
    
    def __init__(self, node1, node2, node3):
        
        # sommets centre et aire
        self.nodes = [node1, node2, node3]
        self.area = triangle_area(node1.x, node1.y, node2.x, node2.y, node3.x, node3.y)
        self.center = (
            (node1.x + node2.x + node3.x) / 3,
            (node1.y + node2.y + node3.y) / 3
        )
        
        # valeurs moyennes de la zone en fonction du temps
        self.um_avg = [np.mean([node1.um[t], node2.um[t], node3.um[t]]) for t in range(MAX_T)]  # Moyenne de ut
        self.ub_avg = [np.mean([node1.ub[t], node2.ub[t], node3.ub[t]]) for t in range(MAX_T)]  # Moyenne de ub
        
        # vrai nombre d'arbres dans la zone (fonction de l'aire)
        self.true_nb_mixtes = [um_avg_t * self.area * AREA_SCALE for um_avg_t in self.um_avg]
        self.true_nb_boreal = [ub_avg_t * self.area * AREA_SCALE for ub_avg_t in self.ub_avg]
        
        # nombre d'arbres à blitter (fonction du nombre max d'arbres à montrer)
        self.screen_nb_mixtes = [int((true_nb_mixes_t - TREES_T_MIN) * MAX_ELMTS_PER_TILE // 2
                                     / (TREES_T_MAX - TREES_T_MIN))
                                 for true_nb_mixes_t in self.true_nb_mixtes]
        self.screen_nb_boreal = [int((true_nb_boreal_t - TREES_T_MIN) * MAX_ELMTS_PER_TILE // 2
                                     / (TREES_T_MAX - TREES_T_MIN))
                                 for true_nb_boreal_t in self.true_nb_boreal]
        
        self.trees = []  # liste d'instance de trees
        self.generate_trees()
        random.shuffle(self.trees)
    
    # generate_trees : construction des positions d'abres (avec random_point_in_triangle) et affectation des positions
    # à des instances d'arbres. Par défaut, tous les arbres sont .show=False
    def generate_trees(self):
        
        ## 1 - génération des positions
        generated_points = []
        
        while len(generated_points) < MAX_ELMTS_PER_TILE:
            r1 = random.uniform(0, 1)
            r2 = random.uniform(0, 1)
            if r1 + r2 > 1:
                r1 = 1 - r1
                r2 = 1 - r2
            
            r3 = 1 - r1 - r2
            
            x = r1 * self.nodes[0].x + r2 * self.nodes[1].x + r3 * self.nodes[2].x
            y = r1 * self.nodes[0].y + r2 * self.nodes[1].y + r3 * self.nodes[2].y
            
            iso_x, iso_y = projeter_en_isometrique(x, y)
            iso_point = (int(iso_x), int(iso_y))
            
            if iso_point not in generated_points:
                generated_points.append(iso_point)
        
        ## 2- construction des arbres
        for point in generated_points:
            tree = Tree('hide', *point)
            self.trees.append(tree)
        
        # construction des showtimes
        for t in range(MAX_T):
            
            nb_boreals_t = self.screen_nb_boreal[t]
            nb_mixtes_t = self.screen_nb_mixtes[t]
            for tree in self.trees[:nb_boreals_t]:
                tree.states[t] = 'boreal'
            # contournement d'un défaut de slice index natif de python : [-0:] == [0:] !! >:(
            if -nb_mixtes_t != 0:
                for tree in self.trees[-nb_mixtes_t:]:
                    tree.states[t] = 'mixte'
    
    # dessin de la maille, et de son contour si décommenté ci-dessous
    def draw_tile(self, screen, t):
        
        color = SNOW
        points_iso = [projeter_en_isometrique(node.x, node.y) for node in self.nodes]
        pygame.draw.polygon(screen, SNOW, points_iso)
    
    def draw_tile_borders(self, screen):
        points_iso = [projeter_en_isometrique(node.x, node.y) for node in self.nodes]
        pygame.draw.polygon(screen, BLACK, points_iso, 1)
    
    # affichage des arbres ayant .show=True (appelé après draw_tiles pour bliter par dessus la maille)
    def draw_trees(self, screen, t):
        for tree in self.trees:
            tree.blit(screen, t)


# create_tiles : Création des mailles à partir des nodes (issues de build_from_solution a priori)
def create_tiles(nodes):
    print("Generating tiles...")
    # Extraire les coordonnées des nœuds
    points = np.array([(node.x, node.y) for node in nodes])
    
    # Effectuer la triangulation de Delaunay
    tri = Delaunay(points)
    tiles_ = []
    
    # Pour chaque triangle, calculer la valeur moyenne des nœuds et créer la zone
    for simplex in tri.simplices:
        # Extraire les indices des trois nœuds du triangle
        node1, node2, node3 = [nodes[i] for i in simplex]
        
        # Créer la zone polygonale et ajouter à la liste
        tile = Tile(node1, node2, node3)
        tiles_.append(tile)
    
    print(f"\tGenerated {len(tiles_)} tiles.")
    
    mean_area = np.mean([tile.area for tile in tiles_])
    max_true_nb_boreals = np.max([tile.true_nb_boreal for tile in tiles_])
    max_true_nb_mixtes = np.max([tile.true_nb_mixtes for tile in tiles_])
    max_screen_nb_boreals = np.max([tile.screen_nb_boreal for tile in tiles_])
    max_screen_nb_mixtes = np.max([tile.screen_nb_mixtes for tile in tiles_])
    
    print("\nTILES PROPERTIES")
    print(f"\tmean_area: {mean_area}")
    print(f"\tmax_true_nb_mixtes: {max_true_nb_mixtes}")
    print(f"\tmax_true_nb_boreal: {max_true_nb_boreals}")
    print(f"\tmax_screen_nb_mixtese: {max_screen_nb_mixtes}")
    print(f"\tmax_screen_nb_boreal: {max_screen_nb_boreals}\n")
    
    return tiles_


# resolve_burning_trees : fonction qui relie les feux et les tiles. Détermine les arbres qui brûlent.
def resolve_burning_trees(tiles, df_feux):
    ## 1 - construction des feux
    fires = []
    for _, fire in df_feux.iterrows():
        fire = Fire(fire["t"], fire["x"], fire["y"], fire["r"], fire["I"])
        fires.append(fire)
    
    ## 2 - affectation des feux aux arbres
    for fire in fires:
        
        # on pose les temps de transition pour aérer le code plus bas
        # pourquoi le +1 ? parce que lorsque t_start est à x, les arbres disparaissent à x+1 dans la simu.
        # on utilise toujours t_start pour regarder "eligible_trees" (cf plus bas) ; mais les images de feu doivent
        # commencer lorsque les arbres ont disparus, pour les remplacer, càd à x+1.
        
        fire_start = int(fire.t_start) + 1
        fire_end = min(fire_start + FIRE_DURATION, MAX_T)
        dead_start = fire_end
        dead_end = min(dead_start + DEAD_TREE_DURATION, MAX_T)
        
        for tile in tiles:
            
            dif_x = abs(tile.center[0] - fire.x)
            dif_y = abs(tile.center[1] - fire.y)
            
            # vérifier que le centre de la tuile est dans le feu = tuile en feu
            if dif_x ** 2 + dif_y ** 2 < fire.r ** 2:
                
                # arbres eligibles à brûler si pas 'hide' ou 'dead' au temps du feu
                eligible_trees = [tree for tree in tile.trees
                                  if tree.states[int(fire.t_start) - 1] not in ['hide', 'dead']]
                
                nb_to_burn = int(len(eligible_trees) * fire.I)  # nb d'arbres qui brûlent parmi les eligibles
                selected_trees = random.sample(eligible_trees, nb_to_burn)
                
                for tree in selected_trees:
                    for fire_time in range(fire_start, fire_end):
                        tree.states[fire_time] = 'burning'
                    for dead_time in range(dead_start, dead_end):
                        tree.states[dead_time] = 'dead'


########################################################################################
# ------------------------ Main
########################################################################################


if __name__ == "__main__":
    
    # C'est parti
    print(" ###### STARTED.")
    print("Running main.\n")
    
    nodes_, df_feux = build_from_solution()
    # décommenter ci-dessous pour affiner le maillage (attention au coût de calcul : 4* plus de mailles !)
    # nodes_ +=  generate_intermediate_nodes(nodes_)
    tiles = create_tiles(nodes_)
    resolve_burning_trees(tiles, df_feux)
    # finaliser la construction du slider (MAX_T dépend de la simulation)
    slider.value_range = (0, MAX_T - 1)
    
    
    if os.path.exists(FRAMES_FOLDER):
        shutil.rmtree(FRAMES_FOLDER)
    os.makedirs(FRAMES_FOLDER)
    
    ########################################################################################
    # -------------------- Boucle principale de pygame
    ########################################################################################
    
    clock = pygame.time.Clock()
    running = True
    play_mode = True
    said_last_words = False
    
    t = 0
    FPS = 60
    
    # tick_it permet de régler la vitesse de simulation sans affecter le vrai fps de pygame (= GUI fluide)
    tick_it = 1
    
    print("Running pygame loop...")
    while running:
        
        screen.fill(ICE)
    
        if play_mode:
            # en play_mode (default) le slider suit la simulation
            slider.set_current_value(t)
            # tick_it augmente. Si tick_it est 0 modulo truc, avancer d'un pas de temps, et remettre tick à 0
            tick_it += 1
            if not tick_it % (FPS // TPS):
                t = min(t + 1, MAX_T - 1)
                tick_it = 1
        else:
            # hors play_mode, c'est le slider qui contrôle le temps
            t = int(slider.get_current_value())
        
        ##########################################
        ## Partie affichage
        ##########################################
        
        for tile in tiles:
            tile.draw_tile(screen, t)
        """
        for tile in tiles:
            tile.draw_tile_borders(screen)
        """
        for tile in tiles:
            tile.draw_trees(screen, t)
        
        # affichage des paramètres
        # screen.blit(text_surface, (20, 50))
        
        # affichage de t
        t_surface = font.render(f"T = {t / 10:.1f}", True, BLACK)
        screen.blit(t_surface, (SCREEN_WIDTH // 2 - 30, 50))
        
        ##########################################
        ## Gestion fin de simulation
        ##########################################
        
        # si t est maximum, on affiche un cercle rouge, on arrête l'enregistrement vidéo
        if t == MAX_T - 1:
            
            pygame.draw.circle(screen, (255, 0, 0), (SCREEN_WIDTH - 40, 40), 30)
            
            if not said_last_words:
                print(f"Max time ({t + 1}) reached.")
                print("Writing video...")
                video_writer.release()
                print(f"\t Wrote video as \"{video_path}\".\n")
                said_last_words = True
        
        # sinon, on continue d'enregistrer la video
        else:
            frame = pygame.surfarray.array3d(screen)
            # transformations pour OpenCV
            frame = np.rot90(frame)
            frame = np.flipud(frame)
            # conversion OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # écriture de la frame
            video_writer.write(frame)
            
            # Sauvegarde en JPEG (plan B)
            try:
                frame_filename = os.path.join(FRAMES_FOLDER, f"frame_{t}.jpg")
                cv2.imwrite(frame_filename, frame)
            except NameError:
                pass
            
        ##########################################
        ## Gestion des évenements
        ##########################################
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # si le slider bouge, on arrête le play mode
            elif event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                if event.ui_element == slider:
                    if said_last_words:
                        play_mode = False
            # on alterne play_mode lorsqu'on clique le bouton associé
            elif event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == play_button:
                    if said_last_words:
                        play_mode = not play_mode
            
            manager.process_events(event)
        
        ###########################################
        
        time_delta = clock.tick(FPS) / 1000.0
        
        manager.update(time_delta)
        manager.draw_ui(screen)
        
        pygame.display.flip()
    
    pygame.quit()
    print("###### END OF PROGRAM.")

