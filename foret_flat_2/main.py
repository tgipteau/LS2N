import time
import pygame
import pygame_gui
import pygame
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
import math
import random
import io
from Model.data_util import get_df

########################################################################################
# ----------------------- Définition des constantes
########################################################################################


SCREEN_WIDTH = 1680
SCREEN_HEIGHT = 937
MARGIN = 0.1

ISO_SCALE = 16
ANGLE = 30
COS = math.cos(math.radians(ANGLE))
SIN = math.sin(math.radians(ANGLE))
ORIGIN_X = SCREEN_WIDTH // 1.5
ORIGIN_Y = 20

MAX_ELMTS_PER_TILE = 50  # == 0 % 2
AREA_SCALE = 1  # hectares par pixel
ELMT_SIZE = 8  # taille des assets blittés en pixel

FPS = 45
FIRE_DURATION = 2 # durée en secondes

# Max d'arbres (réel)
TREES_T_MIN = 0
TREES_T_MAX = 22

# MAX T est valué par data_util lors du chargement des données (voir fonction "build_from_solution")
MAX_T = -1
LOSSES = []

# constantes couleurs
BG_FOREST = (143, 50, 43)
BG = (0, 0, 0)
BLACK = (0, 0, 0)
DARK_GREEN = (35, 94, 35)
LIGHT_GREEN = (86, 179, 86)
RED = (255, 0, 0)

########################################################################################
# ----------------------- Initialisation de Pygame
########################################################################################

print("Program started.")
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption('Foret Flat 2')


########################################################################################
# ----------------------- Chargement des assets et sources externes
########################################################################################


def load_and_scale_image(image_path, size):
    image = pygame.image.load(image_path).convert_alpha()
    return pygame.transform.scale(image, (size, size))


images = {
    "temperate_tree": load_and_scale_image("../assets/temperate.png", ELMT_SIZE),
    "boreal_tree": load_and_scale_image("../assets/boreal.png", ELMT_SIZE),
    "fire_tree": load_and_scale_image("../assets/fire.png", ELMT_SIZE),
}


########################################################################################
# ----------------------- Construction du slider et du bouton play
########################################################################################

manager = pygame_gui.UIManager((SCREEN_WIDTH, SCREEN_HEIGHT))
ui_height = 25

slider_rect = pygame.Rect((10, SCREEN_HEIGHT-ui_height),
                                  (SCREEN_WIDTH-20, ui_height))
play_button_rect = pygame.Rect((10, 10), (ui_height, ui_height))

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
# ----------- Importation des résultats FreeFem++ et construction de la carte
########################################################################################

## On obtient les noeuds depuis FreeFem++ (points des éléments finis)

class Node:
    """ Une node est d'abord un point réel / non-interpolé, donné par la sortie freefem"""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.ut = None
        self.ub = None


def build_from_solution():
    global MAX_T
    
    print("Building from solution...")
    nodes_ = []
    
    # fonction get_df de Model.data_util.py
    df, MAX_T = get_df('./Model/data-raw')
    
    print(f"Max time is {MAX_T}")
    ub_min, ub_max = df["ut"].min(), df["ut"].max()
    vb_min, vb_max = df["ub"].min(), df["ub"].max()
    
    unique_xy_list = list(df[['x', 'y']].drop_duplicates().itertuples(index=False, name=None))
    
    for xy in unique_xy_list:
        x = xy[0]
        y = xy[1]
        node = Node(x, y)
        
        filtered_df = df[(df['x'] == x) & (df['y'] == y)]
        node.ut = filtered_df['ut'].to_numpy()
        node.ub = filtered_df['ub'].to_numpy()
        
        nodes_.append(node)
    
    print(f"\tBuilt {len(nodes_)} nodes.")
    return nodes_, ub_min, ub_max, vb_min, vb_max


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
    return new_nodes


## On a les noeuds, on calcule les tuiles (triangles) avec leur aire et valeur moyenne des sommets

def triangle_area(x1, y1, x2, y2, x3, y3):
    """ Calcule l'aire d'un triangle à partir de ses sommets """
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)


def random_iso_point_in_triangle(self, nodes):
    """ Génère un point aléatoire à l'intérieur du triangle défini par les nœuds. """
    r1 = random.random()
    r2 = random.random()
    if r1 + r2 > 1:
        r1 = 1 - r1
        r2 = 1 - r2
    
    r3 = 1 - r1 - r2
    
    x = r1 * nodes[0].x + r2 * nodes[1].x + r3 * nodes[2].x
    y = r1 * nodes[0].y + r2 * nodes[1].y + r3 * nodes[2].y
    
    return projeter_en_isometrique(int(x), int(y))


class Tree:
    
    def __init__(self, type, *pos):
        self.show = False
        self.iso_pos = (pos[0] - ELMT_SIZE // 2, pos[1] - ELMT_SIZE)
        self.type = type
    
    def blit(self, screen):
        if self.show:
            if self.type == 'boreal':
                screen.blit(images["boreal_tree"], self.iso_pos)
            elif self.type == 'temperate':
                screen.blit(images["temperate_tree"], self.iso_pos)


class Tile:
    """ Représente une zone polygonale (triangle) dans l'écran avec ses nœuds et ses propriétés."""
    
    def __init__(self, node1, node2, node3):
        
        self.nodes = [node1, node2, node3]
        self.area = triangle_area(node1.x, node1.y, node2.x, node2.y, node3.x, node3.y)
        
        # valeurs moyennes de la zone en fonction du temps
        self.ut_avg = [np.mean([node1.ut[t], node2.ut[t], node3.ut[t]]) for t in range(MAX_T)]  # Moyenne de ut
        self.ub_avg = [np.mean([node1.ub[t], node2.ub[t], node3.ub[t]]) for t in range(MAX_T)]  # Moyenne de ub
        
        # vrai nombre d'arbres dans la zone
        self.true_nb_trees_temperate = [ut_avg_t * self.area * AREA_SCALE for ut_avg_t in self.ut_avg]
        self.true_nb_trees_boreal = [ub_avg_t * self.area * AREA_SCALE for ub_avg_t in self.ub_avg]
        
        # nombre d'arbres à blitter (normalisé)
        self.screen_nb_trees_temperate = [int((true_nb_trees_temperate_t - TREES_T_MIN) * MAX_ELMTS_PER_TILE // 2
                                              / (TREES_T_MAX - TREES_T_MIN))
                                          for true_nb_trees_temperate_t in self.true_nb_trees_temperate]
        self.screen_nb_trees_boreal = [int((true_nb_trees_boreal_t - TREES_T_MIN) * MAX_ELMTS_PER_TILE // 2
                                           / (TREES_T_MAX - TREES_T_MIN))
                                       for true_nb_trees_boreal_t in self.true_nb_trees_boreal]
        
        self.on_fire = 0
        
        # construction des arbres et update initial
        self.temperate_trees = []  # liste d'instances de Tree
        self.boreal_trees = []
        self.generate_trees()
        self.update(t=0)
    
    def generate_trees(self):
        
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
        
        # première moitié des points : arbres tempérés (show=False par défaut)
        for point in generated_points[:MAX_ELMTS_PER_TILE // 2]:
            self.temperate_trees.append(Tree('temperate', *point))
        # deuxième moitié : boreal
        for point in generated_points[MAX_ELMTS_PER_TILE // 2:]:
            self.boreal_trees.append(Tree('boreal', *point))
    
    def detect_fire(self, t):
        
        global LOSSES
        
        if t < MAX_T - 2:
            loss_factor = self.true_nb_trees_temperate[t + 1] / self.true_nb_trees_temperate[t]
            LOSSES.append(loss_factor)
            if  loss_factor < 0.8:
                print("Fire detected")
                # on considère qu'il y a un feu au temps t
                self.on_fire = FIRE_DURATION * FPS
    
    def update(self, t):
        
        self.detect_fire(t)
        
        # réinitialiser tout le monde à show=False
        for tree in self.boreal_trees + self.temperate_trees:
            tree.show = False
        
        nb_temperate_t = self.screen_nb_trees_temperate[t]
        nb_boreal_t = self.screen_nb_trees_boreal[t]
        
        # Afficher nb_foo_t arbres
        for boreal_tree in self.boreal_trees[:nb_boreal_t]:
            boreal_tree.show = True
        for temperate_tree in self.temperate_trees[:nb_temperate_t]:
            temperate_tree.show = True
    
    def draw_tile(self, screen, t):
        
        if self.on_fire:
            color = RED
            self.on_fire -= 1
        
        else:
            if self.true_nb_trees_temperate[t] > self.true_nb_trees_boreal[t]:
                # majorité tempérés
                color = LIGHT_GREEN
            else:
                color = DARK_GREEN
        
        points_iso = [projeter_en_isometrique(node.x, node.y) for node in self.nodes]
        pygame.draw.polygon(screen, color, points_iso)
        # ci dessous, affichage de bordure de tuile
        pygame.draw.polygon(screen, BLACK, points_iso, 1)
    
    def draw_trees(self, screen):
        for i in range(MAX_ELMTS_PER_TILE // 2):
            self.boreal_trees[i].blit(screen)
            self.temperate_trees[i].blit(screen)


def triangulate_and_create_tiles(nodes):
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
    
    print("\tGenerated tiles.")
    
    mean_area = np.mean([tile.area for tile in tiles_])
    max_true_nb_temperate = np.max([tile.true_nb_trees_temperate for tile in tiles_])
    max_true_nb_boreal = np.max([tile.true_nb_trees_boreal for tile in tiles_])
    max_screen_nb_temperate = np.max([tile.screen_nb_trees_temperate for tile in tiles_])
    max_screen_nb_boreal = np.max([tile.screen_nb_trees_boreal for tile in tiles_])
    
    print(f"mean_area: {mean_area}")
    print(f"max_true_nb_temperate: {max_true_nb_temperate}")
    print(f"max_true_nb_boreal: {max_true_nb_boreal}")
    print(f"max_screen_nb_temperate: {max_screen_nb_temperate}")
    print(f"max_screen_nb_boreal: {max_screen_nb_boreal}")
    
    return tiles_


## On définit des fonctions pour gérer la projection isométrique

def projeter_en_isometrique(x, y):
    x_proj = (x - y) * COS * ISO_SCALE + ORIGIN_X
    y_proj = (x + y) * SIN * ISO_SCALE + ORIGIN_Y
    
    return x_proj, y_proj


########################################################################################
# ------------------------ Main
########################################################################################


if __name__ == "__main__":
    
    print("Running main.")
    if MAX_ELMTS_PER_TILE % 2:
        raise ValueError("MAX_ELMTS_PER_TILE must be divisible by 2")
    
    clock = pygame.time.Clock()
    
    ## Utilisation des fonctions de création de la carte
    nodes_, ub_min, ub_max, vb_min, vb_max = build_from_solution()
    more_nodes = nodes_  #+ generate_intermediate_nodes(nodes_)
    tiles = triangulate_and_create_tiles(more_nodes)
    
    ########################################################################################
    # -------------------- Boucle principale de pygame
    ########################################################################################
    
    running = True
    play_mode = True
    t = 0
    
    print("Running pygame loop.")
    while running:
        
        slider.value_range = (0, MAX_T-1)  # Nouveau range, on avait pas MAX_T avant..
        
        if play_mode:
            slider.set_current_value(t)
            t = min(t + 1, MAX_T-1)
        else:
            t = int(slider.get_current_value())
        
        ##########################################
        ## Partie affichage
        screen.fill(BG)
        
        for tile in tiles:
            tile.draw_tile(screen, t)
        for tile in tiles:
            tile.update(t)
            tile.draw_trees(screen)
        
        manager.update(clock.get_time())
        manager.draw_ui(screen)
        
        ###########################################
           
        if t == MAX_T:
            pygame.draw.circle(screen, (255, 0, 0), projeter_en_isometrique(0, 0), 30)
        
        
        ######################################
   
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                if event.ui_element == slider:
                    play_mode = False
            elif event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == play_button:
                    play_mode = not play_mode
        
            manager.process_events(event)
        

        pygame.display.flip()
        clock.tick(FPS)
    pygame.quit()
