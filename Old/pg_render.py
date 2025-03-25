""" Module chargé des travaux de rendus.
 Dont dépend pg_main."""

from matplotlib import pyplot as plt

### Imports et chargement de la config yaml
import pygame
import yaml
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
import random
import io

config = yaml.load(open('pg_config.yaml', 'r'), Loader=yaml.FullLoader)


### Chargement des assets dans le cache

def load_and_scale_image(image_path, size):
    image = pygame.image.load(image_path).convert_alpha()
    return pygame.transform.scale(image, size)


# Taille cible pour chaque image
elmt_size = (config['elmt_size'], config['elmt_size'])
fire_size = (config['fire_size'], config['fire_size'])
"""
# Chargement et redimensionnement des images
images = {
    "young_boreal": load_and_scale_image("assets/young_boreal.png", elmt_size),
    "old_boreal": load_and_scale_image("assets/old_boreal.png", elmt_size),
    "seed": load_and_scale_image("assets/seed_big.png", elmt_size),
    "fire": load_and_scale_image("assets/fire.png", fire_size),
}
"""
images = {
    "young_boreal": load_and_scale_image("../assets/temperate.png", elmt_size),
    "old_boreal": load_and_scale_image("../assets/boreal.png", elmt_size),
    "seed": load_and_scale_image("../assets/seed.png", elmt_size),
    "fire": load_and_scale_image("../assets/fire.png", fire_size),
}

"""
images = {
    "young_boreal": load_and_scale_image("assets/biout/young_boreal.png", elmt_size),
    "old_boreal": load_and_scale_image("assets/biout/old_boreal.png", elmt_size),
    "seed": load_and_scale_image("assets/biout/seed.png", elmt_size),
    "fire": load_and_scale_image("assets/fire.png", fire_size),
}
"""

def triangle_area(x1, y1, x2, y2, x3, y3):
    """ Calcule l'aire d'un triangle à partir de ses sommets """
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)


### Définition des classes
class Node:
    """ Une node est d'abord un point réel / non-interpolé, donné par la sortie freefem"""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.u = None
        self.v = None
        self.w = None

        self.color = (0,0,0)
        
                
    def draw(self, screen, show_nodes):
        if show_nodes :
            pygame.draw.circle(screen, self.color, (self.x, self.y), 3)
    
    
    """
    def generate_counts(self, u_min, u_max,
                                v_min, v_max,
                                w_min, w_max):
        self.counts_young = []
        self.counts_old = []
        self.counts_seed = []
        
        for t in range(config['max_time']):
            
            # combien d'élément à chaque temps pour représenter la densité
            N = 4
            self.counts_young.append(int((self.u[t] - u_min) * N / (u_max - u_min)))
            self.counts_old.append(int((self.v[t] - v_min) * N / (v_max - v_min)))
            self.counts_seed.append(int((self.w[t] - w_min) * N / (w_max - w_min)))
        
        plt.plot(self.u, self.w, self.v)
        plt.show()
    """
    

class Zone:
    """ Représente une zone polygonale (triangle) dans l'écran avec ses nœuds et ses propriétés."""
    
    def __init__(self, node1, node2, node3):
        self.nodes = [node1, node2, node3]
        
        # bounding de la zone
        self.x_min = int(min([node.x for node in self.nodes]))
        self.x_max = int(max([node.x for node in self.nodes]))
        self.y_min = int(min([node.y for node in self.nodes]))
        self.y_max = int(max([node.y for node in self.nodes]))
        
        self.centre = self.centre_zone()
        
        # valeurs moyennes de la zone en fonction du temps
        self.u_avg = [np.mean([node1.u[t], node2.u[t], node3.u[t]]) for t in range(len(node1.u))]  # Moyenne de u
        self.v_avg = [np.mean([node1.v[t], node2.v[t], node3.v[t]]) for t in range(len(node1.v))]  # Moyenne de v
        self.w_avg = [np.mean([node1.v[t], node2.v[t], node3.v[t]]) for t in range(len(node1.v))]  # Moyenne de v
        
        self.pos_young_boreal = []
        self.pos_old_boreal = []
        self.pos_seed = []
        self.fire = []
        # les positions sont générées dans la boucle main, après que build_from_solution
        # ait donné les densités min et max
        
    def handle_fires(self):
        
        for t in range(config['max_time']):
            loss_intensity = self.u_avg[t + 1] / self.u_avg[t]
            if loss_intensity < config['fire_threshold']:
                # il y a un feu au temps t
                self.fire.append(self.centre)
            else:
                self.fire.append(None)
    
    def generate_positions(self, u_min, u_max,
                                v_min, v_max,
                                w_min, w_max):
        """ Génère des positions aléatoires dans la zone polygonale en fonction de la densité. """
        
        for t in range(config['max_time']):
            
            self.pos_young_boreal.append([])
            self.pos_old_boreal.append([])
            self.pos_seed.append([])
            
            if self.fire[t] is None:
                # on n'affiche que si pas de feu
            
                # combien d'éléments de chaque type ?
                N = config['nb_elmts_as_max_density']
                count_young_boreal = int((self.u_avg[t] - u_min) * N / (u_max - u_min))
                count_old_boreal = int((self.v_avg[t] - v_min) * N / (v_max - v_min))
                count_seed = int((self.w_avg[t] - w_min) * N / (w_max - w_min))
                
                for _ in range(count_young_boreal):
                    x, y = self.random_point_in_triangle(self.nodes)
                    self.pos_young_boreal[t].append((x, y))
                
                for _ in range(count_old_boreal):
                    x, y = self.random_point_in_triangle(self.nodes)
                    self.pos_old_boreal[t].append((x, y))
                
                for _ in range(count_seed):
                    x, y = self.random_point_in_triangle(self.nodes)
                    self.pos_seed[t].append((x, y))
    
    
    def contains(self, x, y):
        # Sommets du triangle
        x1, y1 = self.nodes[0].x, self.nodes[0].y
        x2, y2 = self.nodes[1].x, self.nodes[1].y
        x3, y3 = self.nodes[2].x, self.nodes[2].y
        
        # Aire totale du triangle ABC
        area_ABC = triangle_area(x1, y1, x2, y2, x3, y3)
        
        # Aires des sous-triangles formés avec le point (x, y)
        area_PAB = triangle_area(x, y, x2, y2, x3, y3)
        area_PBC = triangle_area(x1, y1, x, y, x3, y3)
        area_PCA = triangle_area(x1, y1, x2, y2, x, y)
        
        # Vérification : si la somme des sous-aires est égale à l’aire totale
        return abs(area_ABC - (area_PAB + area_PBC + area_PCA)) < 1e-6  # Tolérance numérique
    
    
    def centre_zone(self):
        A, B, C = self.nodes[0], self.nodes[1], self.nodes[2]
        x_G = (A.x + B.x + C.x) / 3
        y_G = (A.y + B.y + C.y) / 3
        return x_G, y_G
  
  
    def random_point_in_triangle(self, nodes):
        """ Génère un point aléatoire à l'intérieur du triangle défini par les nœuds. """
        # Chaque coordonnée (x, y) est obtenue en calculant une combinaison convexe des 3 points du triangle
        # L'algorithme génère des coordonnées uniformément réparties à l'intérieur du triangle.
        # Le principe est d'utiliser les barycentriques (lambda1, lambda2, lambda3), où la somme de ces lambdas = 1.
        
        # Générer trois valeurs aléatoires entre 0 et 1, et les normaliser pour que leur somme soit 1
        r1 = random.random()
        r2 = random.random()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        
        r3 = 1 - r1 - r2
        
        x = r1 * nodes[0].x + r2 * nodes[1].x + r3 * nodes[2].x
        y = r1 * nodes[0].y + r2 * nodes[1].y + r3 * nodes[2].y
        
        return int(x), int(y)
    
    
    def draw_content(self, screen, t, show_young, show_old, show_seed):
        """ Dessine les éléments (arbres, graines, etc.) dans la zone."""
        if show_young:
            for pos in self.pos_young_boreal[t]:
                screen.blit(images["young_boreal"], pos)
        
        if show_old:
            for pos in self.pos_old_boreal[t]:
                screen.blit(images["old_boreal"], pos)
                
        if show_seed:
            for pos in self.pos_seed[t]:
                screen.blit(images["seed"], pos)
                
        if self.fire[t] is not None:
            pos = (self.fire[t][0]- fire_size[0]//2, self.fire[t][1]- fire_size[0]//2)
            screen.blit(images["fire"], pos)
                
            
    def draw_border(self, screen, color):
        points = [(node.x, node.y) for node in self.nodes]
        pygame.draw.polygon(screen, color, points, 4)
    


def build_from_solution(filepath=config['solution_file_path']):
    nodes_ = []
    df = pd.read_csv(filepath, header=0, sep=' ')
    
    x_min, x_max = df["x"].min(), df["x"].max()
    y_min, y_max = df["y"].min(), df["y"].max()
    
    u_min, u_max = df["u(x,y,t)"].min(), df["u(x,y,t)"].max()
    v_min, v_max = df["v(x,y,t)"].min(), df["v(x,y,t)"].max()
    w_min, w_max = df["w(x,y,t)"].min(), df["w(x,y,t)"].max()
    
    ratio_x = (config['view_width'] * config['margin_render']) / (x_max - x_min)
    ratio_y = (config['view_height'] * config['margin_render']) / (y_max - y_min)
    unique_xy_list = list(df[['x', 'y']].drop_duplicates().itertuples(index=False, name=None))
    
    for xy in unique_xy_list:
        x = xy[0]
        y = xy[1]
        
        screen_x = (x - x_min) * ratio_x + config['view_width'] * (1 - config['margin_render']) / 2
        screen_x += config['toolbar_width']
        screen_y = (y - y_min) * ratio_y + config['view_height'] * (1 - config['margin_render']) / 2
        
        node = Node(screen_x, screen_y)
        
        filtered_df = df[(df['x'] == x) & (df['y'] == y)]
        node.u = filtered_df['u(x,y,t)'].to_numpy()
        node.v = filtered_df['v(x,y,t)'].to_numpy()
        node.w = filtered_df['w(x,y,t)'].to_numpy()
        
        nodes_.append(node)
        
    
    return nodes_, u_min, u_max, v_min, v_max, w_min, w_max


def generate_intermediate_nodes(nodes):
    """ Génère des nœuds intermédiaires en utilisant Delaunay et les milieux des côtés des triangles. """
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
                new_node.u = (p1.u + p2.u) / 2
                new_node.v = (p1.v + p2.v) / 2
                new_node.w = (p1.w + p2.w) / 2
                
                existing_nodes[(mid_x, mid_y)] = new_node
                new_nodes.append(new_node)
    
    return new_nodes


def triangulate_and_create_zones(nodes):
    # Extraire les coordonnées des nœuds
    points = np.array([(node.x, node.y) for node in nodes])
    
    # Effectuer la triangulation de Delaunay
    tri = Delaunay(points)
    
    zones = []
    
    # Pour chaque triangle, calculer la valeur moyenne des nœuds et créer la zone
    for simplex in tri.simplices:
        # Extraire les indices des trois nœuds du triangle
        node1, node2, node3 = [nodes[i] for i in simplex]
        
        # Créer la zone polygonale et ajouter à la liste
        zone = Zone(node1, node2, node3)
        zones.append(zone)
    
    return zones


def get_series_from_zones(zones):
    
    u_avg = [np.mean([zone.u_avg[t] for zone in zones]) for t in range(config['max_time'])]
    v_avg = [np.mean([zone.v_avg[t] for zone in zones]) for t in range(config['max_time'])]
    w_avg = [np.mean([zone.w_avg[t] for zone in zones]) for t in range(config['max_time'])]
    
    return u_avg, v_avg, w_avg


def create_matplotlib_figure(u, v, w):
    """ Génère une figure Matplotlib avec trois sous-graphiques pour U, V et W. """
    fig, axes = plt.subplots(3, 1, figsize=(5, 8), dpi=100)  # 3 lignes, 1 colonne

    x = list(range(len(u)))  # Axe des abscisses basé sur la longueur de u

    # Tracé des séries sur trois subplots différents
    axes[0].plot(x, u, label="Young trees", color="lime", linestyle="--", marker="o", linewidth=0.5)
    axes[0].set_ylabel("U")
    axes[0].legend()
    axes[0].set_title("Evolution of selected zones")

    axes[1].plot(x, v, label="Old trees", color="darkgreen", linestyle="--", marker="s", linewidth=0.5)
    axes[1].set_ylabel("V")
    axes[1].legend()

    axes[2].plot(x, w, label="Seeds", color="brown", linestyle=":", marker="d", linewidth=0.5)
    axes[2].set_xlabel("Temps")
    axes[2].set_ylabel("W")
    axes[2].legend()

    # Ajuster la disposition des graphes pour éviter le chevauchement
    plt.tight_layout()

    return fig
    
    
def plot_to_surface(fig):
    """ Convertit une figure Matplotlib en surface Pygame. """
    
    # Sauvegarde de la figure en mémoire (format PNG)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')  # Suppression des marges inutiles
    plt.close(fig)
    buf.seek(0)
    
    # Convertir en image Pygame
    return pygame.image.load(buf)


def screen_plots_to(screen, als_rect, zones):
    
    u,v,w = get_series_from_zones(zones)
    fig = create_matplotlib_figure(u, v, w)
    surf = plot_to_surface(fig)
    screen.blit(surf, als_rect)
    