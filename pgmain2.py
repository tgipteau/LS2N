import pygame
import pygame_gui
import numpy as np
import pandas as pd

from scipy.spatial import Delaunay

# Chargement fichier de config
import yaml
config = yaml.load(open('pg_config.yaml', 'r'), Loader=yaml.FullLoader)


# Initialisation de Pygame
pygame.init()

    
# Création de la fenêtre
screen = pygame.display.set_mode((config['screen_width'], config['screen_height']))
pygame.display.set_caption(config['caption'])


# Initialisation de pygame_gui
manager = pygame_gui.UIManager((config['screen_width'], config['screen_height']))

# Création du curseur (Slider)
slider = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=pygame.Rect(10, config['screen_height'] - 50, 300, 20),
    start_value=0,  # Valeur de départ
    value_range=(0, config['max_time']),  # Plage de valeurs (min, max)
    manager=manager
)

class Node:
    """ Une node est un point non-interpolé, donné par la sortie freefem"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.u = None
        self.v = None
        self.w = None
        
    def draw(self, screen, t=0):
        
        # Normalisation de u pour obtenir une opacité entre 50 et 255
        u_min, u_max = 3, 6  # À ajuster selon tes données réelles
        alpha = int(np.interp(self.u[t], [u_min, u_max], [50, 255]))
        
        # Création d'une surface transparente
        radius = config['node_radius']
        alpha_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        
        # Dessiner le cercle avec la couleur de config et l'alpha
        pygame.draw.circle(alpha_surface,
                           (*config['node_color'], alpha),  # Décomposition RGB + alpha
                           (radius, radius),
                           radius)
        
        # Affichage sur l'écran (en centrant)
        screen.blit(alpha_surface, (self.x - radius, self.y - radius))


class Zone:
    """ Représente une zone polygonale (triangle) dans l'écran avec ses nœuds et ses propriétés."""
    
    def __init__(self, node1, node2, node3):
        self.nodes = [node1, node2, node3]
        self.color = None  # La couleur sera calculée plus tard
        self.u_avg = [np.mean([node1.u[t], node2.u[t], node3.u[t]]) for t in
                      range(len(node1.u))]  # Moyenne de u (ou toute autre valeur)
        self.update_color(0)
    
    def update_color(self, t):
        """ Met à jour la couleur de la zone en fonction de la valeur moyenne de u."""
        u_min, u_max = 3, 6  # À ajuster en fonction des données réelles
        normalized_u = np.interp(self.u_avg[t], [u_min, u_max], [0, 255])
        # Définir la couleur en fonction de u_avg
        self.color = (int(normalized_u), 0, int(255 - normalized_u))  # Dégradé du rouge au bleu
    
    def draw(self, screen, t):
        """ Dessine la zone polygonale sur l'écran."""
        self.update_color(t=t)
        pygame.draw.polygon(screen, self.color, [(node.x, node.y) for node in self.nodes])
    
    def add_sprite(self, sprite):
        """ Ajoute un sprite à la zone en fonction de sa valeur moyenne."""
        # Cette méthode peut être utilisée pour dessiner un sprite dans cette zone
        pass
    
    
def build_true_nodes_from_solution(filepath='Test_model/output/solution.txt'):
    nodes_ = []
    df = pd.read_csv(filepath, header=0, sep=' ')
    
    x_min, x_max = df["x"].min(), df["x"].max()
    y_min, y_max = df["y"].min(), df["y"].max()
    
    ratio_x = (config['screen_width'] * config['margin_alpha']) / (x_max - x_min)
    ratio_y = (config['screen_height'] * config['margin_alpha']) / (y_max - y_min)
    unique_xy_list = list(df[['x', 'y']].drop_duplicates().itertuples(index=False, name=None))
    
    for xy in unique_xy_list:
        x = xy[0]
        y = xy[1]
        
        screen_x = (x - x_min) * ratio_x + config['screen_width'] * (1 - config['margin_alpha']) / 2
        screen_y = (y - y_min) * ratio_y + config['screen_height'] * (1 - config['margin_alpha']) / 2
        
        node = Node(screen_x, screen_y)
        
        filtered_df = df[(df['x'] == x) & (df['y'] == y)]
        node.u = filtered_df['u(x,y,t)'].to_numpy()
        node.v = filtered_df['v(x,y,t)'].to_numpy()
        node.w = filtered_df['w(x,y,t)'].to_numpy()
        
        nodes_.append(node)
    
    return nodes_


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


def draw_text(screen, text, x=10, y=10, font_size=24, color=(255, 255, 200)):
    """Affiche un texte à l'écran à la position (x, y)."""
    font = pygame.font.Font(None, font_size)  # Police par défaut
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (x, y))





if __name__ == "__main__":
    
    true_nodes = build_true_nodes_from_solution()
    fake_nodes = generate_intermediate_nodes(true_nodes)
    zones = triangulate_and_create_zones(true_nodes + fake_nodes)
    
    # Boucle principale de Pygame
    running = True
    clock = pygame.time.Clock()
    FPS = 40
    t = 0

    while running:
        screen.fill(tuple(config['bg_color']))
        
        t = int(slider.get_current_value())
        
        for zone in zones:
            zone.draw(screen, t)
        
        manager.update(clock.get_time())
        manager.draw_ui(screen)
        
        pygame.display.flip()  # Mettre à jour l'affichage
        clock.tick(FPS)
        
        # Gérer les événements (fermer la fenêtre)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            # Passer les événements à pygame_gui pour gérer les interactions avec l'UI
            manager.process_events(event)
    
    pygame.quit()