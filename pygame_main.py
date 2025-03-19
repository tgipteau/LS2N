import pygame
import numpy as np
import pandas as pd
from scipy.interpolate import griddata


# Initialisation de Pygame
pygame.init()

# Dimensions de la fenêtre et de la grille
WIDTH, HEIGHT = 800, 600
CELL_SIZE = 10  # Taille des cellules
ROWS, COLS = HEIGHT // CELL_SIZE, WIDTH // CELL_SIZE  # Nombre de cellules

# Création de la fenêtre
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulation Écosystème")

# Couleurs
BACKGROUND_COLOR = (245, 222, 139) # Sable
CELL_GRID_COLOR = (0, 0, 0)  # Noir


class Cellule:
    def __init__(self, x, y):
        # position haut-gauche
        self.x = x
        self.y = y
        self.data = None  # Données associées à la cellule (ex. valeur moyenne des points dans cette cellule)
    
    def draw(self, surface):
        """Affiche la cellule en fonction de sa donnée (par ex. couleur)."""
        color = (255, 0, 0)  # Par défaut rouge
        if self.data.size > 0 :
            # Change la couleur en fonction de la donnée
            print(self.data)
            color = (int((self.data.iloc[0,3]) * 40), 0, 0)  # Exemple simple : la couleur varie selon la donnée
        pygame.draw.rect(surface, color, (self.x * CELL_SIZE, self.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

"""
# Fonction pour lire les données et les transformer en DataFrame par cellule
def load_data(file_name):

    data = pd.read_csv(file_name, sep=' ', header=0)
    data.columns = ['time', 'x', 'y', 'u', 'v', 'w']  # Nommer les colonnes pour clarifier
    
    # Créer un dictionnaire de DataFrames pour chaque cellule
    cells_data = {}
    
    for ix in range(COLS):
        for iy in range(ROWS):
            cells_data[(ix, iy)] = pd.DataFrame(columns=['x', 'y', 'time', 'u', 'v', 'w'])
    
    # Assigner les données aux cellules appropriées
    for _, row in data.iterrows():
        x, y = float(row['x']), float(row['y'])
        
        # map from data range to grid range
        data_to_grid_ratio_x = (WIDTH*0.8)/(data['x'].max() - data['x'].min())
        data_to_grid_ratio_y = (HEIGHT*0.8)/(data['y'].max() - data['y'].min())
        
        x, y = (x*data_to_grid_ratio_x + WIDTH*0.2,
                y*data_to_grid_ratio_y + HEIGHT*0.2)
        
        
        cell_x = int(x // CELL_SIZE)
        cell_y = int(y // CELL_SIZE)
        
        # Ajouter les données à la cellule correspondante dans le dictionnaire
        if 0 <= cell_x < COLS and 0 <= cell_y < ROWS:
            cells_data[(cell_x, cell_y)] = pd.concat([cells_data[(cell_x, cell_y)], row.to_frame().T], ignore_index=True)
    
    return cells_data
"""


def load_data(file_name):
    """Charge les données depuis un fichier et les associe à une cellule, avec interpolation."""
    data = pd.read_csv(file_name, sep=' ', header=0)
    data.columns = ['time', 'x', 'y', 'u', 'v', 'w']  # Adapter les noms des colonnes si besoin
    print(data)
    # Déterminer les limites pour la mise à l’échelle
    x_min, x_max = data['x'].min(), data['x'].max()
    y_min, y_max = data['y'].min(), data['y'].max()
    
    # Mapping entre les données et la grille
    data_to_grid_ratio_x = (WIDTH * 0.8) / (x_max - x_min)
    data_to_grid_ratio_y = (HEIGHT * 0.8) / (y_max - y_min)
    
    # Créer une grille régulière pour l'interpolation
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, COLS),
        np.linspace(y_min, y_max, ROWS)
    )
    
    # Interpoler chaque variable d'intérêt
    grid_vars = {}
    for var in ['u', 'v', 'w']:  # Ajouter les variables pertinentes
        grid_vars[var] = griddata(
            (data['x'], data['y']),
            data[var],
            (grid_x, grid_y),
            method='nearest'
        )
    
    # Créer un dictionnaire de DataFrames pour chaque cellule
    cells_data = {(ix, iy): pd.DataFrame(columns=['x', 'y', 'time', 'u', 'v', 'w'])
                  for ix in range(COLS) for iy in range(ROWS)}
    
    # Assigner les données aux cellules
    for _, row in data.iterrows():
        x, y = row['x'], row['y']
        
        # Mise à l’échelle et mapping sur la grille
        x = x * data_to_grid_ratio_x
        y = y * data_to_grid_ratio_y
        cell_x = int(x // CELL_SIZE)
        cell_y = int(y // CELL_SIZE)
        
        # Ajouter aux cellules valides
        if 0 <= cell_x < COLS and 0 <= cell_y < ROWS:
            cells_data[(cell_x, cell_y)] = pd.concat([cells_data[(cell_x, cell_y)], row.to_frame().T],
                                                     ignore_index=True)
    
    # Remplir les cellules vides par interpolation
    for ix in range(COLS):
        for iy in range(ROWS):
            if cells_data[(ix, iy)].empty:
                # Créer une ligne interpolée avec les valeurs les plus proches
                interpolated_row = pd.DataFrame({
                    'x': [grid_x[iy, ix]],
                    'y': [grid_y[iy, ix]],
                    'time': [data['time'].mean()],  # Temps moyen
                    'u': [grid_vars['u'][iy, ix]],
                    'v': [grid_vars['v'][iy, ix]],
                    'w': [grid_vars['w'][iy, ix]],
                })
                cells_data[(ix, iy)] = interpolated_row
    
    return cells_data


# Diviser les données en cellules de la grille
def assign_data_to_cells(data):
    """Crée une grille de cellules et leur associe des DataFrames contenant les données."""
    cells = [[Cellule(x, y) for y in range(ROWS)] for x in range(COLS)]
    
    # Pour chaque cellule, assigner son DataFrame
    for (x, y), df in data.items():
        cells[x][y].data = df
    
    return cells


# Charger les données
file_name = 'Test_model/output/solution.txt'
data = load_data(file_name)

# Créer les cellules de la grille et les assigner aux données
cells = assign_data_to_cells(data)

# Boucle principale de Pygame
running = True
while running:
    screen.fill(BACKGROUND_COLOR)  # Remplir l'écran de blanc
    
    # Dessiner les cellules
    for row in cells:
        for cell in row:
            cell.draw(screen)
    
    pygame.display.flip()  # Mettre à jour l'affichage
    
    # Gérer les événements (fermer la fenêtre)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()