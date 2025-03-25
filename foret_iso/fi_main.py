import pygame
import itertools

# --- CONFIGURATION ---
WIDTH, HEIGHT = 800, 600  # Taille de la fenêtre
TILE_WIDTH, TILE_HEIGHT = 64, 32  # Taille d'une tuile (base des triangles)
GRID_WIDTH, GRID_HEIGHT = 10, 10  # Taille de la grille en triangles
ORIGIN_X, ORIGIN_Y = WIDTH // 2, 100  # Position de la grille

# --- INITIALISATION ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()


def cartesian_to_isometric(car_x, car_y):
    """Convertit des coordonnées cartésiennes en isométriques"""
    iso_x = (car_x - car_y) * (TILE_WIDTH // 2) + ORIGIN_X
    iso_y = (car_x + car_y) * (TILE_HEIGHT // 2) + ORIGIN_Y
    return iso_x, iso_y


def generate_triangle_grid():
    """Crée une grille triangulaire en cartésien"""
    triangles = []
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            # Convertir en coordonnées isométriques
            iso_x, iso_y = cartesian_to_isometric(x, y)
            
            # Triangles pointant vers le haut et vers le bas
            top_triangle = [
                (iso_x, iso_y - TILE_HEIGHT // 2),  # Haut
                (iso_x + TILE_WIDTH // 2, iso_y + TILE_HEIGHT // 2),  # Bas droite
                (iso_x - TILE_WIDTH // 2, iso_y + TILE_HEIGHT // 2)  # Bas gauche
            ]
            
            bottom_triangle = [
                (iso_x, iso_y + TILE_HEIGHT // 2),  # Bas
                (iso_x + TILE_WIDTH // 2, iso_y - TILE_HEIGHT // 2),  # Haut droite
                (iso_x - TILE_WIDTH // 2, iso_y - TILE_HEIGHT // 2)  # Haut gauche
            ]
            
            triangles.append(top_triangle)
            triangles.append(bottom_triangle)
    
    return triangles


triangles = generate_triangle_grid()

running = True
while running:
    screen.fill((30, 30, 30))  # Fond sombre
    
    # Dessiner les triangles
    for triangle in triangles:
        pygame.draw.polygon(screen, (100, 100, 100), triangle, 0)  # Remplissage
        pygame.draw.polygon(screen, (255, 255, 255), triangle, 1)  # Bordure blanche
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()