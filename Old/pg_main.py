"""
Module chargé de faire tourner la boucle pygame. Dépendant de pg_render pour
les travaux de rendus et de pg_toolbar pour l'UI.
"""

import time

### Imports et chargement de la config yaml
import pygame
import pygame_gui
import yaml
config = yaml.load(open('pg_config.yaml', 'r'), Loader=yaml.FullLoader)

### Initialisation de Pygame
pygame.init()
screen = pygame.display.set_mode((config['screen_width'], config['screen_height']), pygame.RESIZABLE)
pygame.display.set_caption(config['caption'])

import pg_render as pgr
import pg_toolbar as pgt # ici car pygame_gui a besoin de pygame.init()

### Création toolbar / UI
manager = pygame_gui.UIManager((config['screen_width'], config['screen_height']),
                               "theme.json")
tools = pgt.make_tools(manager)


if __name__ == "__main__":
    
    ### Préparation render
    true_nodes, u_min, u_max, v_min, v_max, w_min, w_max = pgr.build_from_solution()
    print(u_min, u_max, v_min, v_max, w_min, w_max)
    
    if config['interpolate_nodes']:
        fake_nodes = pgr.generate_intermediate_nodes(true_nodes)
        for node in fake_nodes:
            node.color = (200,0,0)
        zones = pgr.triangulate_and_create_zones(true_nodes + fake_nodes)
    else:
        zones = pgr.triangulate_and_create_zones(true_nodes)
    
    for zone in zones:
        zone.handle_fires()
        zone.generate_positions(u_min, u_max,
                                v_min, v_max,
                                w_min, w_max)
        
        
    running = True
    clock = pygame.time.Clock()
    FPS = 60
    
    render_mode = config['render_mode']
    
    show_young = True
    show_old = True
    show_seed = True
    show_nodes = True
    play_mode = False
    hovered_zone = None
    selected_zones = []
    update_als = True
    
    t = 0
    max_time = config['max_time']
    
    ### Boucle principale de Pygame
    while running:
        
        if play_mode:
            FPS = 25
            t = min(t + 1, max_time-1)
            tools['slider'].set_current_value(t)
        else:
            FPS = 60
            t = int(tools['slider'].get_current_value())
        
        ### Partie dessin
        screen.fill(tuple(config['bg_color']))  # Effacer l'écran
        
        if render_mode == 'zones':
            for zone in zones: # Mettre à jour les positions
                zone.draw_content(screen, t, show_young, show_old, show_seed)
                
            if hovered_zone is not None:
                hovered_zone.draw_border(screen, (150, 0, 150))
                
            for zone in selected_zones:
                zone.draw_border(screen, (230,0,230))
                
            for node in true_nodes:
                node.draw(screen, show_nodes)
        
            if update_als:
                pygame.draw.rect(screen, (255,255,255), tools['als_rect'])
                pgr.screen_plots_to(screen, tools['als_rect'], selected_zones)
                update_als = True
                
                
        
        manager.update(clock.get_time())
        manager.draw_ui(screen)
        
        # Gérer les événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == tools['young_button']:
                    show_young = not show_young  # Toggle l'état
                elif event.ui_element == tools['old_button']:
                    show_old = not show_old  # Toggle l'état
                elif event.ui_element == tools['seed_button']:
                    show_seed = not show_seed  # Toggle l'état
                elif event.ui_element == tools['nodes_button']:
                    show_nodes = not show_nodes  # Toggle l'état
                elif event.ui_element == tools['play_button']:
                    if play_mode:
                        tools['play_button'].set_text('Play')
                    else:
                        tools['play_button'].set_text('Pause')
                    play_mode = not play_mode  # Toggle l'état
                elif event.ui_element == tools['clear_sel_button']:
                    selected_zones.clear()
                    
                    
                    
            if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                if event.ui_element == tools['slider']:
                    play_mode = False
            
            if event.type == pygame.MOUSEMOTION:
                mouse_x, mouse_y = event.pos
                hovered_zone = None  # Réinitialisation
                
                # Vérifier si la souris est dans une des zones
                for zone in zones:
                    if zone.contains(mouse_x, mouse_y):
                        hovered_zone = zone
                        break
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                
                # Vérifier si la souris est dans une des zones
                for zone in zones:
                    if zone.contains(mouse_x, mouse_y):
                        clicked_zone = zone
                        update_als = True
                        if clicked_zone in selected_zones:
                            try:
                                selected_zones.remove(clicked_zone)
                            except ValueError:
                                pass
                        else:
                            selected_zones.append(clicked_zone)
                    
            
            # Passer les événements à pygame_gui pour gérer les interactions avec l'UI
            manager.process_events(event)
            
        pygame.display.flip()  # Mettre à jour l'affichage
        clock.tick(FPS)
    
    pygame.quit()
