"""Module chargé de la partie toolbar / UI.
Dont dépend pg_main"""

### Imports et chargement de la config yaml
import pygame_gui
import pygame
import yaml

from pygame_gui.core import ObjectID

config = yaml.load(open('pg_config.yaml', 'r'), Loader=yaml.FullLoader)

# Définition des mesures
screen_height = config['screen_height']
screen_width = config['screen_width']

view_width = config['view_width']
view_height = config['view_height']

toolbar_width = config['toolbar_width']
toolbar_height = config['toolbar_height']

tool_start_left = toolbar_width * config['toolbar_margin']
tool_width = toolbar_width * (1 - 2 * config['toolbar_margin'])
tool_height = (toolbar_height // config['toolbar_capacity'])

time_slider_height = 25

als_start = toolbar_width + view_width
als_width = screen_width - als_start
als_height = screen_height


def make_tools(manager):
   
    ### Atelier création de rectangles
    rects = [pygame.Rect(tool_start_left, i * tool_height + 5,
                         tool_width, tool_height) for i in range(config['toolbar_capacity'])]
    slider_rect = pygame.Rect((toolbar_width, screen_height-time_slider_height),
                                  (screen_width-toolbar_width, time_slider_height))
    play_button_rect = pygame.Rect((3, screen_height-time_slider_height), (tool_width, time_slider_height))
    
    
    ### Définition des containers
    toolbar_container = pygame_gui.elements.UIPanel(
        relative_rect=pygame.Rect((0, 0), (toolbar_width, toolbar_height)),  # Position et taille
        manager=manager
    )
    
    
    ### Définition des outils
    slider = pygame_gui.elements.UIHorizontalSlider(
        relative_rect=slider_rect,
        start_value=0,  # Valeur de départ
        value_range=(0, config['max_time']-1),  # Plage de valeurs (min, max)
        manager=manager,
    )
    
    play_button = pygame_gui.elements.UIButton(
        relative_rect=play_button_rect,
        text="Play",
        manager=manager,
    )
    
    young_button = pygame_gui.elements.UIButton(
        relative_rect=rects[0],
        text="young",
        object_id=ObjectID(object_id="#young_button"),
        manager=manager,
        container=toolbar_container,
    )
    
    old_button = pygame_gui.elements.UIButton(
        relative_rect=rects[1],
        text="old",
        object_id=ObjectID(object_id="#old_button"),
        manager=manager,
        container=toolbar_container,
    )
    
    seed_button = pygame_gui.elements.UIButton(
        relative_rect=rects[2],
        text="seed",
        object_id=ObjectID(object_id="#seed_button"),
        manager=manager,
        container=toolbar_container,
    )
    
    clear_sel_button = pygame_gui.elements.UIButton(
        relative_rect=rects[3],
        text="clear selection",
        manager=manager,
        container=toolbar_container,
    )
    
    nodes_button = pygame_gui.elements.UIButton(
        relative_rect=rects[4],
        text="Nodes",
        manager=manager,
        container=toolbar_container,
    )
    
    ## partie analyse
    
    als_rect = pygame.Rect((als_start + 10, 10), (als_width - 10, als_height - 10))
    
    
    
    return {
        "slider": slider,
        "slider_rect": slider_rect,
        "young_button": young_button,
        "old_button": old_button,
        "seed_button": seed_button,
        "play_button": play_button,
        "clear_sel_button": clear_sel_button,
        "nodes_button": nodes_button,
        "als_rect": als_rect,
    }


    
    