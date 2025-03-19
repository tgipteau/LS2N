import sys
import matplotlib
import time

matplotlib.use('Qt5Agg')

from PySide6.QtWidgets import (QMainWindow, QVBoxLayout, QWidget, QApplication, QSlider, QToolBar, QPushButton,
QDialog, QLabel, QLineEdit, QSpinBox, QFormLayout)
from PySide6.QtCore import QTimer, Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from scipy.interpolate import griddata
import numpy as np
import os

import read
import model

TIMESTEP = 0.2


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=10, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)



class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Paramètres")
        self.setGeometry(200, 200, 300, 150)

        # Layout
        layout = QFormLayout()

        # Champs de paramètres
        self.params = parent.params
        self.inputs = {}
        
        for key, value in self.params.items():
            input_field = QLineEdit(self)
            input_field.setText(str(value))
            self.inputs[key] = input_field
            layout.addRow(f"{key} :", input_field)

        # Bouton de validation
        self.ok_button = QPushButton("Valider")
        self.ok_button.clicked.connect(self.accept)
        layout.addWidget(self.ok_button)
        
        self.setLayout(layout)

    def get_values(self):
        """retourne les valeurs saisies, en dictionnaire"""
        return {key: float(self.inputs[key].text()) for key in self.inputs}



class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # paramètres simulation
        self.params = {
            "a": 0.006,
            "b": 0.247,
            "c": 0.01,
            "r": 0.134,
            "f": 0.017,
            "h": 0.04,
            "alpha": 0.5,
            "beta": 0.5,
            "delta": 0.268,
            "tmax": 50,
            "p": 0.1,
            "I": 0.5,
            "ui": 5.838126612843206,
            "vi": 2.481203810458363,
            "wi": 2.481203810458363
        }
        
        ### Construction de la figure et intégration widget
        self.canvas = MplCanvas(self, width=10, height=8, dpi=100)
        self.setCentralWidget(self.canvas)
        
        
        # initialisation attributs (valuation par "run_simulation")
        filename = 'output/solution.txt'
        if not os.path.exists(filename):
            print("No simulation detected. Running first simulation...")
            self.run_simulation(do_sim=True)
        else :
            # si un fichier existe, pas de simu au démarrage
            # (= init des attributs)
            print("Simulation detected : Loaded.")
            self.run_simulation(do_sim=False)
            
        
        
        """
        # initialisation
        filename = 'Test_model/output/solution.txt'
        df_list, t = read.solution(filename)
        self.df_list = df_list
        self.current_time = 0
        self.max_time_index = len(df_list) - 1
   
        # extraction vmin et vmax
        self.vmax = max([df_t["u(x,y)"].max() for df_t in df_list])
        self.vmin = min([df_t["u(x,y)"].min() for df_t in df_list])
        print(f"vmax: {self.vmax}, vmin: {self.vmin}")
        """
        
        ### Constructions Widgets
        
        # toolbar matplotlib
        plot_toolbar = NavigationToolbar(self.canvas, self)

        # slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.max_time_index)
        self.slider.valueChanged.connect(self.update_plot)  # Mise à jour lors du déplacement
        
        # toolbar appli
        self.app_toolbar = QToolBar()
        
        self.run_sim_button = QPushButton("Run new simulation.")
        self.run_sim_button.clicked.connect(lambda: self.run_simulation(do_sim=True))
        self.app_toolbar.addWidget(self.run_sim_button)
        
        # Bouton "Paramètres"
        self.settings_button = QPushButton("Paramètres")
        self.settings_button.clicked.connect(self.open_settings_dialog)
        self.app_toolbar.addWidget(self.settings_button)
        
        # intégration layout
        layout = QVBoxLayout()
        layout.addWidget(plot_toolbar)
        layout.addWidget(self.app_toolbar) # essayer aussi "self.addToolbar(app_toolbar)"
        layout.addWidget(self.canvas)
        layout.addWidget(self.slider)
        
        
        # Create a placeholder widget to hold our toolbar and canvas.
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        
        
        ### Affichage de la figure
        self.update_plot()
        self.show()
        
        """
        # Setup a timer to trigger the redraw by calling update_plot.
        self.timer = QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()
        """
    
    
    
    def open_settings_dialog(self):
        """Ouvre la fenêtre des paramètres et récupère les valeurs saisies."""
        dialog = SettingsDialog(self)
        if dialog.exec():  # Ouvre la fenêtre en mode modal
            self.params = dialog.get_values()
            
    
    
    def run_simulation(self, do_sim=True):
        """ Mise à jour des attributs relatifs à la simulation. Lance une simulation, qui supprime la précédente
        pour le moment. Si do_sim=False, c'est qu'on initialise la fenêtre avec une solution déjà existante
        (ex : au démarrage)."""
        
        self.canvas.axes.clear()
        
        if do_sim :
            model.sim_PDE(self.params)

        filename = 'output/solution.txt'
        df_list, t = read.solution(filename)
        self.df_list = df_list
        self.current_time = 0
        self.max_time_index = self.params["tmax"]/TIMESTEP
        
        # extraction vmin et vmax
        self.vmax = max([df_t["u(x,y)"].max() for df_t in df_list])
        self.vmin = min([df_t["u(x,y)"].min() for df_t in df_list])
        print(f"vmax: {self.vmax}, vmin: {self.vmin}")
        
        if do_sim :
            self.slider.setValue(0)
            self.slider.setMaximum(self.max_time_index)
            
        print("i dont wait")
        # mise à jour
        self.update_plot()
        
        
    def update_plot(self):

        ### Construction de la heatmap
        try :
            try :
                self.current_time = self.slider.value()
            except AttributeError :
                # cadre update initial, dans run_sim
                self.current_time = 0
                
            # Récupération du DataFrame correspondant au temps sélectionné
            df_t = self.df_list[int(self.current_time)]
            # Extraction des données x, y, u
            x, y, z = df_t['x'], df_t['y'], df_t['u(x,y)']
            # Création d'une grille régulière pour l'interpolation
            grid_x, grid_y = np.meshgrid(
                np.linspace(x.min(), x.max(), 100),
                np.linspace(y.min(), y.max(), 100)
            )
            grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')
            
            self.canvas.axes.cla()  # Clear the canvas.
            # Imshow de la heatmap
            self.canvas.axes.imshow(grid_z, extent=[x.min(), x.max(), y.min(), y.max()],
                                    origin="lower", cmap="YlGn", aspect="auto",
                                    vmin=self.vmin, vmax=self.vmax)
            self.canvas.fig.suptitle(f"Temps {self.current_time * TIMESTEP}")
            # Trigger the canvas to update and redraw.
            self.canvas.draw()
        except IndexError:
            pass
            

app = QApplication(sys.argv)
w = MainWindow()
app.exec()