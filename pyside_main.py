import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QSlider
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

import Test_model.read as read  # Assure-toi que ce module fonctionne correctement

class HeatmapApp(QMainWindow):
    def __init__(self, dfs):
        super().__init__()

        self.dfs = dfs  # Liste de DataFrames, un par pas de temps
        self.setWindowTitle("Carte de Chaleur - Diffusion Réaction")
        self.setGeometry(100, 100, 800, 600)
        self.colorbar = None  # Stocker la colorbar pour la supprimer après

        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layout principal
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Création de la figure Matplotlib
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)  # Désactivation raccourcis clavier
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Curseur pour contrôler le temps
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.dfs) - 1)  # Nombre de DataFrames
        self.slider.valueChanged.connect(self.update_heatmap)
        layout.addWidget(self.slider)

        # Initialisation de la heatmap
        self.update_heatmap(0)

    def update_heatmap(self, time_index=None):
        """ Met à jour la heatmap lorsqu'on change le pas de temps """

        print("IN UPDATE HEATMAP")
        # Récupération de l'index du slider si non spécifié
        if time_index is None:
            print("time_index is None")
            time_index = self.slider.value()
        
        print("time is now : ", time_index)
        
        # Récupération du DataFrame correspondant au temps sélectionné
        df_t = self.dfs[time_index]

        # Extraction des données x, y, valeur
        x, y, z = df_t['x'], df_t['y'], df_t['u(x,y)']

        # Création d'une grille régulière pour l'interpolation
        grid_x, grid_y = np.meshgrid(
            np.linspace(x.min(), x.max(), 100),
            np.linspace(y.min(), y.max(), 100)
        )
        grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

        # Effacer l'ancienne heatmap
        self.ax.clear()

        # Supprimer l'ancienne colorbar si elle existe et est valide
        if self.colorbar:
            try:
                self.colorbar.ax.remove()
            except AttributeError:
                pass  # Évite l'erreur si `ax` est déjà supprimé
            self.colorbar = None  # Remet à zéro pour éviter une suppression multiple

        # Affichage de la nouvelle carte de chaleur
        heatmap = self.ax.imshow(grid_z, extent=[x.min(), x.max(), y.min(), y.max()],
                                 origin="lower", cmap="viridis", aspect="auto",
                                 vmin=0, vmax=5)

        # Ajouter la nouvelle colorbar
        self.colorbar = self.fig.colorbar(heatmap, ax=self.ax)

        self.ax.set_title(f"Temps = {time_index + 1}")  # Affichage du numéro du pas de temps

        # Rafraîchir l'affichage
        self.canvas.draw()

if __name__ == "__main__":
    # Chargement des données
    filename = 'Test_model/output/old_sol.txt'
    df_list, t = read.solution(filename)  # df_list est une liste de DataFrames

    print(df_list[0].columns)  # Vérification des colonnes

    app = QApplication(sys.argv)
    window = HeatmapApp(df_list)
    window.show()
    sys.exit(app.exec())