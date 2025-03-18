import sys
import numpy as np
# attention : use numpy 1.23.0 for full compatibility

import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QSlider
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


import Test_model.read as read


class HeatmapApp(QMainWindow):
    def __init__(self, dfs):
        super().__init__()

        self.dfs = dfs  # Liste de DataFrames, chaque df correspond à un pas de temps
        self.setWindowTitle("Carte de Chaleur - Diffusion Réaction")
        self.setGeometry(100, 100, 800, 600)

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
        self.slider.setMaximum(len(self.dfs) - 1)  # Nombre de DataFrames dans la liste
        self.slider.valueChanged.connect(self.update_heatmap)
        layout.addWidget(self.slider)

        # Initialisation de la heatmap
        self.update_heatmap()

    def update_heatmap(self):
        # Récupération du DataFrame correspondant au temps sélectionné
        time_index = self.slider.value()
        df_t = self.dfs[time_index]

        # Extraction des données x, y, valeur
        x, y, z = df_t['x'], df_t['y'], df_t['u']

        # Création d'une grille régulière pour l'interpolation
        grid_x, grid_y = np.meshgrid(
            np.linspace(x.min(), x.max(), 100),
            np.linspace(y.min(), y.max(), 100)
        )
        grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

        # Affichage de la carte de chaleur
        self.ax.clear()
        heatmap = self.ax.imshow(grid_z, extent=[x.min(), x.max(), y.min(), y.max()],
                                 origin="lower", cmap="viridis", aspect="auto")

        self.fig.colorbar(heatmap, ax=self.ax)
        self.ax.set_title(f"Temps = {time_index + 1}")  # Affichage du numéro du pas de temps

        # Rafraîchir l'affichage
        self.canvas.draw()

if __name__ == "__main__":


    filename = 'Test_model/output/solution.txt'
    df, t = read.solution(filename)

    print(df[0]["x"])



    app = QApplication(sys.argv)
    window = HeatmapApp(df)
    window.show()
    sys.exit(app.exec())

