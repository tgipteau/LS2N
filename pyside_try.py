import sys
import matplotlib
import time

matplotlib.use('Qt5Agg')

from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QApplication, QSlider
from PySide6.QtCore import QTimer, Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from scipy.interpolate import griddata
import numpy as np
import Test_model.read as read


TIMESTEP = 0.2


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=10, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)


class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        
        ### Construction de la figure et intégration widget
        self.canvas = MplCanvas(self, width=10, height=8, dpi=100)
        self.setCentralWidget(self.canvas)
        
        # initialisation
        filename = 'Test_model/output/solution.txt'
        df_list, t = read.solution(filename)
        self.df_list = df_list
        self.current_time = 0
        self.max_time_index = len(df_list) - 1
   
        # extraction vmin et vmax
        self.vmax = max([df_t["u(x,y)"].max() for df_t in df_list])
        self.vmin = min([df_t["u(x,y)"].max() for df_t in df_list])
        print(f"vmax: {self.vmax}, vmin: {self.vmin}")
        
        
        ### Construction toolbar
        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        toolbar = NavigationToolbar(self.canvas, self)

        
        # Ajout du curseur pour le contrôle du temps
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.max_time_index)
        self.slider.valueChanged.connect(self.update_plot)  # Mise à jour lors du déplacement
        
        # construction layout
        layout = QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.slider)
        
        # Create a placeholder widget to hold our toolbar and canvas.
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        
        
        ### Affichage de la figure
        self.show()
        
        """
        # Setup a timer to trigger the redraw by calling update_plot.
        self.timer = QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()
        """
    
    
    def update_plot(self):

        ### Construction de la heatmap
        try :
            self.current_time = self.slider.value()
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
                                    origin="lower", cmap="YlGn_r", aspect="auto",
                                    vmin=self.vmin, vmax=self.vmax)
            self.canvas.fig.suptitle(f"Temps {self.current_time * TIMESTEP}")
            # Trigger the canvas to update and redraw.
            self.canvas.draw()
        except IndexError:
            pass
            

app = QApplication(sys.argv)
w = MainWindow()
app.exec()