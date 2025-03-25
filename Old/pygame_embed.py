import sys
import pygame
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QSlider
from PySide6.QtCore import Qt, QTimer


class PygameWidget(QWidget):
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window
        
        self.setFixedSize(500, 500)  # Taille du widget Pygame

        # Initialisation de Pygame
        pygame.init()
        self.surface = pygame.Surface((500, 500))
        self.running = True
        self.color = (0, 100, 255)  # Couleur initiale

        # Création d'un timer pour rafraîchir le rendu Pygame
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_pygame)
        self.timer.start(16)  # 60 FPS

    def update_pygame(self):
        if not self.running:
            return

        # Dessiner sur la surface Pygame
        self.surface.fill(self.color)  # Remplit l'écran avec la couleur actuelle
        pygame.draw.circle(self.surface, (255, 255, 255), (250, 250), 50)  # Cercle blanc

        # Récupérer l'image de Pygame et l'afficher dans PySide
        img = pygame.image.tobytes(self.surface, "RGB")
        self.setPixmap(img)

    def setPixmap(self, img_data):
        from PySide6.QtGui import QImage, QPixmap
        qimg = QImage(img_data, 500, 500, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        print("appel upd")
        self.main_window.update_pygame_display(pixmap)  # Met à jour l'affichage PySide

    def change_color(self, value):
        """ Change la couleur de fond en fonction du slider """
        self.color = (value, 100, 255)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PySide6 + Pygame Integration")

        # Layout principal
        central_widget = QWidget()
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Ajout du widget Pygame
        self.pygame_widget = PygameWidget(self, main_window=self)
        layout.addWidget(self.pygame_widget)

        # Ajout du bouton
        self.button = QPushButton("Changer Couleur")
        self.button.clicked.connect(lambda: self.pygame_widget.change_color(self.slider.value()))
        layout.addWidget(self.button)

        # Ajout d'un slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(255)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(lambda:
                                         self.pygame_widget.change_color(value=self.slider.value()))
        layout.addWidget(self.slider)

    def update_pygame_display(self, pixmap):
        """ Met à jour l'affichage Pygame dans PySide """
        print("in upd")
        self.pygame_widget.setFixedSize(500, 500)
        self.pygame_widget.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())