import sys
import yaml
import socket
import struct
import threading
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QGroupBox, QSpinBox, QDoubleSpinBox,
    QMessageBox, QTextEdit, QTabWidget
)
from PyQt5.QtCore import QProcess
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class ConfigEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Éditeur de configuration MNIST")
        self.setGeometry(100, 100, 600, 500)
        self.config = {}
        self.process = QProcess()
        self.process.readyReadStandardOutput.connect(self.update_logs)
        self.process.readyReadStandardError.connect(self.update_logs)
        self.loss_values = []
        self.init_ui()
        self.start_socket_server()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        tabs = QTabWidget()
        tabs.addTab(self.create_config_tab(), "Configuration")
        tabs.addTab(self.create_training_tab(), "Entraînement")
        layout.addWidget(tabs)

    def create_config_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        self.training_group = self.create_training_group()
        self.model_group = self.create_model_group()
        self.data_group = self.create_data_group()

        layout.addWidget(self.training_group)
        layout.addWidget(self.model_group)
        layout.addWidget(self.data_group)

        self.load_button = QPushButton("Charger la configuration")
        self.save_button = QPushButton("Sauvegarder la configuration")
        self.load_button.clicked.connect(self.load_config)
        self.save_button.clicked.connect(self.save_config)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.save_button)
        layout.addLayout(button_layout)

        tab.setLayout(layout)
        return tab

    # def create_training_group(self):
    #     group = QGroupBox("Paramètres d'entraînement")
    #     layout = QVBoxLayout()

    #     self.batch_size_input = QSpinBox()
    #     self.batch_size_input.setRange(1, 256)
    #     self.num_epochs_input = QSpinBox()
    #     self.num_epochs_input.setRange(1, 100)
    #     self.learning_rate_input = QDoubleSpinBox()
    #     self.learning_rate_input.setRange(0.0001, 1.0)
    #     self.learning_rate_input.setSingleStep(0.0001)

    #     layout.addWidget(QLabel("Taille du batch :"))
    #     layout.addWidget(self.batch_size_input)
    #     layout.addWidget(QLabel("Nombre d'époques :"))
    #     layout.addWidget(self.num_epochs_input)
    #     layout.addWidget(QLabel("Taux d'apprentissage :"))
    #     layout.addWidget(self.learning_rate_input)

    #     group.setLayout(layout)
    #     return group
    def create_training_group(self):
        group = QGroupBox("Paramètres d'entraînement")
        layout = QVBoxLayout()

        self.batch_size_input = QSpinBox()
        self.batch_size_input.setRange(1, 256)
        self.num_epochs_input = QSpinBox()
        self.num_epochs_input.setRange(1, 100)
        self.learning_rate_input = QDoubleSpinBox()
        self.learning_rate_input.setRange(0.0001, 1.0)
        self.learning_rate_input.setSingleStep(0.0001)
        self.learning_rate_input.setDecimals(6)  # Ajoute cette ligne pour afficher plus de décimales

        layout.addWidget(QLabel("Taille du batch :"))
        layout.addWidget(self.batch_size_input)
        layout.addWidget(QLabel("Nombre d'époques :"))
        layout.addWidget(self.num_epochs_input)
        layout.addWidget(QLabel("Taux d'apprentissage :"))
        layout.addWidget(self.learning_rate_input)

        group.setLayout(layout)
        return group

    def create_model_group(self):
        group = QGroupBox("Paramètres du modèle")
        layout = QVBoxLayout()

        self.conv1_out_channels_input = QSpinBox()
        self.conv1_out_channels_input.setRange(1, 128)
        self.conv2_out_channels_input = QSpinBox()
        self.conv2_out_channels_input.setRange(1, 128)
        self.fc1_out_features_input = QSpinBox()
        self.fc1_out_features_input.setRange(1, 512)
        self.fc2_out_features_input = QSpinBox()
        self.fc2_out_features_input.setRange(1, 100)

        layout.addWidget(QLabel("Canaux de sortie conv1 :"))
        layout.addWidget(self.conv1_out_channels_input)
        layout.addWidget(QLabel("Canaux de sortie conv2 :"))
        layout.addWidget(self.conv2_out_channels_input)
        layout.addWidget(QLabel("Features fc1 :"))
        layout.addWidget(self.fc1_out_features_input)
        layout.addWidget(QLabel("Features fc2 :"))
        layout.addWidget(self.fc2_out_features_input)

        group.setLayout(layout)
        return group

    def create_data_group(self):
        group = QGroupBox("Paramètres des données")
        layout = QVBoxLayout()

        self.root_dir_input = QLineEdit()

        layout.addWidget(QLabel("Dossier racine des données :"))
        layout.addWidget(self.root_dir_input)

        group.setLayout(layout)
        return group

    def create_training_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        self.train_button = QPushButton("Lancer l'entraînement")
        self.train_button.clicked.connect(self.start_training)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)

        # Ajouter un graphique pour la perte
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Évolution de la perte")
        self.ax.set_xlabel("Étape")
        self.ax.set_ylabel("Perte")

        layout.addWidget(self.train_button)
        layout.addWidget(QLabel("Sortie de l'entraînement :"))
        layout.addWidget(self.log_area)
        layout.addWidget(QLabel("Graphique de la perte :"))
        layout.addWidget(self.canvas)

        tab.setLayout(layout)
        return tab

    def load_config(self):
        try:
            with open('config.yaml', 'r') as f:
                self.config = yaml.safe_load(f)
            self.batch_size_input.setValue(self.config['training']['batch_size'])
            self.num_epochs_input.setValue(self.config['training']['num_epochs'])
            #self.learning_rate_input.setValue(self.config['training']['learning_rate'])
            self.learning_rate_input.setValue(float(self.config['training']['learning_rate']))
            self.conv1_out_channels_input.setValue(self.config['model']['conv1_out_channels'])
            self.conv2_out_channels_input.setValue(self.config['model']['conv2_out_channels'])
            self.fc1_out_features_input.setValue(self.config['model']['fc1_out_features'])
            self.fc2_out_features_input.setValue(self.config['model']['fc2_out_features'])
            self.root_dir_input.setText(self.config['data']['root_dir'])
            QMessageBox.information(self, "Succès", "Configuration chargée avec succès !")
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Impossible de charger la configuration : {e}")

    def save_config(self):
        try:
            self.config = {
                'training': {
                    'batch_size': self.batch_size_input.value(),
                    'num_epochs': self.num_epochs_input.value(),
                    #'learning_rate': self.learning_rate_input.value(),
                     'learning_rate': float(self.learning_rate_input.value()),
                },
                'model': {
                    'conv1_out_channels': self.conv1_out_channels_input.value(),
                    'conv2_out_channels': self.conv2_out_channels_input.value(),
                    'fc1_out_features': self.fc1_out_features_input.value(),
                    'fc2_out_features': self.fc2_out_features_input.value(),
                },
                'data': {
                    'root_dir': self.root_dir_input.text(),
                }
            }
            with open('config.yaml', 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            QMessageBox.information(self, "Succès", "Configuration sauvegardée avec succès !")
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Impossible de sauvegarder la configuration : {e}")

    def start_training(self):
        try:
            self.log_area.clear()
            self.loss_values = []
            self.ax.clear()
            self.ax.set_title("Évolution de la perte")
            self.ax.set_xlabel("Étape")
            self.ax.set_ylabel("Perte")
            self.canvas.draw()
            self.process.start("python", ["train.py"])
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Impossible de lancer l'entraînement : {e}")

    def update_logs(self):
        output = self.process.readAllStandardOutput().data().decode()
        error = self.process.readAllStandardError().data().decode()
        if output:
            self.log_area.append(output)
        if error:
            self.log_area.append(f"ERREUR: {error}")

    def start_socket_server(self):
        HOST = '127.0.0.1'
        PORT = 65432

        def socket_listener():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((HOST, PORT))
                s.listen()
                while True:
                    conn, addr = s.accept()
                    with conn:
                        while True:
                            data = conn.recv(4)
                            if not data:
                                break
                            loss_value = struct.unpack('!f', data)[0]
                            self.loss_values.append(loss_value)
                            self.update_plot()

        threading.Thread(target=socket_listener, daemon=True).start()

    def update_plot(self):
        self.ax.clear()
        self.ax.plot(self.loss_values, 'r-')
        self.ax.set_title("Évolution de la perte")
        self.ax.set_xlabel("Étape")
        self.ax.set_ylabel("Perte")
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ConfigEditor()
    window.show()
    sys.exit(app.exec_())
