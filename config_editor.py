import sys
import yaml
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QGroupBox, QSpinBox, QDoubleSpinBox, QMessageBox
)

class ConfigEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Éditeur de configuration MNIST")
        self.setGeometry(100, 100, 500, 400)
        self.config = {}
        self.init_ui()

    def init_ui(self):
        # Widget central et layout principal
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Groupes de paramètres
        self.training_group = self.create_training_group()
        self.model_group = self.create_model_group()
        self.data_group = self.create_data_group()

        layout.addWidget(self.training_group)
        layout.addWidget(self.model_group)
        layout.addWidget(self.data_group)

        # Boutons
        self.load_button = QPushButton("Charger la configuration")
        self.save_button = QPushButton("Sauvegarder la configuration")
        self.load_button.clicked.connect(self.load_config)
        self.save_button.clicked.connect(self.save_config)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.save_button)
        layout.addLayout(button_layout)

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

    def load_config(self):
        try:
            with open('config.yaml', 'r') as f:
                self.config = yaml.safe_load(f)
            self.batch_size_input.setValue(self.config['training']['batch_size'])
            self.num_epochs_input.setValue(self.config['training']['num_epochs'])
            self.learning_rate_input.setValue(self.config['training']['learning_rate'])
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
                    'learning_rate': self.learning_rate_input.value(),
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ConfigEditor()
    window.show()
    sys.exit(app.exec_())
