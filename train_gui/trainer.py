# coding:utf-8
import atexit
import ctypes
import os
import sys

import torch
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QIcon, QDesktopServices
from PySide6.QtWidgets import QApplication
from qfluentwidgets import (NavigationItemPosition, MessageBox, setTheme, Theme,
                            NavigationAvatarWidget, SplitFluentWindow, FluentTranslator, QConfig, ConfigItem,
                            OptionsConfigItem, RangeConfigItem, OptionsValidator, RangeValidator, BoolValidator,
                            qconfig, FluentIcon)
from qfluentwidgets import FluentIcon as FIF

from train_gui.view.train_page import TrainPage

RESOURCES_DIR = os.path.join(os.path.dirname(__file__), "resources")
CONFIG_FILE_NAME = "config.json"
CONFIG_FILE = os.path.join(RESOURCES_DIR, CONFIG_FILE_NAME)

ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("rain")


class GuiConfig(QConfig):
    last_trained_model: ConfigItem = ConfigItem("models", "last_model", "null")
    model_list = ["null"]
    models: OptionsConfigItem = OptionsConfigItem("models", "models", "", OptionsValidator(model_list))
    batch_size: ConfigItem = ConfigItem("train_args", "batch_size", 32)
    epochs: ConfigItem = ConfigItem("train_args", "epochs", 1)
    learning_rate: RangeConfigItem = RangeConfigItem("train_args", "learning_rate", 5e-4, RangeValidator(1e-10, 9e-1))
    device: ConfigItem = ConfigItem("train_args", "device", "cuda" if torch.cuda.is_available() else "cpu")
    dtype: ConfigItem = ConfigItem("train_args", "dtype", "bfloat16")
    dim: ConfigItem = ConfigItem("train_args", "dim", 256)
    n_layers: ConfigItem = ConfigItem("train_args", "n_layers", 16)
    max_seq_len: ConfigItem = ConfigItem("train_args", "max_seq_len", 512)
    use_moe: ConfigItem = ConfigItem("train_args", "use_moe", False, BoolValidator())
    data_path: ConfigItem = ConfigItem("train_args", "data_path", os.path.join(RESOURCES_DIR, "out"))


class Trainer(SplitFluentWindow):

    def __init__(self):
        super().__init__()

        self.train_page = TrainPage()
        self.config = GuiConfig()
        self.initWindow()


    def setup(self):
        pass
        qconfig.load(CONFIG_FILE, self.config)
        atexit.register(self.save)

    def save(self):
        self.config.save()

    def initWindow(self):
        self.resize(900, 700)
        setTheme(Theme.DARK)
        self.setWindowTitle("RainLLM Trainer")
        self.setWindowIcon(QIcon('resources/rain.png'))
        self.addSubInterface(self.train_page, FluentIcon.HOME, "训练模型")
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        desktop = self.screen().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w // 2 - self.width() // 2, h // 2 - self.height() // 2)



if __name__ == '__main__':
    app = QApplication(sys.argv)

    trainer = Trainer()
    trainer.setup()
    trainer.show()
    sys.exit(app.exec())
