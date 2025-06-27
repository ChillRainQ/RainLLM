import sys

from PySide6 import QtCore
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QApplication
from qfluentwidgets import CardWidget, ExpandGroupSettingCard, BodyLabel, SpinBox

from train_gui.trainer import GuiConfig


class TrainArgsCard(ExpandGroupSettingCard):
    def __init__(self, title, args: GuiConfig, icon="", parent=None):
        super().__init__(parent=parent, title=title, icon=icon)
        self.setObjectName("trainArgsCard")
        self.args = args
        self.modelLabel1 = BodyLabel("batch_size")
        self.box = SpinBox()
        self.box.setMinimumWidth(1)
        self.box.setSingleStep(1)
        self.box.setValue(args.batch_size.value)




class TrainPage(QWidget):
    def __init__(self, trainArgs: dict, parent=None):
        super().__init__()
        self.trainArgs = trainArgs
        self.setObjectName("trainPage")
        # 主布局 纵向
        mainLayout = QVBoxLayout()
        up = QWidget()
        low = QWidget()
        mainLayout.addWidget(up)
        mainLayout.addWidget(low)
        # 上部横向布局 box1
        upLayout = QHBoxLayout()
        up.setLayout(upLayout)
        left = ExpandGroupSettingCard(title="训练参数", parent=self, icon="")
        left.addActions()
        left.maximumWidth()
        right = QWidget()
        upLayout.addWidget(left)
        upLayout.addWidget(right)
        # # 上部左半部分布局
        # upLeft = QWidget()
        # leftInUpLayout = QHBoxLayout(upLeft)
        # self.trainArgsCard = ExpandGroupSettingCard(title="训练参数", parent=self, icon="")
        # leftInUpLayout.addWidget(self.trainArgsCard)
        # # 上部右半部分布局
        # rightInUpLayout = QHBoxLayout()
        # # 下部横向布局
        # lowLayout = QHBoxLayout()
        self.setLayout(mainLayout)


def main():
    app = QApplication(sys.argv)
    trainPage = TrainPage()
    trainPage.show()
    app.exec()

if __name__ == "__main__":
    main()