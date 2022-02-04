from videolabeler import utils as vl
from videolabeler import qtwindow as q
from videolabeler.qtwindow import App as App
import cv2
import sys # for exiting
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow

location = '/home/ssaradhi/Desktop/mirrorvid/testvideo.avi'


# q.window()
# vl.mirror(location)

# App(Q)

# app = QApplication(sys.argv)
# ex = App()
# ex.show()
# sys.exit(app.exec_())

app = QApplication(sys.argv)
player = App()
player.resize(640, 480)
player.show()
sys.exit(app.exec_())