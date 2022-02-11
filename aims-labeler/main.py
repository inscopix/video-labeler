from videolabeler import utils as vl
from videolabeler import qtwindow as q
from videolabeler import videostream as vs
from videolabeler.qtwindow import App as App
import cv2
import sys # for exiting
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow

location = '/home/ssaradhi/Desktop/mirrorvid/mp4test.mp4'


# vl.mirror(location)

app = QApplication(sys.argv)
player = App()
player.resize(640, 480)
player.show()
sys.exit(app.exec_())

# vs.run_video(location, 1)