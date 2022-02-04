import cv2
import numpy as np
from PyQt5.QtMultimediaWidgets import QVideoWidget
from tqdm.notebook import tqdm
from tqdm import tnrange
from skvideo.io import FFmpegWriter
import numpy as np
import pandas as pd
import os
import random
import math

from itertools import groupby
from operator import itemgetter

from datetime import datetime

# from PyQt5 import QtWidgets, QtGui
import sys

from PyQt5.QtWidgets import QWidget, QLabel, QApplication, QMainWindow, QSlider, QVBoxLayout, QHBoxLayout, QButtonGroup, \
    QRadioButton, QGridLayout, QPushButton, QStyle, QSizePolicy, QFileDialog, QAction
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer, QUrl, QDir
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtGui import QImage, QPixmap, QPalette, QIcon
from videolabeler import utils as vl

def window():
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setGeometry(800, 800, 800, 800)
    win.setWindowTitle("Horizontal")
    win.show()
    sys.exit(app.exec_())

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def isolate_angle(frame, rotation, title):
    # cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    if rotation == None:
        rotation = 0
    rotated_frame = rotate_image(frame, rotation)
    # cv2.imshow(title, rotated_frame)
    return rotated_frame

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('test')
        self.scaleupSignal = pyqtSignal()
        self.initUI()
        self.show()
        # self.label = QLabel()
        self.show()
    def initUI(self):
        self.setWindowTitle("PyQt Video Player Widget Example - pythonprogramminglanguage.com")
        self.label = QLabel()

        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        videoWidget = QVideoWidget()

        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        self.errorLabel = QLabel()
        self.errorLabel.setSizePolicy(QSizePolicy.Preferred,
                                      QSizePolicy.Maximum)

        # Create new action
        openAction = QAction(QIcon('open.png'), '&Open', self)
        openAction.setShortcut('Ctrl+O')
        openAction.setStatusTip('Open movie')
        openAction.triggered.connect(self.openFile)

        # Create exit action
        exitAction = QAction(QIcon('exit.png'), '&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.exitCall)

        # Create menu bar and add action
        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu('&File')
        # fileMenu.addAction(newAction)
        fileMenu.addAction(openAction)
        fileMenu.addAction(exitAction)

        # Create a widget for window contents
        wid = QWidget(self)
        self.setCentralWidget(wid)

        # Create layouts to place inside widget
        controlLayout = QHBoxLayout()
        controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.positionSlider)

        layout = QVBoxLayout()
        layout.addWidget(videoWidget)
        layout.addLayout(controlLayout)
        layout.addWidget(self.errorLabel)
        layout.addLayout(Buttons().layout)

        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.start()
        layout.addWidget(self.label)
        # elif user == 0:
        self.show()
       # Set widget to contain window contents
        wid.setLayout(layout)
        self.mediaPlayer.setVideoOutput(videoWidget)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handleError)

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))
        self.sp = QSlider(Qt.Horizontal)

    def openFile(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Movie",
                QDir.homePath())

        if fileName != '':
            self.mediaPlayer.setMedia(
                    QMediaContent(QUrl.fromLocalFile(fileName)))
            self.playButton.setEnabled(True)

    def exitCall(self):
        sys.exit(app.exec_())

    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))

    def positionChanged(self, position):
        self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    def handleError(self):
        self.playButton.setEnabled(False)
        self.errorLabel.setText("Error: " + self.mediaPlayer.errorString())


class Buttons(QThread):
    def __init__(self):
        super().__init__()
        self.vbox = QVBoxLayout()
        self.hbox1 = QHBoxLayout()
        self.hbox2 = QHBoxLayout()
        self.hbox3 = QHBoxLayout()
        self.layout = self.buttons()

    def buttons(self):
        # buttons
        one = QRadioButton("option 1")
        two = QRadioButton("option 2")
        three = QRadioButton("option 3")
        four = QRadioButton("option 4")
        five = QRadioButton("option 5")
        six = QRadioButton("option 6")

        # one.toggled.connect(lambda:self.())

        hbox = self.hbox1
        hbox.addStretch(1)
        hbox.addWidget(one)
        hbox.addWidget(two)

        hbox2 = self.hbox2
        hbox2.addStretch(1)
        hbox2.addWidget(three)
        hbox2.addWidget(four)

        hbox3 = self.hbox3
        hbox3.addStretch(1)
        hbox3.addWidget(five)
        hbox3.addWidget(six)

        vbox = self.vbox
        vbox.addStretch(0)
        vbox.addLayout(hbox)
        vbox.addStretch(0)
        vbox.addLayout(hbox2)
        vbox.addStretch(0)
        vbox.addLayout(hbox3)

        return vbox


class Thread(QThread):
    changePixmap = pyqtSignal(QImage)
    location = '/home/ssaradhi/Desktop/mirrorvid/testvideo.avi'
    frame_counter = 0
    num_frames = 0

    def run(self):


        location = '/home/ssaradhi/Desktop/mirrorvid/testvideo.avi'
        cap = cv2.VideoCapture(location)
        frames = vl.LoadVideoFrames(location, num_frames=100)

        frames_out = frames.copy()
        frame_height = frames_out[0].shape[0]
        frame_width = frames_out[0].shape[1]
        bordersize = 50

        num_frames = len(frames)

        # initialize frame_counter, set PlayVideo boolean to True, and start displaying video
        # for labeling
        playVideo = True
        frame_counter = 0

        # create display window
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Video',frame_width,frame_height)

        # show previous labels if they exist
        interp_mode = False
        tent_label_ind = None

        '''
        Play & Label Video
        '''
        while playVideo is True:

            cap.set(1,frame_counter)
            ret, frame = cap.read(frame_counter)

            StackedImages = vl.mirror2(frame)
            h, w, ch = StackedImages.shape
            bytesPerLine = ch * w
            convertToQtFormat = QImage(StackedImages.data, w, h, bytesPerLine, QImage.Format_RGB888)
            p = convertToQtFormat.scaled(1000, 400, Qt.IgnoreAspectRatio)
            self.changePixmap.emit(p)
            key = cv2.waitKey(0)

            '''
            Check to see if the user is using interpolation
            '''
            if key == ord('i'):

                # opening interpolate mode
                if interp_mode == False:
                    # find last label


                    # interp mode is activated
                    interp_mode = True

                # closing interpolate mode
                else:
                    '''
                    Current implementation of interpolate assumes strictly chronological labeling due to the way the last label is found. 
                    You can't interpolate backwards
                    '''
                    if frame_counter < tent_label_ind:
                        continue

                    # interp mode is turned off
                    interp_mode = False
                    tent_label_ind = None

            elif key == ord(','):  # if `<` then go back
                frame_counter -= 1
                frame_counter = vl.setFrameCounter(frame_counter, num_frames)
                cv2.setTrackbarPos("frame", "Video", frame_counter)

            elif key == ord('.'):  # if `>` then advance
                frame_counter += 1
                frame_counter = vl.setFrameCounter(frame_counter, num_frames)
                cv2.setTrackbarPos("frame", "Video", frame_counter)

            elif key == ord('x'):  # if `x` then qiuit
                playVideo = False

            elif key == ord('\b'):

                # update rectangle to show label is gone
                cv2.rectangle(frame, (0, frame_height), (300, frame_height - 50), (255, 255, 255), -1)
                cv2.putText(frame, 'no_label', (0, frame_height - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2,
                            cv2.LINE_AA)
                frames_out[frame_counter] = frame

            # wait for keypress

        # close any opencv windows
        cv2.destroyAllWindows()
        cv2.waitKey(1)