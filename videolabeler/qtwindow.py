import argparse
import threading

import cv2
import numpy as np
from PyQt5.QtMultimediaWidgets import QVideoWidget
from tqdm.notebook import tqdm
from tqdm import tnrange
from skvideo.io import FFmpegWriter
import numpy as np
import pandas as pd
from queue import Queue
# from imutils.video import FileVideoStream
from imutils.video import FPS
import random
import time
from moviepy.editor import ImageSequenceClip

from threading import Thread
from timeit import default_timer as timer
from datetime import timedelta

import sys

from PyQt5.QtWidgets import QWidget, QLabel, QApplication, QMainWindow, QSlider, QVBoxLayout, QHBoxLayout, QButtonGroup, \
    QRadioButton, QGridLayout, QPushButton, QStyle, QSizePolicy, QFileDialog, QAction, QTableWidget, QShortcut
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer, QUrl, QDir, QEvent
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtGui import QImage, QPixmap, QPalette, QIcon, QKeySequence
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
        self.keyPressed = pyqtSignal(QEvent)
        self.initUI()
        self.show()

    def initUI(self):
        self._run_flag = True
        self.mousename = 'test'
        self.clip_num = 1
        data = []
        self.radioButtons = pd.DataFrame(data=data, columns = ['ID', 'Segment', 'Score 1', 'Score 2'])
        self.radioButtons.set_index('ID')
        # 'Score 2', 'Score 3', 'Score 4'])

        self.setWindowTitle("PyQt Video Player Widget Example - pythonprogramminglanguage.com")
        self.label = QLabel()

        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.frames = 0

        videoWidget = QVideoWidget()

        self.playButton = QPushButton()
        self.playButton.setEnabled(True)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
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

        # Buttons
        Lo_one = QRadioButton("Lo 0")
        Lo_two = QRadioButton("Lo 1")
        Lo_three = QRadioButton("Lo 2")
        Lo_four = QRadioButton("Lo 3")
        Lo_five = QRadioButton("Lo 4")
        # cs1.move(130, 20)

        Li_one = QRadioButton("Li 0")
        Li_two = QRadioButton("Li 1")
        Li_three = QRadioButton("Li 2")
        Li_four = QRadioButton("Li 3")
        Li_five = QRadioButton("Li 4")
        # cs2.move(130, 40)

        Ax_one = QRadioButton("Ax 0")
        Ax_two = QRadioButton("Ax 1")
        Ax_three = QRadioButton("Ax 2")
        Ax_four = QRadioButton("Ax 3")
        Ax_five = QRadioButton("Ax 4")
        # cs3.move(130, 60)

        Or_one = QRadioButton("Or 0")
        Or_two = QRadioButton("Or 1")
        Or_three = QRadioButton("Or 2")
        Or_four = QRadioButton("Or 3")
        Or_five = QRadioButton("Or 4")
        # cs4.move(130, 80)

        one = QHBoxLayout()
        one.addWidget(Lo_one)
        one.addWidget(Lo_two)
        one.addWidget(Lo_three)
        one.addWidget(Lo_four)
        one.addWidget(Lo_five)

        two = QHBoxLayout()
        two.addWidget(Li_one)
        two.addWidget(Li_two)
        two.addWidget(Li_three)
        two.addWidget(Li_four)
        two.addWidget(Li_five)

        three = QHBoxLayout()
        three.addWidget(Ax_one)
        three.addWidget(Ax_two)
        three.addWidget(Ax_three)
        three.addWidget(Ax_four)
        three.addWidget(Ax_five)

        four = QHBoxLayout()
        four.addWidget(Or_one)
        four.addWidget(Or_two)
        four.addWidget(Or_three)
        four.addWidget(Or_four)
        four.addWidget(Or_five)

        score1 = QButtonGroup(one)
        score1.addButton(Lo_one, 0)
        score1.addButton(Lo_two, 1)
        score1.addButton(Lo_three, 2)
        score1.addButton(Lo_four, 3)
        score1.addButton(Lo_five, 4)

        score2 = QButtonGroup(two)
        score2.addButton(Li_one, 0)
        score2.addButton(Li_two, 1)
        score2.addButton(Li_three, 2)
        score2.addButton(Li_four, 3)
        score2.addButton(Li_five, 4)

        score3 = QButtonGroup(three)
        score3.addButton(Ax_one, 0)
        score3.addButton(Ax_two, 1)
        score3.addButton(Ax_three, 2)
        score3.addButton(Ax_four, 3)
        score3.addButton(Ax_five, 4)

        score4 = QButtonGroup(four)
        score4.addButton(Or_one, 0)
        score4.addButton(Or_two, 1)
        score4.addButton(Or_three, 2)
        score4.addButton(Or_four, 3)
        score4.addButton(Or_five, 4)



        def slot1(object):
            print("Key was pressed, id is:", score1.id(object))
            id = score1.id(object)
            # self.radioButtons = self.radioButtons.append({'ID': 1, 'Score 1': id}, ignore_index=False)
            self.radioButtons.loc[self.clip_num, 'Score 1'] = id
            print(self.radioButtons)

        def slot2(object):
            id = score2.id(object)
            # self.radioButtons = self.radioButtons.append({'ID': 1, 'Score 2': id}, ignore_index=False)
            self.radioButtons.loc[self.clip_num, 'Score 2'] = id

            print(self.radioButtons)
        def slot3(object):
            id = score3.id(object)
            self.radioButtons.loc[self.clip_num, 'Score 3'] = id
            print(self.radioButtons)

        def slot4(object):
            id = score4.id(object)
            self.radioButtons.loc[self.clip_num, 'Score 4'] = id
            print(self.radioButtons)

        print(self.radioButtons)
        score1.buttonClicked.connect(slot1)
        score2.buttonClicked.connect(slot2)
        score3.buttonClicked.connect(slot3)
        score4.buttonClicked.connect(slot4)
        layout.addLayout(one)
        layout.addLayout(two)
        layout.addLayout(three)
        layout.addLayout(four)





    # FRAME BY FRAME
    #     th = Thread3(self)
    #     th.changePixmap.connect(self.setImage)
    #     th.frames.connect(self.setframes)
    #     th.start()
    #     layout.addWidget(self.label)

        self.rotator = 0
        self.clips = self.startthreads()
        self.clips[0].clip_num_flag == True


        layout.addWidget(self.label)



        # self.keyPressed.connect(self.toggle())

        self.mediaPlayer.setVideoOutput(videoWidget)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handleError)
        wid.setLayout(layout)
        for thread in threading.enumerate():
            print(thread.name)


    def keyPressEvent(self, event):

        if event.key() == Qt.Key_E:
            self.rotator += 1
            if self.rotator == 10:
                self.rotator = 0
            self.clips[self.rotator-1].clip_num_flag = False
            self.clips[self.rotator].clip_num_flag = True

        ## new
        if event.key() == Qt.Key_R:
            self.rotator -= 1
            if self.rotator == -1:
                self.rotator = 9
            self.clips[self.rotator+1].clip_num_flag = False
            self.clips[self.rotator].clip_num_flag = True


    def startthreads(self):
        allclips = []
        for clips in range(10):
            th = Thread4(self)
            th.thread_ID = clips
            th.changePixmap.connect(self.setImage)
            th.frames.connect(self.setframes)
            th.start()
            allclips.append(th)

        return allclips



    @pyqtSlot(QImage)
    def setImage(self, image):
        if self._run_flag == True:
            self.label.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(object)
    def setframes(self, frames):
        framenum = frames[0]
        self.clip_num = frames[1]
        self.radioButtons.loc[self.clip_num, 'Segment' ] = framenum
        self.radioButtons.loc[self.clip_num, 'ID'] = self.mousename


    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False

    def on_key(self, event):
        sys.exit(app.exec_())


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
        if self._run_flag == False:
            self.playButton.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPause))
            self.mediaPlayer.play()
            self._run_flag = True
            print('play')
        else:
            self.playButton.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPlay))
            self.mediaPlayer.pause()
            self._run_flag = False
            print('pause')


    def positionChanged(self, position):
        self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    def handleError(self):
        self.playButton.setEnabled(False)
        self.errorLabel.setText("Error: " + self.mediaPlayer.errorString())


class Thread3(QThread):
    changePixmap = pyqtSignal(QImage)
    location = '/home/ssaradhi/Desktop/mirrorvid/avitest.avi'
    frame_counter = 0
    num_frames = 0
    frames = pyqtSignal(object)
    clip_num = 0
    All_frames = []*10

    def getframes(self, frames):
        self.frames.emit([frames, self.clip_num])


    def run(self):


        location = '/home/ssaradhi/Desktop/mirrorvid/avitest.avi'
        rig = 1
        prev = 0
        frame_rate = 40
        cap = cv2.VideoCapture(location)
        frames = vl.LoadVideoFrames(location, num_frames=100)

        num_frames = len(frames)

        # initialize frame_counter, set PlayVideo boolean to True, and start displaying video
        # for labeling
        playVideo = True
        lastframe = 700
        lastframe = 24000
        frame_start = random.randint(0,lastframe-100)
        fps = 20
        seconds = 60
        frame_end = frame_start + (fps*seconds)
        frame_counter = frame_start



        # create display window
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Video',frame_width,frame_height)

        # show previous labels if they exist
        interp_mode = False
        tent_label_ind = None
        start = time.time()

        '''
        Play & Label Video
        '''
        print("[INFO] starting video file thread...")
        start = timer()
        begin = random.randint(0,6000)

        # self.frames = begin
        self.clip_num += 1
        self.getframes(begin)

        fvs = FileVideoStream(location, begin, queue_size=128).start()
        # time.sleep(1.0)
        frames = []
        # start the FPS timer
        print('starting at ', begin)
        fps = FPS().start()
        counter = 0
        # loop over frames from the video file stream
        while (fvs.more() and counter < 1000):
            # grab the frame from the threaded video file stream, resize
            # it, and convert it to grayscale (while still retaining 3
            # channels)
            prev = 0
            time_elapsed = time.time() - prev
            fram = fvs.read()

            # this if statement is to update the images for the desired frame rate

            if rig == 1:
                frame = vl.mirror(fram)
                frames.append(frame)
                StackedImages = vl.mirror(frame)
                h, w, ch = frame.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(StackedImages.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(1000, 400, Qt.IgnoreAspectRatio)
                self.changePixmap.emit(p)
                counter += 1
                cv2.waitKey(1)
                fps.update()
            elif rig == 4:
                frame = vl.mirror2(fram)
                frames.append(frame)
                StackedImages = vl.mirror2(frame)
                h, w, ch = frame.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(StackedImages.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(1000, 400, Qt.IgnoreAspectRatio)
                self.changePixmap.emit(p)
                counter += 1
                cv2.waitKey(1)
                fps.update()
            # display the size of the queue on the frame
            # print( "Queue Size: {}".format(fvs.Q.qsize()))

        end = timer()
        print(timedelta(seconds=end - start))
        test = list(frames)

        key = cv2.waitKey(0)

        if key == ord(','):  # if `<` then go back
            frame_counter -= 1
            frame_counter = vl.setFrameCounter(frame_counter, num_frames)
            cv2.setTrackbarPos("frame", "Video", frame_counter)

        elif key == ord('.'):  # if `>` then advance
            frame_counter += 1
            frame_counter = vl.setFrameCounter(frame_counter, num_frames)
            cv2.setTrackbarPos("frame", "Video", frame_counter)
            self.run()
        elif key == ord('x'):
            print('System closing')
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            return
        clip = ImageSequenceClip(test, fps=20)
        # clip.write_videofile("movie.mp4",fps=80)#
        clip.ipython_display()

        # do a bit of cleanup
        cv2.destroyAllWindows()
        fvs.stop()

class Thread4(QThread):
    changePixmap = pyqtSignal(QImage)
    thread_ID = None
    location = '/home/ssaradhi/Desktop/mirrorvid/avitest.avi'
    frame_counter = 0
    num_frames = 0
    frames = pyqtSignal(object)
    clip_num_flag = False
    clip_num = 0
    def getframes(self, frames):
        self.frames.emit([frames, self.clip_num])


    def run(self):

        location = '/home/ssaradhi/Desktop/mirrorvid/avitest.avi'
        rig = 1
        frames = vl.LoadVideoFrames(location, num_frames=100)
        num_frames = len(frames)
        lastframe = 24000
        frame_start = random.randint(0,lastframe-100)
        fps = 20
        seconds = 60
        frame_end = frame_start + (fps*seconds)
        frame_counter = frame_start
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        print("[INFO] starting video file thread...")
        start = timer()
        begin = random.randint(0,6000)
        # self.frames = begin
        self.clip_num += 1
        self.getframes(begin)
        fvs = FileVideoStream(location, begin, queue_size=128).start()
        frames = []
        # start the FPS timer
        print('starting at ', begin)
        fps = FPS().start()
        counter = 0
        while (fvs.more() and counter < 1000):
            prev = 0
            fram = fvs.read()
            StackedImages = []
            frame = []
            if rig == 1:
                frame = vl.mirror(fram)
                frames.append(frame)
                StackedImages = vl.mirror(frame)
                # h, w, ch = frame.shape
                # bytesPerLine = ch * w
                # convertToQtFormat = QImage(StackedImages.data, w, h, bytesPerLine, QImage.Format_RGB888)
                # p = convertToQtFormat.scaled(1000, 400, Qt.IgnoreAspectRatio)
                # self.changePixmap.emit(p)
                # counter += 1
                # cv2.waitKey(1)
                # fps.update()
            elif rig == 4:
                frame = vl.mirror2(fram)
                frames.append(frame)
                StackedImages = vl.mirror2(frame)
                h, w, ch = frame.shape


            if self.clip_num_flag == True:
                h, w, ch = frame.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(StackedImages.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(1000, 400, Qt.IgnoreAspectRatio)
                self.changePixmap.emit(p)
                # print( "Queue Size: {}".format(fvs.Q.qsize()))
                counter += 1
                cv2.waitKey(1)
                fps.update()

        end = timer()
        print(timedelta(seconds=end - start))
        test = list(frames)
        key = cv2.waitKey(0)

        # clip = ImageSequenceClip(test, fps=20)
        # # clip.write_videofile("movie.mp4",fps=80)#
        # clip.ipython_display()
        # do a bit of cleanup
        cv2.destroyAllWindows()
        fvs.stop()


class FileVideoStream:
    def __init__(self, path, begin, transform=None, queue_size=128):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.begin = begin
        self.stream = (cv2.VideoCapture(path))
        self.stream.set(1, self.begin)
        self.stopped = False
        self.transform = transform

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queue_size)
        # intialize thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        # start a thread to read frames from the file video stream
        self.thread.start()
        return self

    def update(self):
        # keep looping infinitely
        #         counter = 0
        #         condition = True
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                break

            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file

                #                 if counter > 1000:
                #                     condition = False

                (grabbed, frame) = self.stream.read()
                #                 counter += 1
                self.begin += 1

                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stopped = True

                # if there are transforms to be done, might as well
                # do them on producer thread before handing back to
                # consumer thread. ie. Usually the producer is so far
                # ahead of consumer that we have time to spare.
                #
                # Python is not parallel but the transform operations
                # are usually OpenCV native so release the GIL.
                #
                # Really just trying to avoid spinning up additional
                # native threads and overheads of additional
                # producer/consumer queues since this one was generally
                # idle grabbing frames.
                if self.transform:
                    frame = self.transform(frame)

                # add the frame to the queue
                self.Q.put(frame)
            else:
                time.sleep(0.1)  # Rest for 10ms, we have a full queue

        self.stream.release()

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    # Insufficient to have consumer use while(more()) which does
    # not take into account if the producer has reached end of
    # file stream.
    def running(self):
        return self.more() or not self.stopped

    def more(self):
        # return True if there are still frames in the queue. If stream is not stopped, try to wait a moment
        tries = 0
        while self.Q.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1

        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        # wait until stream resources are released (producer thread might be still grabbing frame)
        self.thread.join()

