import cv2 as cv
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import pickle

from PyQt5 import QtCore, QtGui, uic
print('[INFO] Successful import of uic') #often reinstallation of PyQt5 is required

from PyQt5.QtCore import (QRect, QPoint, QCoreApplication, QThread, QThreadPool, pyqtSignal, pyqtSlot, Qt, QTimer, QDateTime, QObject, QMutex)
from PyQt5.QtGui import (QImage, QPixmap, QPainter, QPen, QTextCursor, QIntValidator)
from PyQt5.QtWidgets import (QWidget, QMainWindow, QApplication, QLabel, QPushButton, QVBoxLayout, QGridLayout, QSizePolicy, QMessageBox, QFileDialog, QSlider, QComboBox, QProgressDialog)

print('[INFO] Loaded Packages and Starting Labeling APP...')
################################# VARIABLES #####################################################
# Qt_Designer File to open:
qtCreatorFile = r"D:\a_WoundDetection\Project\previews\data\03.10\Data\Labeling.ui"
# Create a GUI window (inherited from in class App())
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)
# Folder where images are to be stored (can be changed by user)
# save_folder = "/PROJECT/SHARED_WORKSPACE/PU/PYTHON/"

#################################### APP ###################################################
# Class for the central widget of the application defining the order/start-up/callbacks etc.; Opens the imported .ui GUI and inherits from it
class App(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setMouseTracking(True)
        self.setupUi(self)
        self.initUI()

    # Determining the callbacks for certain ui-actions
    def initUI(self):
        self.pBnext.clicked.connect(self.show_next_image)
        self.pBpreviews.clicked.connect(self.show_previews_image)

        self.pBloadfile.clicked.connect(self.prepare_im_list)
        self.pBcalcaneus.clicked.connect(self.labeling_as_calcaneus)
        self.pBcephalon.clicked.connect(self.labeling_as_cephalon)
        self.pBdigits.clicked.connect(self.labeling_as_digits)
        self.pBdorsum.clicked.connect(self.labeling_as_dorsum)
        self.pBear.clicked.connect(self.labeling_as_ear)
        self.pBelbow.clicked.connect(self.labeling_as_elbow)
        self.pBknee.clicked.connect(self.labeling_as_knee)
        self.pBlumbus.clicked.connect(self.labeling_as_lumbus)
        self.pBmunus.clicked.connect(self.labeling_as_munus)
        self.pBothers.clicked.connect(self.labeling_as_others)
        self.pBplantoffoot.clicked.connect(self.labeling_as_plantoffoot)
        self.pBshoulder.clicked.connect(self.labeling_as_shoulder)
        self.pBexport.clicked.connect(self.export)
        self.pBenter.clicked.connect(self.go_to_image)

        self.tEinputpath.setPlainText(r'D:\a_WoundDetection\Project\datasets\temp')
        self.tEimnumber.setPlainText("0")


    def prepare_im_list(self):
        self.folder = self.tEinputpath.toPlainText()
        self.im = []
        self.im_path = []
        self.count = -1
        self.path_to_pkl = os.path.join(self.tEinputpath.toPlainText(), 'label.pkl')
        try:
            with open(self.path_to_pkl, 'rb') as handle:
                try:
                    self.label_table = pickle.load(handle)
                except:
                    self.label_table = dict()
        except:
            with open(self.path_to_pkl, "xb") as handle:
                try:
                    self.label_table = pickle.load(handle)
                except:
                    self.label_table = dict()
    
        # self.label_table = dict()
        for file in os.listdir(self.folder):
            self.im.append(file)
            self.im_path.append(os.path.join(self.folder, file))
        print("[INFO] List created!")

    def show_next_image(self):
        self.count += 1
        try:
            self.im = cv.imread(self.im_path[self.count])
            frame_rgb_preview = cv.resize(self.im, (280, 280)) #
            frame_rgb_preview = cv.cvtColor(frame_rgb_preview, cv.COLOR_BGR2RGB)
            bytesPerLine = 3 * frame_rgb_preview.shape[1]
            self.Q_image = QImage(frame_rgb_preview.data, frame_rgb_preview.shape[1], frame_rgb_preview.shape[0], bytesPerLine, QImage.Format_RGB888)
            self.QLdisplay.setPixmap(QPixmap.fromImage(self.Q_image))
            print(f"[INFO] Showing: {self.im_path[self.count]}")
        except:
            print("[INFO] No more image in current folder!")
    
    def show_previews_image(self):
        self.count -= 1
        self.im = cv.imread(self.im_path[self.count])
        frame_rgb_preview = cv.resize(self.im, (280, 280)) #
        frame_rgb_preview = cv.cvtColor(frame_rgb_preview, cv.COLOR_BGR2RGB)
        bytesPerLine = 3 * frame_rgb_preview.shape[1]
        self.Q_image = QImage(frame_rgb_preview.data, frame_rgb_preview.shape[1], frame_rgb_preview.shape[0], bytesPerLine, QImage.Format_RGB888)
        self.QLdisplay.setPixmap(QPixmap.fromImage(self.Q_image))
        print(f"[INFO] Showing: {self.im_path[self.count]}")

    def go_to_image(self):
        self.count = int(self.tEimnumber.toPlainText())
        self.im = cv.imread(self.im_path[self.count])
        frame_rgb_preview = cv.resize(self.im, (280, 280)) #
        frame_rgb_preview = cv.cvtColor(frame_rgb_preview, cv.COLOR_BGR2RGB)
        bytesPerLine = 3 * frame_rgb_preview.shape[1]
        self.Q_image = QImage(frame_rgb_preview.data, frame_rgb_preview.shape[1], frame_rgb_preview.shape[0], bytesPerLine, QImage.Format_RGB888)
        self.QLdisplay.setPixmap(QPixmap.fromImage(self.Q_image))
        print(f"[INFO] Showing: {self.im_path[self.count]}")


    def labeling_as_calcaneus(self):
        l = "calcaneus"
        self.label_table[self.im_path[self.count]] = l
        print(f"[INFO] The current image is labeled as: {l}")

    def labeling_as_cephalon(self):
        l = "cephalon"
        self.label_table[self.im_path[self.count]] = l
        print(f"[INFO] The current image is labeled as: {l}")

    def labeling_as_digits(self):
        l = "digits"
        self.label_table[self.im_path[self.count]] = l
        print(f"[INFO] The current image is labeled as: {l}")
    
    def labeling_as_dorsum(self):
        l = "dorsum"
        self.label_table[self.im_path[self.count]] = l
        print(f"[INFO] The current image is labeled as: {l}")

    def labeling_as_ear(self):
        l = "ear"
        self.label_table[self.im_path[self.count]] = l
        print(f"[INFO] The current image is labeled as: {l}")
    
    def labeling_as_elbow(self):
        l = "elbow"
        self.label_table[self.im_path[self.count]] = l
        print(f"[INFO] The current image is labeled as: {l}")

    def labeling_as_knee(self):
        l = "knee"
        self.label_table[self.im_path[self.count]] = l
        print(f"[INFO] The current image is labeled as: {l}")
    
    def labeling_as_lumbus(self):
        l = "lumbus"
        self.label_table[self.im_path[self.count]] = l
        print(f"[INFO] The current image is labeled as: {l}")

    def labeling_as_munus(self):
        l = "munus"
        self.label_table[self.im_path[self.count]] = l
        print(f"[INFO] The current image is labeled as: {l}")

    def labeling_as_others(self):
        l = "others"
        self.label_table[self.im_path[self.count]] = l
        print(f"[INFO] The current image is labeled as: {l}")

    def labeling_as_plantoffoot(self):
        l = "plantoffoot"
        self.label_table[self.im_path[self.count]] = l
        print(f"[INFO] The current image is labeled as: {l}")
    
    def labeling_as_shoulder(self):
        l = "shoulder"
        self.label_table[self.im_path[self.count]] = l
        print(f"[INFO] The current image is labeled as: {l}")

    def export(self):
        with open(self.path_to_pkl, 'wb') as handle:
            pickle.dump(self.label_table, handle)
        print("[INFO] Pickle file saved!")
        with open(self.path_to_pkl, 'rb') as handle2:
            t = pickle.load(handle2)
        print(t)

    def closeEvent(self, event):
        print("[INFO] Close event called")

        # Stop the timers
        # self.timer.stop()
        # self.display_timer.stop()

# Main routine for displaying the GUI
def main():
    # The following sequence is the standard code for opening a custom application's (ui, interactive) window
    app = QApplication(sys.argv)
    # Use the class App as central widget
    window = App()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()