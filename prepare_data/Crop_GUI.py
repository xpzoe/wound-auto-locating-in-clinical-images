import cv2 as cv
import matplotlib.pyplot as plt
import sys
import numpy as np
import os

from PyQt5 import QtCore, QtGui, uic
print('[INFO] Successful import of uic') #often reinstallation of PyQt5 is required

from PyQt5.QtCore import (QRect, QPoint, QCoreApplication, QThread, QThreadPool, pyqtSignal, pyqtSlot, Qt, QTimer, QDateTime, QObject, QMutex)
from PyQt5.QtGui import (QImage, QPixmap, QPainter, QPen, QTextCursor, QIntValidator)
from PyQt5.QtWidgets import (QWidget, QMainWindow, QApplication, QLabel, QPushButton, QVBoxLayout, QGridLayout, QSizePolicy, QMessageBox, QFileDialog, QSlider, QComboBox, QProgressDialog)

print('[INFO] Loaded Packages and Starting Cropper APP...')
################################# VARIABLES #####################################################
# Qt_Designer File to open:
qtCreatorFile = "/PROJECT/SHARED_WORKSPACE/PU/PYTHON/Data/Cropper.ui"
# Create a GUI window (inherited from in class App())
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)
# Folder where images are to be stored (can be changed by user)
save_folder = "/PROJECT/SHARED_WORKSPACE/PU/PYTHON/"

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
        self.pB_loadImage.clicked.connect(self.load_image)
        self.pB_saveImage.clicked.connect(self.save_new_image)
        self.l_display.mousePressEvent = self.on_press
        self.l_displayCropped.mousePressEvent = self.on_press
        self.tE_width.setPlainText("256")
        # self.tE_height.setPlainText("256")
        self.tE_path.setPlainText("/media/DATA1/Images_DecuStudy-F_RGB/images/")
        self.tE_savepath.setPlainText("/media/DATA1/images_DB/cropped_images/")

    # Get mouse coordinates, define quadratic new image and save it in save_folder
    def on_press(self, event):
        print(f"[INFO] Mouse coordinates: {event.x(), event.y()}")

        q_cWidth = int(self.tE_width.toPlainText())
        q_cHeight = int(self.tE_width.toPlainText())
        q_r_left = int(event.x() - q_cWidth/2)
        q_r_right = int(event.x() + q_cWidth/2)
        q_r_top =  int(event.y() - q_cHeight/2)
        q_r_bottom = int(event.y() + q_cHeight/2)
        r=QRect(q_r_left, q_r_top, q_cWidth, q_cHeight)
        # print(f"(q_r_left, q_r_top, q_cWidth, q_cHeight): {(q_r_left, q_r_top, q_cWidth, q_cHeight)}")

        im_point_x = np.ceil(int(event.x())-507/2)*8 + 4056/2
        im_point_y = np.ceil(int(event.y())-380/2)*8 + 3040/2

        cWidth = int(self.tE_width.toPlainText())*8
        cHeight = int(self.tE_width.toPlainText())*8
        r_left = int(event.x()*8- cWidth/2) # r_left = int(im_point_x - cWidth/2)
        r_right = int(event.x()*8 + cWidth/2) # r_right = int(im_point_x + cWidth/2)
        r_top =  int(event.y()*8 - cHeight/2) # r_top =  int(im_point_y - cHeight/2)
        r_bottom = int(event.y()*8 + cHeight/2) # r_bottom = int(im_point_y + cHeight/2) 
        print(f"[INFO] (r_left, r_right, r_top, r_top, r_bottom): {(r_left, r_right, r_top, r_bottom)}")
        
        # print(self.im.shape)
        self.new_im = self.im
        self.new_im = self.new_im[max(0,int(r_top)):min(int(r_bottom),3456),max(0,int(r_left)):min(5184,int(r_right))] #(4056, 3040)
        print(f"cropped image size: {self.new_im.shape}")

        #self.new_depth = self.depthim[max(0,int(r_top)):min(int(r_bottom),3040),max(0,int(r_left)):min(4056,int(r_right))]
        


        Q_image = self.Q_image.copy(r).scaled(256, 256)
        self.l_displayCropped.setPixmap(QPixmap.fromImage(Q_image))

        # l_new = cv.resize(self.new_im, (256, 256))
        # bytesPerLine = 3 * l_new.shape[1]
        # Q_image = QImage(l_new.data, l_new.shape[1], l_new.shape[0], bytesPerLine, QImage.Format_RGB888)
        # self.l_displayCropped.setPixmap(QPixmap.fromImage(Q_image))

    # Loading a new image according to path from text edit
    def load_image(self):
        self.folder = self.tE_path.toPlainText()
        im_path = []
        depth_path = []
        name = []
        depth_name = []
        
        try: 
            self.im = cv.imread(self.folder)
        
        except:
            for file in os.listdir(self.folder):
                if 'fullsize.png' in file: 
                    name.append(file)
                    im_path.append(os.path.join(self.folder, file))

            self.im = cv.imread(im_path[len(im_path)-1])
            
        
        frame_rgb_preview = cv.resize(self.im, (648, 432)) #(648, 432) (507, 380)
        frame_rgb_preview = cv.cvtColor(frame_rgb_preview, cv.COLOR_BGR2RGB)
        # plt.imshow(frame_rgb_preview)
        # plt.show()
        bytesPerLine = 3 * frame_rgb_preview.shape[1]
        self.Q_image = QImage(frame_rgb_preview.data, frame_rgb_preview.shape[1], frame_rgb_preview.shape[0], bytesPerLine, QImage.Format_RGB888)
        self.l_display.setPixmap(QPixmap.fromImage(self.Q_image))
        
        #print(f"[INFO] New image loaded: {im_path[len(im_path)-1]}")
        print(f"image size: {frame_rgb_preview.shape}")


    def save_new_image(self):
        path = self.tE_savepath.toPlainText()
        #print(f"path: {path}")
        
        cropped = self.new_im
        print(f"shape of cropped: {cropped.shape}")

        try: 
            cv.imwrite(path, cropped)
        except:

            folder = self.folder
            for i,c in enumerate(folder):
                if c == '/':
                    region = folder[i+1:]
            print(f"region: {region}")
            
            # os.mkdir(os.path.join(path, region))
            save_path = os.path.join(path, str(region)+".png")
            cv.imwrite(save_path, cropped)
        print(f"image saved to: {save_path}")

        #save_depth = os.path.join(save_depth, str(region)+".png")
        #cv.imwrite(save_depth, cropped_depth)

    
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
