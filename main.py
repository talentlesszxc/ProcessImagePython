import sys #для передачи argv в QApplication
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets, Qt, QtGui, QtCore
from PyQt5.QtGui import QPixmap

import design
import os
import cv2
import numpy as np
from keras.models import Model, load_model
import datetime

class Everything(QtWidgets.QMainWindow, design.Ui_MainWindow, QGraphicsView):
#В этом классе взаимодействие с интерфейсом и тд, всё в нём.
    def __init__(self):
        super().__init__()
        self.setupUi(self) #Инициализация дизайна
        #self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        #self.scene = Qt.QGraphicsScene()
        #self.graphicsView = Qt.QGraphicsView()
        #self.graphicsView.setScene(self.scene)
        #self.setCentralWidget(self.graphicsView)
        #self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        #self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.choose_file_button.clicked.connect(self.browse_folder)
        self.process_button.clicked.connect(self.process_image)

    def browse_folder(self): #выбор изображения + его отображение
        self.filename = QtWidgets.QFileDialog.getOpenFileName(self, "Выбор изображения", "D:/Nadenenko/POKAZ","Image Files (*.png *.jpg *.PNG *.jpeg)")#"Выберите изображение")
        if self.filename:
            print (self.filename[0])
            self.scene = Qt.QGraphicsScene()
            pixmap = QPixmap(self.filename[0])
            item = Qt.QGraphicsPixmapItem(pixmap)
            #self.graphicsView.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            self.scene.addItem(item)
            self.graphicsView.setScene(self.scene)  
            
            #QGraphicsView.ScrollHandDrag            
            
    def process_image(self):        
        print("Model loading starts: ", datetime.datetime.now().time())
        model = load_model("D:/Nadenenko/POKAZ/model_31k_params_v2.h5")
        print("Model loading ends: ", datetime.datetime.now().time())        
        image = cv2.imread(self.filename[0])
        height, width, channels = image.shape
        nearest_x = width//256 * 256
        nearest_y = height//128 * 128
        resized = cv2.resize(image, (nearest_x, nearest_y), interpolation = cv2.INTER_AREA)
        dataset = np.zeros(shape=(int(resized.shape[0]/128*resized.shape[1]/256),
                        128, 256,3), dtype=np.float32)
        #dataset = np.ndarray(shape=(int(resized.shape[0]/128*resized.shape[1]/256),
                        #128, 256,3), dtype=np.float32)
        counter = 0
        print(dataset.size)
        for j in range(0, resized.shape[0], 128):
            for i in range (0, resized.shape[1], 256):    
                cropped = np.array(resized[j:j+128,i:i+256])         
                dataset[counter] = cropped
                counter += 1
        print("Preprocessing ended at: ", datetime.datetime.now().time())
        print(dataset.size)
        pred = model.predict(dataset)
        print("Preds done at: ", datetime.datetime.now().time())
        blank_image = np.zeros(shape = (resized.shape[0], resized.shape[1],3))
        count = 0
        for j in range(0, resized.shape[0], 128):
            for i in range (0, resized.shape[1], 256):
                blank_image[j:j+128, i:i+256] = pred[count]
                count += 1
        #cv2.imwrite("D:/Nadenenko/POKAZ/from_gui.png", blank_image)
        blank_image = (blank_image * 255).round().astype(np.uint8)
        blank_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(blank_image, 50, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        imgcopy = image.copy()
        z = cv2.fillPoly(imgcopy, contours, color=(250,0,0))
        #blank_image = cv2.imread("D:/Nadenenko/POKAZ/image1.png")
        #self.frame = QtGui.QImage(blank_image.data, blank_image.shape[0], blank_image.shape[1], 256*3, QtGui.QImage.Format_RGB888)
        #self.scene.addPixmap(QtGui.QPixmap.fromImage(self.frame))
        #self.scene.update()
        #self.graphicsView.setScene(self.scene)
        
        #image = cv2.imread("D:/Nadenenko/POKAZ/image1.png")
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = blank_image.shape[:2]

        self.scene = Qt.QGraphicsScene()
        self.scene.clear()
        self.frame = QtGui.QImage(z.data, width, height, QtGui.QImage.Format_RGB888)
        self.scene.addPixmap(QtGui.QPixmap.fromImage(self.frame))
        self.scene.update()
        self.graphicsView.setScene(self.scene)
def main():
    app = QtWidgets.QApplication(sys.argv) #Новый экземпляр QApplication
    window = Everything() #Создание объекта класса Everything
    window.show() #Показать окно
    #window.plot()
    app.exec_() #Запустить приложение    
    
if __name__ == '__main__':
    main()