import pyqtgraph.opengl as gl
import pyqtgraph as pg
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSlot, QTimer
import rosbag
import sensor_msgs
import ros_numpy
import numpy as np
from threading import Thread
import time
import torch
import cv2
import pathlib
from sklearn.cluster import DBSCAN

class ExMain(QWidget):
    def __init__(self):
        super().__init__()
        self.ALGO_FLAG = 1  # 1 : dbscan
        self.clusterLabel = list()
        self.frame = 1
        hbox = QGridLayout()
        self.canvas = pg.GraphicsLayoutWidget()
        hbox.addWidget(self.canvas)
        self.setLayout(hbox)

        self.view = gl.GLViewWidget()
        self.canvas.addItem(self.view)

        self.setWindowTitle("realtime")

        self.scatter_widget = gl.GLScatterPlotItem()
        self.view.addItem(self.scatter_widget)

        # Load bagfile
        test_bagfile = '/Users/kimgyeong-yeon/PycharmProjects/pythonProject/purdue/3D/2022-11-11-15-10-22.bag'
        self.bag_file = rosbag.Bag(test_bagfile)

        # ROS thread
        self.bagthreadFlag = True
        self.bagthread = Thread(target=self.getbagfile)
        self.bagthread.start()

        self.mytimer = QTimer()
        self.mytimer.start(10)
        self.mytimer.timeout.connect(self.get_data)

        # Load YOLO model
        self.net = cv2.dnn.readNetFromDarknet('/path/to/yolo/config/yolov3.cfg',
                                              '/path/to/yolo/weights/yolov3.weights')
        self.layer_names = self.net.getUnconnectedOutLayersNames()

        self.show()

    @pyqtSlot()
    def get_data(self):
        if self.pos is not None:
            self.scatter_widget.setData(pos=self.pos, color='r', size=2)

            # Perform YOLO detection
            boxes = self.detect_people(self.pos)

            # Display 2D bounding boxes
            img = np.zeros((600, 800, 3), dtype=np.uint8)
            for box in boxes:
                x, y, w, h = box
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            cv2.imshow('YOLO Detection', img)

    def detect_people(self, points):
        # Assuming points is a numpy array containing 3D point cloud data

        # Convert points to a 2D image-like format for input to YOLO
        image_like_data = self.convert_points_to_image(points)

        # Perform YOLO detection
        blob = cv2.dnn.blobFromImage(image_like_data, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.layer_names)

        # Get bounding boxes
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 0:  # Class 0 corresponds to 'person'
                    center_x = int(detection[0] * image_like_data.shape[1])
                    center_y = int(detection[1] * image_like_data.shape[0])
                    w = int(detection[2] * image_like_data.shape[1])
                    h = int(detection[3] * image_like_data.shape[0])
                    x = center_x - w / 2
                    y = center_y - h / 2
                    boxes.append((x, y, w, h))

        return boxes

    def convert_points_to_image(self, points):
        # You need to implement this function based on your data and coordinate system
        # This is just a placeholder function
        # You may need to convert 3D points to a 2D image-like format for detection
        # Ensure the output has channels (e.g., RGB or grayscale) suitable for the model
        # You may need to take into account the coordinate system and scaling of your lidar data
        # For simplicity, this function currently returns a placeholder image
        # Please adjust this function according to your specific use case
        return np.zeros((256, 256, 3), dtype=np.uint8)

    def getbagfile(self):
        read_topic = '/velodyne_points'

        for topic, msg, t in self.bag_file.read_messages(read_topic):
            if self.bagthreadFlag is False:
                break

            msg.__class__ = sensor_msgs.msg._PointCloud2.PointCloud2
            pc = ros_numpy.numpify(msg)
            points = np.zeros((pc.shape[0], 3))
            points[:, 0] = pc['x']
            points[:, 1] = pc['y']
            points[:, 2] = pc['z']

            start = time.time()
            self.doYourAlgorithm(points)
            print("time : ", time.time() - start)

            time.sleep(0.1)

    def dbscan(self, points):
        dbscan = DBSCAN(eps=1, min_samples=20, algorithm='ball_tree').fit(points)
        self.clusterLabel = dbscan.labels_

    def doYourAlgorithm(self, points):
        roi = {"x": [-30, 30], "y": [-10, 20], "z": [-1.5, 5.0]}

        x_range = np.logical_and(points[:, 0] >= roi["x"][0], points[:, 0] <= roi["x"][1])
        y_range = np.logical_and(points[:, 1] >= roi["y"][0], points[:, 1] <= roi["y"][1])
        z_range = np.logical_and(points[:, 2] >= roi["z"][0], points[:, 2] <= roi["z"][1])

        pass_through_filter = np.where(np.logical_and(x_range, np.logical_and(y_range, z_range)) == True)[0]
        points = points[pass_through_filter, :]

        # Clustering
        if self.ALGO_FLAG == 1:
            self.dbscan(points)

        self.pos = points

    def closeEvent(self, event):
        print('closed')
        self.bagthreadFlag = False

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)

    ex = ExMain()

    sys.exit(app.exec_())
