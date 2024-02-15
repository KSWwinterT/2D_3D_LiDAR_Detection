import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud
import time
import sys

import pyqtgraph as pg
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSlot, QTimer
import open3d as open3d
import numpy as np
import yaml
from sklearn.cluster import DBSCAN

class LidarNode(Node):
    def __init__(self):
        super().__init__('lidar_node')

        self.ALGO_FLAG = 1  # 1 : dbscan
        self.clusterLabel = list()
        self.frame = 1

        vbox = QVBoxLayout()
        self.canvas = pg.GraphicsLayoutWidget()
        vbox.addWidget(self.canvas)

        self.view = self.canvas.addViewBox()
        self.view.setAspectLocked(True)
        self.view.disableAutoRange()
        self.view.scaleBy(s=(20, 20))
        grid = pg.GridItem()
        self.view.addItem(grid)

        self.setWindowTitle("realtime")

        self.spt = pg.ScatterPlotItem(pen=pg.mkPen(width=1, color='r'), symbol='o', size=2)
        self.view.addItem(self.spt)

        self.objs = list()
        self.objsPos = list()
        self.objsSize = list()

        numofobjs = 100
        for i in range(numofobjs):
            obj = pg.QtWidgets.QGraphicsRectItem(-0.5, -0.5, 0.5, 0.5)
            obj.setPen(pg.mkPen('w'))
            self.view.addItem(obj)
            self.objs.append(obj)

            pos = [0, 0, 0]
            size = [0, 0, 0]
            self.objsPos.append(pos)
            self.objsSize.append(size)

        self.bagfile_path = '/path/to/your/lidar_data.bag'
        self.load_bagfile(self.bagfile_path)

        self.mytimer = QTimer()
        self.mytimer.start(10)
        self.mytimer.timeout.connect(self.get_data)

    def load_bagfile(self, bagfile_path):
        with open(bagfile_path.replace('.bag', '_metadata.yaml'), 'rb') as stream:
            try:
                metadata = yaml.safe_load(stream)
                self.point_cloud_topic = metadata['point_cloud_topic']
            except yaml.YAMLError as exc:
                print(exc)

        self.create_subscription(PointCloud, self.point_cloud_topic, self.point_cloud_callback, 10)

    def point_cloud_callback(self, msg):
        pc = np.zeros((msg.width * msg.height, 3))
        pc[:, 0] = msg['x']
        pc[:, 1] = msg['y']
        pc[:, 2] = msg['z']

        start = time.time()
        self.resetObjPos()
        self.doYourAlgorithm(pc)
        print("time : ", time.time() - start)

    def get_data(self):
        if self.pos is not None:
            self.spt.setData(pos=self.pos)

        for i, obj in enumerate(self.objs):
            objpos = self.objsPos[i]
            objsize = self.objsSize[i]
            if objpos[0] == 0 and objpos[1] == 0:
                obj.setVisible(False)
            else:
                obj.setVisible(True)
                obj.setRect(objpos[0], objpos[1], objsize[0], objsize[1])

    def resetObjPos(self):
        for i, pos in enumerate(self.objsPos):
            pos[0] = 0
            pos[1] = 0
            os = self.objsSize[i]
            os[0] = 0
            os[1] = 0

    def doYourAlgorithm(self, points):
        roi = {"x": [-30, 30], "y": [-10, 20], "z": [-1.5, 5.0]}

        x_range = np.logical_and(points[:, 0] >= roi["x"][0], points[:, 0] <= roi["x"][1])
        y_range = np.logical_and(points[:, 1] >= roi["y"][0], points[:, 1] <= roi["y"][1])
        z_range = np.logical_and(points[:, 2] >= roi["z"][0], points[:, 2] <= roi["z"][1])

        pass_through_filter = np.where(
            np.logical_and(x_range, np.logical_and(y_range, z_range)) == True)[0]
        points = points[pass_through_filter, :]

        if self.ALGO_FLAG == 1:
            self.dbscan(points)

        clusterCnt = max(self.clusterLabel) + 1
        for i in range(1, clusterCnt):
            tempobjPos = self.objsPos[i]
            tempobjSize = self.objsSize[i]

            index = np.asarray(np.where(self.clusterLabel == i))

            if np.max(points[index, 0]) < 0:
                x = np.min(points[index, 0])
                y = np.min(points[index, 1])
                x_size = np.max(points[index, 0]) - np.min(points[index, 0])
                y_size = np.max(points[index, 1]) - np.min(points[index, 1])
            else:
                x = np.max(points[index, 0])
                y = np.max(points[index, 1])
                x_size = -(np.max(points[index, 0]) - np.min(points[index, 0]))
                y_size = -(np.max(points[index, 1]) - np.min(points[index, 1]))

            objLength = 0.5
            objHeight = 0.3
            if (abs(x_size) <= objLength + 1) and (abs(y_size) <= objHeight + 1):
                tempobjPos[0] = x
                tempobjPos[1] = y
                tempobjSize[0] = x_size
                tempobjSize[1] = y_size

        self.pos = points

    def dbscan(self, points):
        dbscan = DBSCAN(eps=1, min_samples=20, algorithm='ball_tree').fit(points)
        self.clusterLabel = dbscan.labels_

class MainWindow(QMainWindow):
    def __init__(self, lidar_node):
        super().__init__()

        self.lidar_node = lidar_node
        self.setCentralWidget(self.lidar_node)

if __name__ == '__main__':
    rclpy.init()
    app = QApplication([])
    lidar_node = LidarNode()
    main_window = MainWindow(lidar_node)
    main_window.show()
    sys.exit(app.exec_())
