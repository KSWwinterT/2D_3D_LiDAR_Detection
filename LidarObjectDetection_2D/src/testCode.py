import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSlot, QTimer
import rosbag
import sensor_msgs
import ros_numpy
import numpy as np
import pyqtgraph as pg
from threading import Thread
import time

class ExMain(QWidget):
    def __init__(self):
        super().__init__()
        self.ALGO_FLAG = 1  # 1: dbscan
        self.clusterLabel = list()
        self.frame = 1
        hbox = QGridLayout()
        self.canvas = pg.GraphicsLayoutWidget()
        hbox.addWidget(self.canvas)
        self.setLayout(hbox)

        self.view = self.canvas.addViewBox()
        self.view.setAspectLocked(True)
        self.view.disableAutoRange()
        self.view.scaleBy(s=(20, 20))
        grid = pg.GridItem()
        self.view.addItem(grid)

        self.setWindowTitle("realtime")

        self.spt = pg.ScatterPlotItem(pen=pg.mkPen(width=1, color='r'), symbol='o', size=2)
        self.view.addItem(self.spt)

        self.pos = None
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

        test_bagfile = '/Users/kimgyeong-yeon/PycharmProjects/pythonProject/purdue/3D/2022-11-11-15-10-22.bag'
        self.bag_file = rosbag.Bag(test_bagfile)

        self.bagthreadFlag = True
        self.bagthread = Thread(target=self.getbagfile)
        self.bagthread.start()

        self.mytimer = QTimer()
        self.mytimer.start(10)
        self.mytimer.timeout.connect(self.get_data)

        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()

        self.show()

    @pyqtSlot()
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
            self.resetObjPos()
            self.doYourAlgorithm(points)
            print("time : ", time.time() - start)

            time.sleep(0.1)

    def detect_people(self, points):
        # Assuming points is a numpy array containing 3D point cloud data
        # Convert points to a 2D image-like format for input to Faster R-CNN
        image_like_data = self.convert_points_to_image(points)

        # Preprocess the image-like data
        transform = transforms.Compose([transforms.ToTensor()])
        image_tensor = transform(image_like_data)

        # Add a batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            prediction = self.model(image_tensor)

        # Get bounding boxes
        boxes = prediction[0]['boxes'].numpy()

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

    def downSampling(self, points):
        # Your downsampling code here
        pass

    def dbscan(self, points):
        # Your DBSCAN code here
        pass

    def doYourAlgorithm(self, points):
        roi = {"x": [-30, 30], "y": [-10, 20], "z": [-1.5, 5.0]}
        x_range = np.logical_and(points[:, 0] >= roi["x"][0], points[:, 0] <= roi["x"][1])
        y_range = np.logical_and(points[:, 1] >= roi["y"][0], points[:, 1] <= roi["y"][1])
        z_range = np.logical_and(points[:, 2] >= roi["z"][0], points[:, 2] <= roi["z"][1])

        pass_through_filter = np.where(np.logical_and(x_range, np.logical_and(y_range, z_range)) == True)[0]
        points = points[pass_through_filter, :]

        # Downsampling
        self.downSampling(points)

        # Object Detection
        if self.ALGO_FLAG == 1:
            detected_boxes = self.detect_people(points)
            self.update_detected_objects(detected_boxes)

        # Clustering
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

            objLength = 1.8
            objHeight = 1.5
            if (abs(x_size) <= objLength + 1) and (abs(y_size) <= objHeight + 1):
                tempobjPos[0] = x
                tempobjPos[1] = y
                tempobjSize[0] = x_size
                tempobjSize[1] = y_size

        self.pos = points
        self.frame += 1

    def update_detected_objects(self, detected_boxes):
        for i, box in enumerate(detected_boxes):
            objpos = self.objsPos[i]
            objsize = self.objsSize[i]
            objpos[0] = box[0]
            objpos[1] = box[1]
            objsize[0] = box[2] - box[0]
            objsize[1] = box[3] - box[1]

    def resetObjPos(self):
        for i, pos in enumerate(self.objsPos):
            pos[0] = 0
            pos[1] = 0
            os = self.objsSize[i]
            os[0] = 0
            os[1] = 0

    def closeEvent(self, event):
        print('closed')
        self.bagthreadFlag = False

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)

    ex = ExMain()

    sys.exit(app.exec_())
