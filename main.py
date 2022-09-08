# encoding:utf-8
import numpy as np
from PySide2.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QPlainTextEdit, QMessageBox
from PySide2.QtGui import QIcon
import cv2
import os
import imghdr


class MainWindow:
    def __init__(self):
        self.label_err = None
        self.window_err = None
        self.center = None
        self.image = None

        self.window = QMainWindow()
        self.window.resize(500, 400)
        self.window.move(300, 300)
        self.window.setWindowTitle('图像变换')

        self.textEdit_add = QPlainTextEdit(self.window)
        self.textEdit_add.setPlaceholderText("请输入要处理的图片地址")
        self.textEdit_add.move(10, 25)
        self.textEdit_add.resize(455, 30)

        self.textEdit_angle = QPlainTextEdit(self.window)
        self.textEdit_angle.setPlaceholderText("请输入要旋转的角度")
        self.textEdit_angle.move(10, 65)
        self.textEdit_angle.resize(145, 30)

        self.textEdit_scale = QPlainTextEdit(self.window)
        self.textEdit_scale.setPlaceholderText("请输入要缩放的大小")
        self.textEdit_scale.move(165, 65)
        self.textEdit_scale.resize(145, 30)

        self.textEdit_flip = QPlainTextEdit(self.window)
        self.textEdit_flip.setPlaceholderText("请输入要翻转的代码")
        self.textEdit_flip.move(320, 65)
        self.textEdit_flip.resize(145, 30)

        self.button_addconfirm = QPushButton('确认地址', self.window)
        self.button_addconfirm.move(360, 180)

        self.button_rotated = QPushButton('旋转缩放', self.window)
        self.button_rotated.move(360, 220)

        self.button_flip = QPushButton('图片翻转', self.window)
        self.button_flip.move(360, 260)

        self.button_histogram = QPushButton('彩色直方图', self.window)
        self.button_histogram.move(360, 300)

        self.button_color_equal = QPushButton('彩直均衡化', self.window)
        self.button_color_equal.move(360, 340)

        self.label = QLabel(self.window)
        self.label.resize(250, 120)
        self.label.move(65, 160)
        self.label.setText(
            "请尽量输入纯英文路径\n旋转角度绝对值不大于360\n\n翻转代码：\n水平翻转输入1\n垂直翻转输入0\n两者皆要输入-1")

        self.button_addconfirm.clicked.connect(self.handle_calc_add_confirm)
        self.button_rotated.clicked.connect(self.handle_calc_rotated)
        self.button_flip.clicked.connect(self.handle_calc_flip)
        self.button_histogram.clicked.connect(self.handle_histogram_display)
        self.button_color_equal.clicked.connect(self.handle_color_equal)

    @staticmethod
    def calc_draw_hist(image, color):
        hist = cv2.calcHist([image], [0], None, [256], [0.0, 255.0])
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
        histImg = np.zeros([256, 256, 3], np.uint8)
        hpt = int(0.9 * 256);

        for h in range(256):
            intensity = int(hist[h] * hpt / maxVal)
            cv2.line(histImg, (h, 256), (h, 256 - intensity), color)

        return histImg;

    def handle_calc_add_confirm(self):
        info_add = self.textEdit_add.toPlainText()

        img_type_list = {'jpg', 'bmp', 'png', 'jpeg', 'tif'}
        flag = 0

        if not os.path.exists(info_add):
            self.handle_calc_error()
            return

        for item in img_type_list:
            # imghdr 可以用来判断文件是否是图片
            if imghdr.what(info_add) == item:
                flag = 1
                break
            else:
                flag = 0

        if not flag:
            self.handle_calc_error()
            return

        # self.image = cv2.imread(info_add, -1)
        self.image = cv2.imdecode(np.fromfile(info_add, dtype=np.uint8), -1)

        (self.height, self.width) = self.image.shape[:2]
        self.center = (self.width / 2, self.height / 2)

        cv2.imshow("Original", self.image)
        cv2.waitKey(0)

    def handle_calc_rotated(self):
        info_angle = self.textEdit_angle.toPlainText()
        info_scale = self.textEdit_scale.toPlainText()

        if info_angle == "":
            info_angle = 0
        if info_scale == "":
            info_scale = 1

        if float(info_angle) > 360 or float(info_angle) < -360:
            self.handle_calc_error()
            return

        # 旋转缩放矩阵：(旋转中心，旋转角度，缩放因子)
        m = cv2.getRotationMatrix2D(
            self.center, float(info_angle), float(info_scale))
        rotated = cv2.warpAffine(self.image, m, (self.width, self.height))

        cv2.imshow("Rotated by %s Degrees, Scale is %s" %
                   (info_angle, info_scale), rotated)
        cv2.waitKey(0)

    def handle_calc_flip(self):
        info_flip = self.textEdit_flip.toPlainText()
        int_info_flip = int(info_flip)
        horiz = "Flipped Horizontally"
        vert = "Flipped Vertically"
        ho_ve = "Flipped Horizontally & Vertically"

        if info_flip == "":
            self.handle_calc_error()
            return
        elif int_info_flip == 1:
            last_print = horiz
        elif int_info_flip == 0:
            last_print = vert
        elif int_info_flip == -1:
            last_print = ho_ve
        else:
            self.handle_calc_error()
            return

        flipped = cv2.flip(self.image, int_info_flip)

        cv2.imshow("%s" % last_print, flipped)
        cv2.waitKey(0)

    def handle_histogram_display(self):
        self.image = cv2.imread("./test.jpg", -1)  # 测试用临时地址
        b, g, r = cv2.split(self.image)

        histImgB = self.calc_draw_hist(b, [255, 0, 0])
        histImgG = self.calc_draw_hist(g, [0, 255, 0])
        histImgR = self.calc_draw_hist(r, [0, 0, 255])

        cv2.imshow("histImgB", histImgB)
        cv2.imshow("histImgG", histImgG)
        cv2.imshow("histImgR", histImgR)
        cv2.waitKey(0)

    def handle_color_equal(self):
        self.image = cv2.imread("./test.jpg", -1)  # 测试用临时地址
        (b, g, r) = cv2.split(self.image)

        bH = cv2.equalizeHist(b)
        gH = cv2.equalizeHist(g)
        rH = cv2.equalizeHist(r)

        # 合并每一个通道
        result = cv2.merge((bH, gH, rH))

        cv2.imshow("Original", self.image)
        cv2.imshow("Result", result)
        cv2.waitKey(0)

    def handle_calc_error(self):
        self.window_err = QMainWindow()
        self.window_err.resize(300, 100)
        self.window_err.move(400, 400)
        self.window_err.setWindowTitle('Error!')

        self.label_err = QLabel(self.window_err)
        self.label_err.resize(250, 80)
        self.label_err.move(10, 10)
        self.label_err.setText(
            "Please enter a valid value!")

        self.window_err.show()
        app.exec_()


app = QApplication([])
app.setWindowIcon(QIcon('./logo.png'))
win = MainWindow()
win.window.show()
app.exec_()
