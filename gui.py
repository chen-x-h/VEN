import json
import os.path
import queue
import sys
import time

import SimpleITK as sitk
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QFileDialog, QMessageBox, QButtonGroup, QRadioButton, QMdiSubWindow, QMdiArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# 设置中文字体和样式
from vedo import Plotter, Volume

from do_pred import Predictor

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong', 'Arial']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# pyinstaller -D --name ven_demo --noconfirm ^
#   --add-data "C:\Users\yejp2\anaconda3\envs\pytorch_gpu\Lib\site-packages\vedo\fonts;vedo/fonts" ^
#   --hidden-import=numpy.core.multiarray ^
#   --hidden-import=numpy.core.umath ^
#   --hidden-import=numpy.core._methods ^
#   --hidden-import=numpy.lib.format ^
#   gui.py

class LongRunningTaskThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            start = time.perf_counter()
            result = self.func(*self.args, **self.kwargs)
            print(f'using {(time.perf_counter()-start)*1000}ms')
            self.finished.emit(result)
        except Exception as e:
            print(e)
            self.error.emit(str(e))


class Mask3DViewer:
    def __init__(self):
        self.vp = None
        self.actor = None
        self.add_actor = None
        self.rm_actor = None
        self.update_queue = queue.Queue()
        self.data_to_update = None
        self.is_open = False
        self.last_mask_data = None
        self.show_diff = True

    def start(self):
        """在主线程中启动 3D 查看器"""
        if self.is_open:
            return

        # 创建 Plotter（必须在主线程）
        self.vp = Plotter(offscreen=False, bg='black', axes=1)
        self.vp.show(interactive=0)  # 非阻塞
        self.is_open = True
        print("3D viewer started")

    def update_mask(self, mask_data):
        """由主线程调用，线程安全"""
        if not self.is_open:
            return
        # 将数据放入队列（可选：用于异步处理大数据
        if self.data_to_update is not None:
            self.last_mask_data = (self.data_to_update > 0).astype(np.uint8)
        self.data_to_update = (mask_data > 0).astype(np.uint8)
        self._apply_update()

    def _apply_update(self):
        """在主线程中执行更新（安全）"""
        if self.vp is None or self.data_to_update is None:
            return

        mask_data = self.data_to_update

        # 移除旧 actor
        if self.actor is not None:
            self.vp.remove(self.actor)
        if self.add_actor is not None:
            self.vp.remove(self.add_actor)
        if self.rm_actor is not None:
            self.vp.remove(self.rm_actor)

        # 创建 Volume 和 mesh
        if not self.show_diff or self.last_mask_data is None:
            vol = Volume(mask_data.astype(np.uint8), spacing=(1, 1, 1))
            self.actor = vol.isosurface().alpha(1.).c('red')
            self.vp.add(self.actor)
        else:
            pass

        if self.last_mask_data is not None and self.show_diff:
            print('show diff')
            a = mask_data
            b = self.last_mask_data
            vol = Volume(((a == b) & (a != 0)).astype(np.uint8) * mask_data.astype(np.int8), spacing=(1, 1, 1))
            self.actor = vol.isosurface().alpha(1.).c('red')
            self.vp.add(self.actor)

            vol = Volume(((a != b) & (a != 0)).astype(np.uint8), spacing=(1, 1, 1))
            self.add_actor = vol.isosurface().alpha(1.).c('green')
            self.vp.add(self.add_actor)

            vol = Volume(((a != b) & (b != 0)).astype(np.uint8), spacing=(1, 1, 1))
            self.rm_actor = vol.isosurface().alpha(0.5).c('blue')
            self.vp.add(self.rm_actor)

        self.vp.reset_camera()
        # if len(self.vp.actors) <= 1:
        #     self.actor.normalize()  # 可选：归一化到单位大小
        #     self.actor.pos(0, 0, 0)  # 确保位置在原点（可选）
        #     # 添加到场景
        #     self.vp.add(self.actor)
        #     self.vp.reset_camera()  # 最常用，自

        self.vp.render()

    def close(self):
        """关闭查看器"""
        if self.vp:
            self.vp.close()
        self.vp = None
        self.actor = None
        self.is_open = False
        print("3D viewer closed")


class MedicalImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.doing_re = False
        self.model_running = False
        self.thread = None
        self.clicking = False

        # 数据变量
        self.image_data = None
        self.mask_data = None
        self.image_sitk = None  # 保存原始 SimpleITK 对象以获取 affine
        self.mask_sitk = None
        self.current_slice = 0
        self.mode = 'none'  # 'draw' 或 'erase' 或 'none'
        self.brush_size = 1  # 半径，实际是 (2*r+1)x(2*r+1)
        config_path = './ven_config.json'

        base_config = {
            'model_path': './checkpoints/VENLite_MSD8',
        }
        if not os.path.exists(config_path):
            configs = base_config
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(configs, f, ensure_ascii=False, indent=4)

        with open(config_path,
                  'r',
                  encoding='utf-8') as load_f:
            configs = json.load(load_f)

        self.setWindowTitle("VEN Demo")
        self.setGeometry(100, 100, 1500, 1200)
        self.model_path = configs['model_path']
        self.predictor = Predictor(device='cpu')
        self.predictor.load_model(path=self.model_path)
        self.mask_3d_viewer = Mask3DViewer()
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        # main_layout = QHBoxLayout(central_widget)
        layout = QVBoxLayout(central_widget)
        # main_layout.addLayout(layout)

        # Matplotlib 画布
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        layout.addWidget(self.canvas)

        # 控件区域
        control_layout = QHBoxLayout()

        # 切片滑块
        control_layout.addWidget(QLabel("Slice:"))
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(1)
        self.slice_slider.setValue(0)
        self.slice_slider.valueChanged.connect(self.update_slice)
        control_layout.addWidget(self.slice_slider)

        # 在 initUI 方法中添加如下行：
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

        # 模式选择：绘制 or 擦除
        self.mode_group = QButtonGroup()
        self.btn_none = QRadioButton("Disable")
        self.btn_none.setChecked(True)
        self.btn_draw = QRadioButton("Draw")
        self.btn_erase = QRadioButton("Erase")
        self.mode_group.addButton(self.btn_draw)
        self.mode_group.addButton(self.btn_erase)
        self.mode_group.addButton(self.btn_none)
        self.mode_group.buttonClicked.connect(self.update_mode)
        control_layout.addWidget(self.btn_draw)
        control_layout.addWidget(self.btn_erase)
        control_layout.addWidget(self.btn_none)

        self.mode_group_diff = QButtonGroup()
        self.btn_diff = QRadioButton("Show Diff")
        self.btn_diff.setChecked(True)
        self.btn_no_diff = QRadioButton("No Diff")
        self.mode_group_diff.addButton(self.btn_diff)
        self.mode_group_diff.addButton(self.btn_no_diff)
        self.mode_group_diff.buttonClicked.connect(self.update_mode_diff)
        control_layout.addWidget(self.btn_diff)
        control_layout.addWidget(self.btn_no_diff)

        # 笔刷大小
        control_layout.addWidget(QLabel("Brush Size:"))
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setMinimum(1)
        self.brush_slider.setMaximum(10)
        self.brush_slider.setValue(self.brush_size)
        self.brush_slider.valueChanged.connect(self.update_brush_size)
        control_layout.addWidget(self.brush_slider)

        # 按钮区
        self.btn_load_img = QPushButton("Load CT Image")
        self.btn_load_img.clicked.connect(self.load_image)
        control_layout.addWidget(self.btn_load_img)

        self.btn_load_mask = QPushButton("Load")
        self.btn_load_mask.clicked.connect(self.load_mask)
        control_layout.addWidget(self.btn_load_mask)

        self.btn_new_mask = QPushButton("New")
        self.btn_new_mask.clicked.connect(self.new_mask)
        control_layout.addWidget(self.btn_new_mask)

        self.btn_save_mask = QPushButton("Save")
        self.btn_save_mask.clicked.connect(self.save_mask)
        self.btn_save_mask.setEnabled(False)
        control_layout.addWidget(self.btn_save_mask)

        self.btn_show_3d = QPushButton("3D")
        self.btn_show_3d.clicked.connect(self.show_3d_view)
        control_layout.addWidget(self.btn_show_3d)

        self.btn_ai_mask = QPushButton("Auto")
        self.btn_ai_mask.clicked.connect(self.do_pred)
        control_layout.addWidget(self.btn_ai_mask)

        self.btn_ai_p_mask = QPushButton("Refine")
        self.btn_ai_p_mask.clicked.connect(self.do_pred_p)
        control_layout.addWidget(self.btn_ai_p_mask)

        self.btn_re_img_mask = QPushButton("Resample")
        self.btn_re_img_mask.clicked.connect(self.do_resample)
        control_layout.addWidget(self.btn_re_img_mask)

        self.gpu_group = QButtonGroup()
        self.btn_cpu = QRadioButton("cpu")
        self.btn_cpu.setChecked(True)
        self.btn_gpu = QRadioButton("gpu")
        self.gpu_group.addButton(self.btn_cpu)
        self.gpu_group.addButton(self.btn_gpu)
        self.gpu_group.buttonClicked.connect(self.update_network)
        control_layout.addWidget(self.btn_cpu)
        control_layout.addWidget(self.btn_gpu)

        layout.addLayout(control_layout)
        self.clicking = False

        self.reset_view()

    def on_scroll(self, event):
        """
        处理鼠标滚轮事件，用于调整切片滑动条。
        """
        # 获取当前的切片索引
        current_slice = self.slice_slider.value()

        # 判断滚动方向，event.step > 0 表示向上滚动，反之向下
        if event.button == 'up':
            new_slice = min(current_slice + 1, self.image_data.shape[2] - 1)  # 避免超出上限
        elif event.button == 'down':
            new_slice = max(current_slice - 1, 0)  # 避免低于下限
        else:
            new_slice = current_slice

        # 如果新的切片索引与当前不同，则更新
        if new_slice != current_slice:
            # 更新滑动条的位置（这会触发 valueChanged 信号，进而调用 update_slice）
            self.slice_slider.setValue(new_slice)

    def show_3d_view(self):
        if self.mask_data is None:
            QMessageBox.warning(self, "Warning", "Please load a mask first")
            return

        if not hasattr(self, 'mask_3d_viewer'):
            self.mask_3d_viewer = Mask3DViewer()
        self.mask_3d_viewer.is_open = False
        self.mask_3d_viewer.start()  # 启动 3D 查看器
        self.mask_3d_viewer.update_mask(self.mask_data)  # 发送初始数据

    def reset_view(self):
        self.ax.clear()
        self.ax.text(0.5, 0.5, 'Load a CT image to start', color='white',
                     ha='center', va='center', fontsize=14)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')
        self.figure.patch.set_facecolor('gray')
        self.canvas.draw()

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load a CT image", "", "NIfTI Files (*.nii *.nii.gz)")
        if not file_path:
            return

        try:
            img_sitk = sitk.ReadImage(file_path)
            self.image_sitk = img_sitk
            self.image_data = sitk.GetArrayFromImage(img_sitk).transpose((1, 2, 0))  # ZYX -> XYZ

            self.current_slice = 0
            self.slice_slider.setMaximum(self.image_data.shape[2] - 1)
            self.slice_slider.setValue(self.current_slice)
            self.adjust_canvas_size()
            # self.adjust_canvas_size()
            # self.status_label.setText(f"已加载图像: {os.path.basename(file_path)}, 形状: {self.image_data.shape}")
            self.update_display()
            self.predictor.load_img(self.image_data.transpose((2, 0, 1)), self.image_sitk.GetSpacing())
            self.new_mask()
        except Exception as e:
            print(e)
            QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")

    def new_mask(self):
        if self.image_data is None:
            QMessageBox.warning(self, "Warning", "Please load an image first")
            return
        self.mask_data = np.zeros_like(self.image_data).astype(np.int32)
        self.mask_sitk = None
        # self.status_label.setText(f"已加载掩码: {os.path.basename(file_path)}")
        self.btn_save_mask.setEnabled(True)
        self.update_display()

    def load_mask(self):
        if self.image_data is None:
            QMessageBox.warning(self, "Warning", "Please load an image first")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Please load a mask", "", "NIfTI Files (*.nii *.nii.gz)")
        if not file_path:
            return

        try:
            mask_sitk = sitk.ReadImage(file_path)
            mask_data = sitk.GetArrayFromImage(mask_sitk).transpose((1, 2, 0))  # ZYX -> XYZ

            if mask_data.shape != self.image_data.shape:
                QMessageBox.critical(self, "Error", "Mask size does not match image")
                return

            self.mask_data = mask_data.astype(np.int32)
            self.mask_sitk = mask_sitk
            # self.status_label.setText(f"已加载掩码: {os.path.basename(file_path)}")
            self.btn_save_mask.setEnabled(True)
            self.update_display()
        except Exception as e:
            print(e)
            QMessageBox.critical(self, "Error", f"Failed to load mask: {str(e)}")

    def update_mode(self):
        self.mode = 'none'
        if self.btn_draw.isChecked():
            self.mode = 'draw'
        if self.btn_erase.isChecked():
            self.mode = 'erase'

    def update_mode_diff(self):
        self.mask_3d_viewer.show_diff = self.btn_diff.isChecked()

    def update_network(self):
        if self.model_running:
            return
        device = 'cpu'
        # if self.btn_npu.isChecked():
        #     device = 'npu'
        if self.btn_gpu.isChecked():
            device = 'gpu'
        check = self.predictor.change_device(device)
        if check:
            print(f'Switched to {device}')
        else:
            self.btn_cpu.setChecked(True)
            QMessageBox.critical(self, "Error", f"Failed to load {device}")

    def update_brush_size(self, value):
        self.brush_size = value

    def update_slice(self, value):
        self.current_slice = value
        self.update_display()

    def update_display(self):
        self.ax.clear()
        if self.image_data is None:
            return

        slice_img = self.image_data[:, :, self.current_slice]
        self.ax.imshow(slice_img, cmap='gray')

        if self.mask_data is not None:
            mask_slice = self.mask_data[:, :, self.current_slice]
            colored_mask = np.zeros((*mask_slice.shape, 4))
            colored_mask[mask_slice > 0, 0] = 1.0  # 红色
            colored_mask[mask_slice > 0, 3] = 0.5  # 透明度
            self.ax.imshow(colored_mask)

        self.ax.set_title(f"Slice {self.current_slice}", color='white')
        self.ax.axis('off')
        self.figure.patch.set_facecolor('black')
        self.canvas.draw()

    def on_mouse_press(self, event):
        self.clicking = True
        if event.inaxes != self.ax or self.mask_data is None:
            return
        self.modify_mask(event)

    def on_mouse_move(self, event):
        if not self.clicking:
            return
        if event.inaxes == self.ax and self.mask_data is not None:
            self.modify_mask(event)

    def on_mouse_release(self, event):
        self.clicking = False

    def modify_mask(self, event):
        if self.mode == 'none':
            return
        x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
        print(f'{x}, {y}')
        if not (0 <= x < self.mask_data.shape[0] and 0 <= y < self.mask_data.shape[1]):
            return

        label_value = 1 if self.mode == 'draw' else 0
        r = self.brush_size

        # 创建一个圆形笔刷（避免方形边缘）
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                # if dx * dx + dy * dy <= r * r:  # 圆形判断
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.mask_data.shape[0] and 0 <= ny < self.mask_data.shape[1]:
                    self.mask_data[ny, nx, self.current_slice] = label_value

        self.update_display()  # 实时刷新
        if hasattr(self, 'mask_3d_viewer'):
            if self.mask_3d_viewer.is_open:
                self.mask_3d_viewer.update_mask(self.mask_data)

    def save_mask(self):
        if self.mask_data is None or self.image_sitk is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save mask", "", "NIfTI Files (*.nii.gz *.nii)")
        if not file_path:
            return

        try:
            # 转回 SimpleITK 格式：XYZ -> ZYX，并创建新图像
            mask_to_save = self.mask_data.transpose((2, 0, 1)).copy()
            new_mask_sitk = sitk.GetImageFromArray(mask_to_save)
            new_mask_sitk.SetOrigin(self.image_sitk.GetOrigin())
            new_mask_sitk.SetSpacing(self.image_sitk.GetSpacing())
            new_mask_sitk.SetDirection(self.image_sitk.GetDirection())

            sitk.WriteImage(new_mask_sitk, file_path)
            # self.status_label.setText(f"掩码已保存: {os.path.basename(file_path)}")
            QMessageBox.information(self, "Success", "Mask Saved！")
        except Exception as e:
            print(e)
            QMessageBox.critical(self, "Error", f"Failed to save: {str(e)}")

    def adjust_canvas_size(self):
        """根据图像尺寸自适应调整 Matplotlib 画布大小"""
        if self.image_data is None:
            return

        # 获取当前切片的宽高 (width, height)
        h, w = self.image_data.shape[0], self.image_data.shape[1]  # 注意：shape 是 (H, W, D)
        aspect_ratio = w / h  # 宽高比

        # 基础高度（单位：英寸）
        fig_height = 8  # 固定高度，避免窗口过大
        fig_width = fig_height * aspect_ratio

        # 限制最大最小尺寸（防止过大或过小）
        max_width = 12
        min_width = 6
        fig_width = np.clip(fig_width, min_width, max_width)

        # 设置 figure 大小
        self.figure.set_size_inches(fig_width, fig_height)

        # 调整 canvas 尺寸（自动适配）
        self.canvas.updateGeometry()  # 触发布局更新
        self.canvas.draw()

        # # 可选：调整主窗口大小（基于画布）
        # # 计算推荐窗口宽度（画布 + 控件）
        control_panel_width = 300  # 估计控件区域宽度
        total_width = int(fig_width * 80) + control_panel_width  # 粗略像素换算
        total_height = int(fig_height * 80) + 200  # 加上滑块、按钮等

        # 限制最大值（适应屏幕）
        screen = QApplication.primaryScreen().size()
        total_width = min(total_width, screen.width() * 4 // 5)
        total_height = min(total_height, screen.height() * 4 // 5)

        self.resize(total_width, total_height)

    def on_task_finished(self, result):
        """子线程完成，更新主界面"""
        # 启用按钮
        self.btn_ai_mask.setEnabled(True)
        self.btn_ai_p_mask.setEnabled(True)
        self.model_running = False
        # self.status_label.setText("任务完成")
        # 更新数据
        old_gt = self.mask_data[:, :, self.current_slice].copy()

        self.mask_data = result.transpose((1, 2, 0))

        if not np.array_equal(self.mask_data[:, :, self.current_slice], old_gt):
            print('change current gt')

        self.update_display()  # 刷新 2D 显示
        if hasattr(self, 'mask_3d_viewer') and self.mask_3d_viewer.is_open:
            self.mask_3d_viewer.update_mask(self.mask_data)  # 更新 3D 显示

        # QMessageBox.information(self, "完成", "推理完成！")
        print("Task done.")

    def on_task_error(self, error_msg):
        """处理Error"""
        self.btn_ai_mask.setEnabled(True)
        self.btn_ai_p_mask.setEnabled(True)
        self.model_running = False
        print(error_msg)
        QMessageBox.critical(self, "Error", f"{error_msg}")

    def do_pred_p(self):
        if self.image_data is None or self.mask_data is None:
            QMessageBox.warning(self, "Warning", "Please load a img and mask first")
            return

            # 👇 禁用按钮，防止重复点击
        self.btn_ai_mask.setEnabled(False)
        self.btn_ai_p_mask.setEnabled(False)
        self.model_running = True
        # self.status_label.setText("正在执行耗时操作...")

        # 创建线程并启动
        self.thread = LongRunningTaskThread(
            self.predictor.pred,  # 假设 predict 是耗时函数
            self.current_slice,
            self.mask_data.transpose((2, 0, 1))
        )
        self.thread.finished.connect(self.on_task_finished)
        self.thread.error.connect(self.on_task_error)
        self.thread.start()

    def do_pred(self):
        """
        推理全部
        :return:
        """
        if self.image_data is None or self.mask_data is None:
            QMessageBox.warning(self, "Warning", "Please load a img and mask first")
            return

            # 👇 禁用按钮，防止重复点击
        self.btn_ai_mask.setEnabled(False)
        self.btn_ai_p_mask.setEnabled(False)
        self.model_running = True
        # self.status_label.setText("正在执行耗时操作...")

        # 创建线程并启动
        self.thread = LongRunningTaskThread(
            self.predictor.pred_all,  # 假设 predict 是耗时函数
            self.mask_data.transpose((2, 0, 1))
        )
        self.thread.finished.connect(self.on_task_finished)
        self.thread.error.connect(self.on_task_error)
        self.thread.start()

        print("Long task started (running in background)")

    def on_task_finished_re(self, result):
        """子线程完成，更新主界面"""
        # 启用按钮
        self.btn_re_img_mask.setEnabled(True)
        self.doing_re = False
        new_itk = result
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save resampled image", "", "NIfTI Files (*.nii.gz *.nii)")
        if not file_path:
            return
        sitk.WriteImage(new_itk, file_path)

    def on_task_error_re(self, error_msg):
        """处理Error"""
        self.btn_re_img_mask.setEnabled(True)
        self.doing_re = False
        print(error_msg)
        QMessageBox.critical(self, "Error", f"{error_msg}")

    def do_resample(self):
        """
        执行重采样 TODO
        :return:
        """
        if self.image_data is None:
            QMessageBox.warning(self, "Warning", "Please load a img first")
            return

            # 👇 禁用按钮，防止重复点击
        self.btn_re_img_mask.setEnabled(False)
        self.doing_re = True
        # self.status_label.setText("正在执行耗时操作...")

        # 创建线程并启动
        self.thread = LongRunningTaskThread(
            self.predictor.img_resample,  # 假设 predict 是耗时函数
            self.image_sitk
        )
        self.thread.finished.connect(self.on_task_finished_re)
        self.thread.error.connect(self.on_task_error_re)
        self.thread.start()

        print("Long task started (running in background)")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = MedicalImageViewer()
    viewer.show()
    sys.exit(app.exec_())
