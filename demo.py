import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import threading
import time


class DriverMonitoringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("睿眸 - 智能驾驶员监控系统")
        self.root.geometry("1200x800")

        # 定义危险行为类别和对应的中文标签
        self.behavior_labels = {
            'write': '写字/发短信',
            'phone': '打电话',
            'drunk': '饮酒',
            'organize': '整理物品',
            'radio': '操作收音机',
            'talk': '与乘客交谈',
            'take': '取物',
            # 根据您的模型添加更多类别
        }

        # 定义不同危险级别的颜色
        self.danger_colors = {
            'high': (0, 0, 255),  # 红色 - 高风险
            'medium': (0, 165, 255),  # 橙色 - 中等风险
            'low': (0, 255, 255)  # 黄色 - 低风险
        }

        # 初始化模型
        self.model = None
        self.load_model_button = tk.Button(root, text="加载模型", command=self.load_model)
        self.load_model_button.pack(pady=10)

        # 状态标签
        self.status_label = tk.Label(root, text="模型未加载", fg="red")
        self.status_label.pack()

        # 创建主框架
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)

        # 右侧显示面板
        display_frame = ttk.LabelFrame(main_frame, text="预览", width=700)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        display_frame.pack_propagate(False)

        # 视频显示画布
        self.canvas = tk.Canvas(display_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 控制按钮
        self.image_button = tk.Button(control_frame, text="打开图片", command=self.open_image, width=20,
                                      state=tk.DISABLED)
        self.image_button.pack(pady=5)

        self.video_button = tk.Button(control_frame, text="打开视频", command=self.open_video, width=20,
                                      state=tk.DISABLED)
        self.video_button.pack(pady=5)

        self.camera_button = tk.Button(control_frame, text="开启摄像头", command=self.start_camera, width=20,
                                       state=tk.DISABLED)
        self.camera_button.pack(pady=5)

        self.stop_button = tk.Button(control_frame, text="停止", command=self.stop, width=20, state=tk.DISABLED)
        self.stop_button.pack(pady=5)

        self.save_button = tk.Button(control_frame, text="保存结果", command=self.save_image, width=20,
                                     state=tk.DISABLED)
        self.save_button.pack(pady=5)

        # 置信度阈值滑块
        ttk.Label(control_frame, text="置信度阈值:").pack(pady=(20, 5))
        self.confidence_var = tk.DoubleVar(value=0.5)
        self.confidence_slider = ttk.Scale(control_frame, from_=0.1, to=0.9,
                                           variable=self.confidence_var,
                                           orient=tk.HORIZONTAL)
        self.confidence_slider.pack(fill=tk.X, padx=10, pady=5)
        self.confidence_label = ttk.Label(control_frame, text=f"当前: {self.confidence_var.get():.2f}")
        self.confidence_label.pack()
        self.confidence_slider.configure(command=self.update_confidence_label)

        # 检测结果显示
        ttk.Label(control_frame, text="检测结果:").pack(pady=(20, 5))
        self.results_text = tk.Text(control_frame, height=10, width=30)
        self.results_text.pack(fill=tk.BOTH, padx=10, pady=5)

        # 警告框
        self.alert_frame = ttk.LabelFrame(control_frame, text="警告")
        self.alert_frame.pack(fill=tk.X, padx=10, pady=10)
        self.alert_label = tk.Label(self.alert_frame, text="无危险行为", fg="green", font=("Arial", 12, "bold"))
        self.alert_label.pack(pady=10)

        # 初始化变量
        self.is_running = False
        self.cap = None
        self.current_image = None
        self.detection_results = []
        self.alert_active = False

    def update_confidence_label(self, value):
        self.confidence_label.config(text=f"当前: {float(value):.2f}")

    def load_model(self):
        try:
            self.status_label.config(text="正在加载模型...", fg="orange")
            self.root.update()

            # 加载YOLOv11模型
            self.model = YOLO("yolo11m.pt")

            # 获取模型类别名称
            if hasattr(self.model, 'names'):
                print("模型类别:", self.model.names)

            self.status_label.config(text="模型加载成功", fg="green")
            self.load_model_button.config(state=tk.DISABLED)
            self.image_button.config(state=tk.NORMAL)
            self.video_button.config(state=tk.NORMAL)
            self.camera_button.config(state=tk.NORMAL)

        except Exception as e:
            self.status_label.config(text=f"模型加载错误: {str(e)}", fg="red")

    def open_image(self):
        if not self.model:
            self.status_label.config(text="请先加载模型", fg="red")
            return

        file_path = filedialog.askopenfilename(
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.stop()
            self.current_image = cv2.imread(file_path)
            self.process_frame(self.current_image)

    def open_video(self):
        if not self.model:
            self.status_label.config(text="请先加载模型", fg="red")
            return

        file_path = filedialog.askopenfilename(
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv")]
        )
        if file_path:
            self.stop()
            self.cap = cv2.VideoCapture(file_path)
            self.is_running = True
            self.stop_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.NORMAL)
            self.process_video()

    def start_camera(self):
        if not self.model:
            self.status_label.config(text="请先加载模型", fg="red")
            return

        self.stop()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.config(text="无法打开摄像头", fg="red")
            return

        self.is_running = True
        self.stop_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)
        self.process_video()

    def stop(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.stop_button.config(state=tk.DISABLED)
        self.alert_label.config(text="无危险行为", fg="green")
        self.alert_active = False

    def process_video(self):
        if not self.is_running or not self.cap:
            return

        ret, frame = self.cap.read()
        if ret:
            self.process_frame(frame)

        if self.is_running:
            self.root.after(10, self.process_video)

    def get_danger_level(self, behavior):
        """根据行为类型确定危险级别"""
        high_risk = ['phone', 'write', 'drunk']  # 高风险行为
        medium_risk = ['organize', 'take']  # 中等风险行为
        low_risk = ['talk', 'radio']  # 低风险行为

        if behavior in high_risk:
            return 'high'
        elif behavior in medium_risk:
            return 'medium'
        elif behavior in low_risk:
            return 'low'
        else:
            return 'low'  # 默认低风险

    def process_frame(self, frame):
        # 执行检测
        conf_threshold = self.confidence_var.get()
        results = self.model(frame, conf=conf_threshold, imgsz=640)

        # 提取检测信息
        self.detection_results = []
        result = results[0]

        # 重置警告状态
        self.alert_active = False
        high_risk_detected = False

        if result.boxes is not None:
            boxes = result.boxes.cpu().numpy()
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0])
                confidence = box.conf[0]
                label = result.names[class_id]
                bbox = box.xyxy[0].astype(int)

                # 获取行为的中文标签
                chinese_label = self.behavior_labels.get(label, label)

                # 确定危险级别
                danger_level = self.get_danger_level(label)

                self.detection_results.append({
                    'label': label,
                    'chinese_label': chinese_label,
                    'confidence': confidence,
                    'bbox': bbox,
                    'danger_level': danger_level
                })

                # 根据危险级别选择颜色
                color = self.danger_colors[danger_level]

                # 绘制边界框和标签
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
                cv2.putText(frame, f"{chinese_label} {confidence:.2f}",
                            (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # 检测到高风险行为
                if danger_level == 'high':
                    high_risk_detected = True

        # 更新警告信息
        if high_risk_detected:
            self.alert_label.config(text="高风险行为 detected!", fg="red")
            self.alert_active = True
        else:
            self.alert_label.config(text="无危险行为", fg="green")
            self.alert_active = False

        # 更新结果文本
        self.results_text.delete(1.0, tk.END)
        if self.detection_results:
            for detection in self.detection_results:
                self.results_text.insert(tk.END,
                                         f"{detection['chinese_label']}: {detection['confidence']:.2f} ({detection['danger_level']})\n")
        else:
            self.results_text.insert(tk.END, "未检测到行为")

        # 显示帧
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)

        # 调整图像大小以适应画布，同时保持宽高比
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:
            img_ratio = img.width / img.height
            canvas_ratio = canvas_width / canvas_height

            if canvas_ratio > img_ratio:
                new_height = canvas_height
                new_width = int(new_height * img_ratio)
            else:
                new_width = canvas_width
                new_height = int(new_width / img_ratio)

            img = img.resize((new_width, new_height), Image.LANCZOS)

        self.photo = ImageTk.PhotoImage(image=img)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width // 2, canvas_height // 2,
                                 image=self.photo, anchor=tk.CENTER)

        self.current_image = frame

    def save_image(self):
        if self.current_image is not None:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".jpg",
                filetypes=[("JPEG 文件", "*.jpg"), ("所有文件", "*.*")]
            )
            if file_path:
                # 转换回BGR格式以供OpenCV使用
                img_bgr = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, img_bgr)
                self.status_label.config(text=f"图像已保存到 {file_path}", fg="green")


if __name__ == "__main__":
    root = tk.Tk()
    app = DriverMonitoringApp(root)
    root.mainloop()