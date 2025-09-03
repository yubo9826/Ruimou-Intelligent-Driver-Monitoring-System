import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("基于深度学习的危险驾驶行为检测识别系统")

        self.model = self.load_model('yolo11n.pt', force_reload=True)

        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack()

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.open_button = tk.Button(root, text="打开图片", command=self.open_image)
        self.open_button.pack()

        self.open_video_button = tk.Button(root, text="打开视频", command=self.open_video)
        self.open_video_button.pack()

        self.save_button = tk.Button(root, text="保存", command=self.save_image)
        self.save_button.pack()

        self.exit_button = tk.Button(root, text="退出", command=root.quit)
        self.exit_button.pack()

        self.image = None
        self.video_capture = None

    def load_model(self, model_path, force_reload):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=force_reload)
        model.conf = 0.25  # 置信度阈值
        model.iou = 0.70  # 交并比阈值
        return model

    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            self.display_image()

    def open_video(self):
        self.video_capture = cv2.VideoCapture(0)
        self.update_frame()

    def update_frame(self):
        if self.video_capture is not None:
            ret, frame = self.video_capture.read()
            if ret:
                self.image = frame
                self.display_image()
            self.root.after(10, self.update_frame)

    def display_image(self):
        if self.image is not None:
            results = self.model(self.image)
            results.show()
            labels, cord = results.xyxy[0][:, -1], results.xyxy[0][:, :-1]
            for label, (x1, y1, x2, y2) in zip(labels, cord):
                cv2.rectangle(self.image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(self.image, f'{label.item()}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (255, 0, 0), 2)

            image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)
            self.image_label.config(image=image)
            self.image_label.image = image

    def save_image(self):
        if self.image is not None:
            cv2.imwrite('saved_image.jpg', self.image)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()