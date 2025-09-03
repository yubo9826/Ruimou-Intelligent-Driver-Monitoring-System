from ultralytics import YOLO
import torch

def main():
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        device = "0"  # 使用第一个GPU
        print(f"使用GPU进行训练: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"  # 使用CPU
        print("未检测到GPU，使用CPU进行训练（速度较慢）")

    # 加载模型
    model = YOLO("yolo11n.pt")

    # 训练模型
    train_results = model.train(
        data="ruimou.yaml",  # 数据集 YAML 路径
        epochs=1,  # 训练轮次
        imgsz=640,  # 训练图像尺寸
        device=device,  # 自动选择设备
    )

    # 其余代码保持不变...
    # 评估模型在验证集上的性能
    metrics = model.val()

    # 在图像上执行对象检测
    results = model("img_3.jpg")
    results[0].show()

    # 将模型导出为 ONNX 格式
    path = model.export(format="onnx")  # 返回导出模型的路径

if __name__ == "__main__":
    main()