import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_wm811k_data(data_dir, resize=(256, 256)):
    """
    加载WM-811K晶圆数据集
    data_dir: 数据集解压后的根目录（需自行从Kaggle下载：https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map）
    resize: 图像缩放尺寸
    返回：预处理后的图像数组、标签数组、标签映射表
    """
    # 读取标签文件
    label_df = pd.read_csv(os.path.join(data_dir, "LSWMD.csv"))

    # 筛选有效数据（去除无缺陷标签的样本）
    label_df = label_df.dropna(subset=["failureType"])
    label_df = label_df.reset_index(drop=True)

    # 标签映射（6类缺陷+1类正常，基础分类）
    defect_mapping = {
        "none": 0,
        "Center": 1,
        "Edge-Loc": 2,
        "Edge-Ring": 3,
        "Loc": 4,
        "Random": 5,
        "Scratch": 6
    }
    label_df["label"] = label_df["failureType"].map(defect_mapping)

    # 加载图像（晶圆图像存储在waferMap列中，为2D数组）
    images = []
    labels = []
    for idx in range(len(label_df)):
        # 提取晶圆图像（转为uint8格式）
        wafer_map = label_df.iloc[idx]["waferMap"]
        wafer_map = (wafer_map * 255).astype(np.uint8)

        # 缩放图像（统一尺寸）
        wafer_map = cv2.resize(wafer_map, resize)

        # 转为3通道（方便后续处理）
        wafer_map = cv2.cvtColor(wafer_map, cv2.COLOR_GRAY2BGR)

        images.append(wafer_map)
        labels.append(label_df.iloc[idx]["label"])

    # 转为numpy数组
    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int32)

    # 划分训练集/测试集（8:2）
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )

    return X_train, X_test, y_train, y_test, defect_mapping


def preprocess_image(image):
    """图像预处理：灰度化→高斯模糊→阈值分割"""
    # 灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 高斯模糊（去噪）
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 自适应阈值分割（分离缺陷与背景）
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=11, C=2
    )

    # 形态学操作（去除小噪点）
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return thresh


if __name__ == "__main__":
    # 测试数据加载（需替换为你的数据集路径）
    data_dir = "./WM-811K"  # 解压后的数据集目录
    X_train, X_test, y_train, y_test, defect_map = load_wm811k_data(data_dir)
    print(f"训练集尺寸：{X_train.shape}, 测试集尺寸：{X_test.shape}")
    print(f"标签映射：{defect_map}")

    # 测试预处理
    sample_img = X_train[0]
    processed_img = preprocess_image(sample_img)
    cv2.imshow("原始图像", sample_img)
    cv2.imshow("预处理后图像", processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()