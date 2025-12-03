import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from data_loader import load_wm811k_data, preprocess_image


def extract_defect_features(processed_img):
    """提取缺陷特征：面积、周长、圆形度、缺陷数量"""
    # 查找轮廓（缺陷区域）
    contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features = []
    defect_count = len(contours)  # 缺陷数量

    if defect_count == 0:
        # 无缺陷（全为背景）
        return [0.0, 0.0, 0.0, 0]

    # 计算每个缺陷的特征，取最大值（重点关注最大缺陷）
    max_area = 0
    max_perimeter = 0
    max_circularity = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)  # 面积
        perimeter = cv2.arcLength(cnt, True)  # 周长
        if perimeter == 0:
            circularity = 0
        else:
            circularity = 4 * np.pi * area / (perimeter ** 2)  # 圆形度（1为完美圆形）

        if area > max_area:
            max_area = area
        if perimeter > max_perimeter:
            max_perimeter = perimeter
        if circularity > max_circularity:
            max_circularity = circularity

    features.extend([max_area, max_perimeter, max_circularity, defect_count])
    return features


def defect_classifier(features, defect_mapping):
    """基础规则分类器（替代机器学习，降低难度）：根据特征判断缺陷类型"""
    area, perimeter, circularity, defect_count = features

    # 规则设计（基于晶圆缺陷的物理特征）
    if defect_count == 0:
        return 0  # 正常（none）
    elif area > 1000 and circularity > 0.7:
        return 1  # 中心缺陷（Center）
    elif area > 800 and defect_count <= 3:
        return 2  # 边缘定位缺陷（Edge-Loc）
    elif perimeter > 500 and defect_count > 5:
        return 3  # 边缘环缺陷（Edge-Ring）
    elif area < 500 and defect_count > 10:
        return 5  # 随机缺陷（Random）
    elif perimeter > 800 and circularity < 0.3:
        return 6  # 划痕缺陷（Scratch）
    else:
        return 4  # 定位缺陷（Loc）


def evaluate_model(X_test, y_test, defect_mapping):
    """评估模型性能"""
    y_pred = []
    for img in X_test:
        # 预处理→提取特征->分类
        processed_img = preprocess_image(img)
        features = extract_defect_features(processed_img)
        pred = defect_classifier(features, defect_mapping)
        y_pred.append(pred)

    # 计算评估指标
    print("分类报告：")
    print(classification_report(
        y_test, y_pred,
        target_names=defect_mapping.keys(),
        zero_division=0
    ))

    # 混淆矩阵可视化
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("缺陷检测混淆矩阵")
    plt.colorbar()
    tick_marks = np.arange(len(defect_mapping))
    plt.xticks(tick_marks, defect_mapping.keys(), rotation=45)
    plt.yticks(tick_marks, defect_mapping.keys())
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

    return y_pred


def visualize_detection_result(img, processed_img, true_label, pred_label, defect_mapping):
    """可视化检测结果"""
    # 绘制轮廓（在原始图像上标记缺陷）
    contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_img = img.copy()
    cv2.drawContours(result_img, contours, -1, (0, 255, 0), 2)

    # 添加标签文本
    true_label_name = [k for k, v in defect_mapping.items() if v == true_label][0]
    pred_label_name = [k for k, v in defect_mapping.items() if v == pred_label][0]
    cv2.putText(
        result_img, f"True: {true_label_name}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2
    )
    cv2.putText(
        result_img, f"Pred: {pred_label_name}", (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
    )

    # 显示图像
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("原始图像")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(processed_img, cmap="gray")
    plt.title("预处理后（缺陷分割）")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title("检测结果（绿色轮廓为缺陷）")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("detection_result.png")
    plt.show()


if __name__ == "__main__":
    # 1. 加载数据
    data_dir = "./WM-811K"  # 替换为你的数据集路径
    X_train, X_test, y_train, y_test, defect_mapping = load_wm811k_data(data_dir)

    # 2. 模型评估
    y_pred = evaluate_model(X_test, y_test, defect_mapping)

    # 3. 可视化部分测试样本结果（选前5个）
    for i in range(5):
        img = X_test[i]
        processed_img = preprocess_image(img)
        true_label = y_test[i]
        pred_label = y_pred[i]
        visualize_detection_result(img, processed_img, true_label, pred_label, defect_mapping)