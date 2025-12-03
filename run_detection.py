"""晶圆缺陷检测一键运行脚本"""
import os
import sys
from defect_detection import evaluate_model, visualize_detection_result
from data_loader import load_wm811k_data, preprocess_image


def main():
    # 检查数据集路径
    data_dir = "./WM-811K"
    if not os.path.exists(data_dir):
        print("错误：未找到数据集！")
        print("请从Kaggle下载WM-811K数据集：https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map")
        print("解压后将文件夹命名为WM-811K，放在当前脚本同级目录下")
        sys.exit(1)
        #默认测试集是从网上找的
        #导入自己的数据集！


    # 1. 加载数据
    print("正在加载数据集...")
    X_train, X_test, y_train, y_test, defect_mapping = load_wm811k_data(data_dir)
    print(f"数据加载完成！训练集：{X_train.shape}，测试集：{X_test.shape}")

    # 2. 模型评估
    print("\n正在评估模型性能...")
    y_pred = evaluate_model(X_test, y_test, defect_mapping)

    # 3. 可视化典型结果
    print("\n正在生成检测结果可视化...")
    # 选择不同标签的样本（各选1个）
    unique_labels = np.unique(y_test)
    for label in unique_labels[:3]:  # 选前3类标签可视化
        idx = np.where(y_test == label)[0][0]
        img = X_test[idx]
        processed_img = preprocess_image(img)
        true_label = y_test[idx]
        pred_label = y_pred[idx]
        visualize_detection_result(img, processed_img, true_label, pred_label, defect_mapping)

    print("\n检测完成！结果文件已保存：confusion_matrix.png、detection_result.png")


if __name__ == "__main__":
    main()