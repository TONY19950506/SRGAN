import cv2
import numpy as np

# 這個函數會讀取一個圖像文件，應用拉普拉斯算子，並保存處理後的圖像
def apply_and_save_laplacian(image_path, output_path):
    # 讀取圖像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 讀取圖像並轉為灰度圖

    if img is None:
        print(f"Error loading image {image_path}")
        return

    # 應用拉普拉斯算子
    laplacian = cv2.Laplacian(img, cv2.CV_64F)

    # 將拉普拉斯算子的結果轉換為 8 位無符號整數
    laplacian = cv2.convertScaleAbs(laplacian)

    # 保存處理後的圖像
    cv2.imwrite(output_path, laplacian)
    print(f"Edge-detected image saved to {output_path}")

# 替換 'your_image.jpg' 為你要處理的圖像文件的路徑
# 替換 'output_image.jpg' 為你想要保存處理後圖像的路徑和文件名
apply_and_save_laplacian('C:/Users/Tony/Desktop/SRGAN-master/EEGAN_jpg_dataset_256_to_1024_srgan/set2_1.jpg', 'set2_1_lap_black.jpg')
