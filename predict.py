import cv2
import numpy as np
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from skimage.filters import threshold_multiotsu


# 定義環形卷積層
class CirConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(CirConv2d, self).__init__()
        self.padding = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0)

    def forward(self, x):
        # 如果需要填充，環形處理圖片邊緣
        if self.padding > 0:
            x = torch.cat([x[:, :, :, -self.padding:], x, x[:, :, :, :self.padding]], dim=3)  # 水平填充
            x = torch.cat([x[:, :, -self.padding:, :], x, x[:, :, :self.padding, :]], dim=2)  # 垂直填充
        return self.conv(x)


# 定義融合模塊
class FusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


# 定義 Snake 模塊
class SnakeModule(nn.Module):
    def __init__(self):
        super(SnakeModule, self).__init__()
        self.iterations = 20
        self.smoothing_factor = 0.05

    def forward(self, boundary, features):
        for _ in range(self.iterations):
            boundary = self.refine_boundary(boundary, features)
        return boundary

    def refine_boundary(self, boundary, features):
        return self.smooth_boundary(boundary)

    def smooth_boundary(self, boundary):
        return boundary


# 定義 DeepSnake 模型
class DeepSnakeModel(nn.Module):
    def __init__(self):
        super(DeepSnakeModel, self).__init__()
        # 編碼器部分
        self.encoder = nn.Sequential(
            CirConv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            CirConv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            CirConv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            CirConv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fusion_block = FusionBlock(512, 1024)

        # 解碼器部分
        self.decoder = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.snake_module = SnakeModule()

    def forward(self, x):
        features = self.encoder(x)
        features = self.fusion_block(features)
        initial_boundary = self.decoder(features)
        refined_boundary = self.snake_module(initial_boundary, features)
        return refined_boundary


# 預處理函數
def apply_contrast_stretching(image, clip_limit=2.5, tile_grid_size=(10, 10)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def adaptive_multi_otsu(image, total_pixels):
    # Step 1: Apply dynamic contrast stretching
    image = apply_contrast_stretching(image)

    # Step 2: Perform 3-class Multi-Otsu thresholding
    T_3 = threshold_multiotsu(image, classes=3)
    regions_3 = np.digitize(image, bins=T_3)

    # Initialize final mask
    final_mask = np.zeros_like(image)

    # Calculate white pixel count and percentage once
    white_pixels_3 = (regions_3 == 2)  # Create boolean mask
    white_pixel_count_3 = np.sum(white_pixels_3)  # Count true values
    percentage_3 = (white_pixel_count_3 / total_pixels) * 100

    # Check both conditions and process accordingly
    if percentage_3 > 4.0:
        if percentage_3 > 7.0:
            # Only compute 4-class threshold if needed
            T_4 = threshold_multiotsu(image, classes=4)
            regions_4 = np.digitize(image, bins=T_4)
            final_mask[regions_4 == 3] = 255
        else:
            # Use pre-computed boolean mask
            final_mask[white_pixels_3] = 255

    return final_mask

def process_image_with_components(img):
    """
    Enhanced image preprocessing with connected components analysis
    """
    # Get total pixel count
    total_pixels = img.size

    # Apply adaptive multi-Otsu thresholding
    binary_img = adaptive_multi_otsu(img, total_pixels)

    # Apply median filtering to the binarized image with a 5x5 window
    median_filtered_gt = cv2.medianBlur(binary_img, 5)

    # Perform connected components analysis on the filtered image
    num_labels_gt, labeled_mask_gt = cv2.connectedComponents(median_filtered_gt, connectivity=4)

    # Get the center x-coordinate of the image
    center_x = img.shape[1] // 2

    # Blank image to store filtered components
    filtered_image_gt = np.zeros_like(img)

    # Filter connected components based on criteria
    for label in range(1, num_labels_gt):
        component_mask = (labeled_mask_gt == label).astype(np.uint8)
        area = np.sum(component_mask)
        moments = cv2.moments(component_mask)

        if moments['mu02'] != 0:
            angle = 0.5 * np.arctan2(2 * moments['mu11'], moments['mu20'] - moments['mu02'])
            angle_degrees = np.degrees(angle)
        else:
            angle_degrees = 0

        x, y, w, h = cv2.boundingRect(component_mask)
        width_height_ratio = w / h if h > 0 else 0
        centroid_x = int(moments['m10'] / (moments['m00'] + 1e-5))

        # Apply filtering criteria
        if (area > 300 and
                abs(angle_degrees) <= 25 and
                abs(centroid_x - center_x) <= 40 and
                50 <= w <= 300 and
                50 <= h <= 250):
            filtered_image_gt[labeled_mask_gt == label] = 255

    return filtered_image_gt


def process_single_image(input_path, model_path, output_dir):
    """
    處理單張圖片的函數，現在使用增強的預處理
    """
    try:
        # 建立輸出目錄
        os.makedirs(output_dir, exist_ok=True)

        # 讀取圖片
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {input_path}")

        # 使用新的預處理函數
        processed_img = process_image_with_components(img)

        # 載入模型
        model = DeepSnakeModel()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # 準備輸入張量
        input_tensor = torch.tensor(processed_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        input_tensor = input_tensor.to(device)

        # 執行推論
        with torch.no_grad():
            output = model(input_tensor)

        # 後處理輸出
        pred = output.clone()
        pred[pred >= 0.5] = 255
        pred_img = pred[0, 0].cpu().numpy().astype(np.uint8)

        # 儲存結果
        image_name = os.path.basename(input_path)
        preprocessed_file = os.path.join(output_dir, f"preprocessed_{image_name}")
        output_file = os.path.join(output_dir, f"prediction_{image_name}")

        cv2.imwrite(preprocessed_file, processed_img)
        cv2.imwrite(output_file, pred_img)

        print(f"Successfully processed image:")
        print(f"Original image: {input_path}")
        print(f"Preprocessed image saved to: {preprocessed_file}")
        print(f"Prediction saved to: {output_file}")

    except Exception as e:
        print(f"Error during processing: {str(e)}")


# [predict 函數也需要類似的修改]
def predict(input_dir, model_path, output_dir, batch_size=4):
    """
    對資料夾中的圖片進行預處理和DeepSnake預測，並將兩種結果分別儲存。
    """
    # 建立輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'preprocessed'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'deepsnake'), exist_ok=True)

    # 獲取所有輸入圖片
    input_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.png')])

    # 加載模型
    model = DeepSnakeModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(input_files), batch_size), desc='Processing'):
            batch_files = input_files[i:i + batch_size]
            batch_imgs = []
            batch_preprocessed = []

            for file in batch_files:
                try:
                    # 讀取原始圖片
                    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"Error reading image: {file}")
                        continue

                    # 使用新的預處理函數
                    processed_img = process_image_with_components(img)

                    # 儲存預處理結果
                    preprocessed_file = os.path.join(output_dir, 'preprocessed', os.path.basename(file))
                    cv2.imwrite(preprocessed_file, processed_img)

                    batch_preprocessed.append(processed_img)
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")
                    continue

            if not batch_preprocessed:
                continue

            # 將預處理後的圖片轉換為張量
            input_tensors = torch.tensor(batch_preprocessed, dtype=torch.float32).unsqueeze(1) / 255.0
            input_tensors = input_tensors.to(device)

            # DeepSnake模型預測
            outputs = model(input_tensors)

            # 處理DeepSnake結果
            pred = outputs.clone()
            pred[pred >= 0.5] = 255
            pred[pred < 0.5] = 0

            # 儲存DeepSnake結果
            for j, file in enumerate(batch_files):
                if j < len(pred):
                    pred_img = pred[j, 0].cpu().numpy().astype(np.uint8)
                    output_file = os.path.join(output_dir, 'deepsnake', os.path.basename(file))
                    cv2.imshow(output_file, pred_img)
if __name__ == "__main__":
    # 設定基礎路徑
    base_input_dir = r'C:/new data/valid/images'
    model_path = 'C:/Users/tim/PycharmProjects/pythonProject/deep snake.pt'
    output_dir = r'C:/new data/prediction/deepsnake_pred'

    while True:
        try:
            # 讓使用者輸入圖片名稱
            image_name = input("請輸入圖片名稱（例如：1.png）或輸入 'q' 退出：")

            # 檢查是否要退出
            if image_name.lower() == 'q':
                print("程式結束")
                break

            # 組合完整的圖片路徑
            if not image_name.endswith('.png'):
                image_name += '.png'

            input_path = os.path.join(base_input_dir, image_name)

            # 檢查檔案是否存在
            if not os.path.exists(input_path):
                print(f"錯誤：找不到檔案 {input_path}")
                continue

            # 處理單張圖片
            process_single_image(input_path, model_path, output_dir)

            # 詢問是否繼續處理其他圖片
            continue_process = input("是否繼續處理其他圖片？(y/n): ")
            if continue_process.lower() != 'y':
                print("程式結束")
                break

        except Exception as e:
            print(f"發生錯誤：{str(e)}")
            print("請重新輸入")
