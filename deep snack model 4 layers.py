import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from skimage.filters import threshold_multiotsu
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter


# Define CirConv2d class
class CirConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(CirConv2d, self).__init__()
        self.padding = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0)

    def forward(self, x):
        if self.padding > 0:
            x = torch.cat([x[:, :, :, -self.padding:], x, x[:, :, :, :self.padding]], dim=3)
            x = torch.cat([x[:, :, -self.padding:, :], x, x[:, :, :self.padding, :]], dim=2)
        return self.conv(x)


# Custom dataset class
class MaskDataset(Dataset):
    def __init__(self, gt_dir, pred_dir):
        self.gt_files = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith('.png')])
        self.pred_files = sorted([os.path.join(pred_dir, f) for f in os.listdir(pred_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, idx):
        # Load ground truth and predicted images in grayscale
        gt_img = cv2.imread(self.gt_files[idx], cv2.IMREAD_GRAYSCALE)
        pred_img = cv2.imread(self.pred_files[idx], cv2.IMREAD_GRAYSCALE)

        def apply_contrast_stretching(image, clip_limit=2.5, tile_grid_size=(10, 10)):
            # Use CLAHE (Contrast Limited Adaptive Histogram Equalization) for dynamic contrast stretching
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

        # Get total pixel count
        total_pixels = gt_img.size
        # Apply adaptive multi-Otsu thresholding
        binary_img = adaptive_multi_otsu(gt_img, total_pixels)

        # Apply median filtering to the binarized image with a 5x5 window
        median_filtered_gt = cv2.medianBlur(binary_img, 5)

        # Perform connected components analysis on the filtered image
        num_labels_gt, labeled_mask_gt = cv2.connectedComponents(median_filtered_gt, connectivity=4)

        # List to store valid components that meet criteria
        valid_components_gt = []

        # Blank image to store filtered components
        filtered_image_gt = np.zeros_like(gt_img)

        # Get the center x-coordinate of the image
        center_x = gt_img.shape[1] // 2

        # Filter connected components in the ground truth image based on criteria
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
            if area > 300 and abs(angle_degrees) <= 25 and abs(
                    centroid_x - center_x) <= 40 and  50 <= w <= 300 and 50 <= h <= 250:
                valid_components_gt.append((label, area))
                filtered_image_gt[labeled_mask_gt == label] = 255

        gtMask = torch.tensor(filtered_image_gt, dtype=torch.float32).unsqueeze(0) / 255.0
        predMask = torch.tensor(pred_img, dtype=torch.float32).unsqueeze(0) / 255.0
        filename = os.path.basename(self.gt_files[idx])

        return gtMask, predMask, filename
# FusionBlock class
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


# DeepSnake model
class DeepSnakeModel(nn.Module):
    def __init__(self):
        super(DeepSnakeModel, self).__init__()
        self.encoder = nn.Sequential(
            CirConv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 512 -> 256

            CirConv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 256 -> 128

            CirConv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128 -> 64

            CirConv2d(256, 512, kernel_size=3, padding=1),  # Additional layer
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 64 -> 32
        )

        self.fusion_block = FusionBlock(512, 1024)  # Adjusted to match the new encoder output size

        self.decoder = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 32 -> 64
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 64 -> 128
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 128 -> 256
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 256 -> 512
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 512 -> 1024
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.snake_module = SnakeModule()

    def forward(self, x):
        features = self.encoder(x)
        features = self.fusion_block(features)  # Apply the fusion block
        initial_boundary = self.decoder(features)
        refined_boundary = self.snake_module(initial_boundary, features)
        return refined_boundary


class SnakeModule(nn.Module):
    def __init__(self):
        super(SnakeModule, self).__init__()
        self.iterations = 10
        self.smoothing_factor = 0.05

    def forward(self, boundary, features):
        for _ in range(self.iterations):
            boundary = self.refine_boundary(boundary, features)
        return boundary

    def refine_boundary(self, boundary, features):
        smoothed_boundary = self.smooth_boundary(boundary)
        return smoothed_boundary

    def smooth_boundary(self, boundary):
        return boundary  # This is a placeholder; implement as needed


class IoUCalculator:

    @staticmethod
    def main():
        gt_dir = r'C:/new data/train/images'
        pred_dir = r'C:/new data/train/label'
        output_dir = r'C:/new data/predicion/10deepsnake_predict1'

        # Create output directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'gt'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'pred'), exist_ok=True)

        learning_rate = 0.0001
        batch_size = 4
        num_epochs = 150

        dataset = MaskDataset(gt_dir, pred_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = DeepSnakeModel()
        if torch.cuda.is_available():
            model.cuda()

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Metrics dictionary
        metrics = {
            'epoch': [],
            'loss': [],
            'accuracy': [],
            'dsc': [],
            'specificity': [],
            'sensitivity': [],
            'precision': [],
            'iou': []
        }

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            for gtMask, predMask, filename in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
                if torch.cuda.is_available():
                    gtMask, predMask = gtMask.cuda(), predMask.cuda()

                outputs = model(predMask)
                loss = criterion(outputs, gtMask)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(dataloader)
            metrics['epoch'].append(epoch + 1)
            metrics['loss'].append(epoch_loss)

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

            # Initialize metrics for this epoch
            epoch_accuracy = 0.0
            epoch_dsc = 0.0
            epoch_specificity = 0.0
            epoch_sensitivity = 0.0
            epoch_precision = 0.0
            epoch_iou = 0.0

            if (epoch + 1) % 4 == 0:
                model.eval()
                with torch.no_grad():
                    for i, (gtMask, predMask, filename) in enumerate(dataloader):
                        if torch.cuda.is_available():
                            gtMask, predMask = gtMask.cuda(), predMask.cuda()

                        outputs = model(predMask)
                        binarized_outputs = (outputs > 0.5).float()

                        accuracy = IoUCalculator.calculateAccuracy(gtMask, binarized_outputs)
                        dsc = IoUCalculator.calculateDSC(gtMask, binarized_outputs)
                        specificity = IoUCalculator.calculateSpecificity(gtMask, binarized_outputs)
                        sensitivity = IoUCalculator.calculateSensitivity(gtMask, binarized_outputs)
                        precision = IoUCalculator.calculatePrecision(gtMask, binarized_outputs)
                        iou = IoUCalculator.calculateIoU(gtMask, binarized_outputs)

                        epoch_accuracy += accuracy
                        epoch_dsc += dsc
                        epoch_specificity += specificity
                        epoch_sensitivity += sensitivity
                        epoch_precision += precision
                        epoch_iou += iou

                        # Save ground truth and predicted images
                        gt_mask = gtMask.cpu().numpy()[0, 0] * 255
                        pred_mask = binarized_outputs.cpu().numpy()[0, 0] * 255

                        cv2.imwrite(os.path.join(output_dir, 'gt', f'epoch_{epoch + 1}_sample_{i + 1}.png'), gt_mask)
                        cv2.imwrite(os.path.join(output_dir, 'pred', f'epoch_{epoch + 1}_sample_{i + 1}.png'),
                                    pred_mask)

                    # Compute average metrics for the epoch
                    num_samples = len(dataloader)
                    epoch_accuracy /= num_samples
                    epoch_dsc /= num_samples
                    epoch_specificity /= num_samples
                    epoch_sensitivity /= num_samples
                    epoch_precision /= num_samples
                    epoch_iou /= num_samples

                    metrics['accuracy'].append(epoch_accuracy)
                    metrics['dsc'].append(epoch_dsc)
                    metrics['specificity'].append(epoch_specificity)
                    metrics['sensitivity'].append(epoch_sensitivity)
                    metrics['precision'].append(epoch_precision)
                    metrics['iou'].append(epoch_iou)

                    print(
                        f'Accuracy: {epoch_accuracy:.4f}, DSC: {epoch_dsc:.4f}, Specificity: {epoch_specificity:.4f}, '
                        f'Sensitivity: {epoch_sensitivity:.4f}, Precision: {epoch_precision:.4f}, IoU: {epoch_iou:.4f}')
            else:
                metrics['accuracy'].append(None)
                metrics['dsc'].append(None)
                metrics['specificity'].append(None)
                metrics['sensitivity'].append(None)
                metrics['precision'].append(None)
                metrics['iou'].append(None)

        # Save metrics to CSV
            metrics_df = pd.DataFrame(metrics)
            metrics_df.to_csv('10deep snake1_metrics.csv', index=False)

        # Save the model
        torch.save(model.state_dict(), os.path.join(output_dir, '10deep snake1.pt'))

        # Plot metrics
        IoUCalculator.plot_metrics(metrics)

    @staticmethod
    def calculateAccuracy(gt, pred):
        correct = torch.sum(gt == pred).item()
        total = gt.numel()
        return correct / total if total > 0 else 0

    @staticmethod
    def calculateDSC(gt, pred):
        intersection = torch.sum(gt * pred).item()
        total = torch.sum(gt).item() + torch.sum(pred).item()
        return (2. * intersection) / total if total > 0 else 0

    @staticmethod
    def calculateSpecificity(gt, pred):
        true_negatives = torch.sum((gt == 0) & (pred == 0)).item()
        false_positives = torch.sum((gt == 0) & (pred == 1)).item()
        denominator = true_negatives + false_positives
        return true_negatives / denominator if denominator > 0 else 0

    @staticmethod
    def calculateSensitivity(gt, pred):
        true_positives = torch.sum((gt == 1) & (pred == 1)).item()
        false_negatives = torch.sum((gt == 1) & (pred == 0)).item()
        denominator = true_positives + false_negatives
        return true_positives / denominator if denominator > 0 else 0

    @staticmethod
    def calculatePrecision(gt, pred):
        true_positives = torch.sum((gt == 1) & (pred == 1)).item()
        false_positives = torch.sum((gt == 0) & (pred == 1)).item()
        denominator = true_positives + false_positives
        return true_positives / denominator if denominator > 0 else 0

    @staticmethod
    def calculateIoU(gt, pred):
        intersection = torch.sum(gt * pred).item()
        union = torch.sum((gt + pred) > 0).item()
        return intersection / union if union > 0 else 0


    @staticmethod
    def apply_smoothing(data, window_length=7):
        """
        Apply Savitzky-Golay filter to smooth the data.
        For shorter sequences, adjust window_length to be odd and less than sequence length.
        """
        # Replace None with np.nan
        valid_data = np.array([x if x is not None else np.nan for x in data])
        valid_indices = ~np.isnan(valid_data)

        # Adjust window_length if necessary
        if np.sum(valid_indices) < window_length:
            window_length = max(3, (np.sum(valid_indices) // 2) * 2 - 1)

        # Apply smoothing only if valid data length meets window_length
        if np.sum(valid_indices) >= window_length:
            valid_data[valid_indices] = savgol_filter(
                valid_data[valid_indices],
                window_length,
                3,  # Polynomial order 3 for a smoother curve
                mode='interp'
            )

        return valid_data

    @staticmethod
    def plot_metrics(metrics):
        # Convert metrics into numpy arrays, replacing None with np.nan
        for key in metrics:
            metrics[key] = np.array([x if x is not None else np.nan for x in metrics[key]])

        epochs = metrics['epoch']

        # Create a figure
        plt.figure(figsize=(20, 15))

        # Define metric names and colors
        metric_names = ['loss', 'accuracy', 'dsc', 'specificity', 'precision', 'iou']
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

        for i, (metric, color) in enumerate(zip(metric_names, colors)):
            plt.subplot(2, 3, i + 1)
            valid_indices = ~np.isnan(metrics[metric])

            # Get smoothed data only
            smoothed_data = IoUCalculator.apply_smoothing(metrics[metric])[valid_indices]

            # Plot smoothed data with higher line width
            plt.plot(epochs[valid_indices], smoothed_data,
                     label=f' {metric.capitalize()}',
                     color=color,
                     linewidth=2)

            plt.title(metric.capitalize())
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()

            # Set y-axis limits based on data range
            y_min = np.nanmin(smoothed_data)
            y_max = np.nanmax(smoothed_data)
            y_range = y_max - y_min
            plt.ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

            # Add grid for better readability
            plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()

        # Save the plot
        plt.savefig('10deepsnake1_plot.png', dpi=300, bbox_inches='tight')
        print("Metrics plot saved as metrics_plot.png")
# Run the script
if __name__ == "__main__":
    IoUCalculator.main()