import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def bimodal_histogram(img, hist, v, w, filename, histogram_folder, idx):
    plt.figure(figsize=(10, 6))

    # Plot the histogram
    plt.hist(img.ravel(), 256, [0, 256], color='gray', alpha=0.75)

    # Find the maximum value for setting y-axis limit
    max_freq = np.max(hist)

    # Set axis limits
    plt.xlim(0, 256)
    plt.ylim(0, max_freq * 1.1)

    # Add labels and title
    plt.xlabel('Pixel Intensity', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title(f'Bimodal Histogram for {filename}', fontsize=16)

    # Mark 'v' and 'w' points on x-axis
    plt.xticks(list(plt.xticks()[0]) + [v, w],
               labels=[str(int(x)) for x in list(plt.xticks()[0])] + [f'v ({v})', f'w ({w})'])

    # Mark 'v' point on the histogram
    plt.plot(v, hist[v], 'ro', markersize=10)
    plt.text(v, hist[v], 'v', fontsize=12, verticalalignment='bottom', horizontalalignment='right')

    # Mark 'w' point on the histogram with a red dot
    plt.plot(w, hist[w], 'ro', markersize=10)
    plt.text(w, hist[w], 'w', fontsize=12, verticalalignment='bottom', horizontalalignment='left')

    # Adjust layout and remove top and right spines
    plt.tight_layout()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Save the figure
    plt.savefig(os.path.join(histogram_folder, f'{idx}histogram.png'), dpi=300, bbox_inches='tight')
    plt.close()


def process_images(input_folder, binary_mask_folder, median_filtered_folder, connected_components_folder,
                   filtered_components_folder, histogram_folder):
    # Create output folders
    output_folders = [binary_mask_folder, median_filtered_folder, connected_components_folder,
                      filtered_components_folder, histogram_folder]
    for folder in output_folders:
        create_folder(folder)

    files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

    for idx, filename in enumerate(tqdm(files, desc="Processing images"), start=1):
        input_image_path = os.path.join(input_folder, filename)

        # Read the image in grayscale
        img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error reading image file: {input_image_path}")
            continue

        # Calculate histogram
        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()

        # Find the first peak (v) and second peak (w)
        smoothed_hist = np.convolve(hist, np.ones(5) / 5, mode='same')  # Smooth histogram
        peaks = np.where((smoothed_hist[1:-1] > smoothed_hist[:-2]) &
                         (smoothed_hist[1:-1] > smoothed_hist[2:]))[0] + 1
        if len(peaks) >= 2:
            v, w = peaks[:2]
        else:
            v, w = np.argmax(smoothed_hist), len(smoothed_hist) - 1

        # Generate and save the bimodal histogram
        bimodal_histogram(img, hist, v, w, filename, histogram_folder, idx)

        # Process the image for intensity values in the range [w-5, w+25]
        img_scaled = img.astype(float)

        def apply_contrast_stretching(image, clip_limit=2.4, tile_grid_size=( 9, 9 )):
            # Use CLAHE (Contrast Limited Adaptive Histogram Equalization) for dynamic contrast stretching
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            return clahe.apply(image)

        def adaptive_multi_otsu(image, total_pixels, output_dir, input_path):
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

            # Save the final mask
            output_filename = os.path.join(output_dir,
                                           f"{os.path.splitext(os.path.basename(input_path))[0]}_adaptive.png")
            cv2.imwrite(output_filename, final_mask)

            return final_mask


        # Example usage
        total_pixels = img.size
        binary_img = adaptive_multi_otsu(img, total_pixels, binary_mask_folder, input_image_path)

        processed_img = np.where((img_scaled >= w - 5) & (img_scaled <= w + 25), img_scaled, 0)

        # Apply median filter
        median_filtered = cv2.medianBlur(binary_img, 5)
        cv2.imwrite(os.path.join(median_filtered_folder, filename), median_filtered)

        # Connected component analysis using 4-connectivity
        num_labels, labeled_mask = cv2.connectedComponents(median_filtered, connectivity=4)

        # Visualize connected components
        color_labeled = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        for label in range(1, num_labels):
            color_labeled[labeled_mask == label] = np.random.randint(0, 255, size=3)
        cv2.imwrite(os.path.join(connected_components_folder, filename), color_labeled)

        # Filter connected components by size, orientation, position, and width/height ratio
        valid_components = []
        filtered_image = np.zeros_like(img)
        center_x = img.shape[1] // 2  # Calculate the center of the image along the x-axis

        for label in range(1, num_labels):
            component_mask = (labeled_mask == label).astype(np.uint8)

            # Calculate size (area in pixels)
            area = np.sum(component_mask)

            # Calculate orientation using image moments
            moments = cv2.moments(component_mask)
            if moments['mu02'] != 0:
                angle = 0.5 * np.arctan2(2 * moments['mu11'], moments['mu20'] - moments['mu02'])
                angle_degrees = np.degrees(angle)
            else:
                angle_degrees = 0  # If the component is perfectly aligned

            # Find bounding box and calculate width and height
            x, y, w, h = cv2.boundingRect(component_mask)  # w is width, h is height

            # Calculate horizontal position (centroid)
            if moments['m00'] != 0:  # Avoid division by zero for centroid calculation
                centroid_x = int(moments['m10'] / moments['m00'])

                # Apply filtering conditions with specific width and height constraints
                if (area > 300 and abs(angle_degrees) <= 25 and
                        abs(centroid_x - center_x) <= 40 and
                        30 <= w <= 300 and 50 <= h <= 250):
                    valid_components.append((label, area))
                    filtered_image[labeled_mask == label] = 255  # Keep this component in the output mask

        # Save the image with valid components
        cv2.imwrite(os.path.join(filtered_components_folder, filename), filtered_image)

# Input and output folder paths
input_folder = 'C:/new data/valid/images'
binary_mask_folder = 'C:/new data/valid/binary_mask'
median_filtered_folder = 'C:/new data/valid/median_filtered'
connected_components_folder = 'C:/new data/valid/connected_components'
filtered_components_folder = 'C:/new data/valid/labels'
histogram_folder = 'C:/new data/valid/histogram'
# Process the images
if __name__ == '__main__':
    process_images(input_folder, binary_mask_folder, median_filtered_folder,
                   connected_components_folder, filtered_components_folder, histogram_folder)

