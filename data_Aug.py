import cv2
import numpy as np
import os
import random
import logging
from PIL import Image, ImageFile
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DataAugmentation:

    def __init__(self):
        pass

    @staticmethod
    def openImage(image):
        logger.info(f"Opening image: {image}")
        return Image.open(image)

    @staticmethod
    def randomRotation(image, label, mode=Image.BICUBIC):
        # Generate eight images with random rotation angles between -20 and 20 degrees
        rotated_images = []
        for _ in range(8):
            random_angle = np.random.uniform(-10, 10)
            rotated_image = image.rotate(random_angle, mode)
            rotated_label = label.rotate(random_angle, Image.NEAREST)
            rotated_images.append((rotated_image, rotated_label))
        return rotated_images

    @staticmethod
    def horizontalFlip(image, label):
        # Horizontal flip
        logger.info("Applying horizontal flip")
        flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_label = label.transpose(Image.FLIP_LEFT_RIGHT)
        return flipped_image, flipped_label

    @staticmethod
    def saveImage(image, path):
        logger.info(f"Saving image to: {path}")
        image.save(path)


def imageOps(func_name, image, label, img_des_path, label_des_path, img_file_name, label_file_name, index):
    funcMap = {
        "randomRotation": DataAugmentation.randomRotation,
        "horizontalFlip": DataAugmentation.horizontalFlip,
    }

    if funcMap.get(func_name) is None:
        logger.error("%s does not exist", func_name)
        return -1

    if func_name == "randomRotation":
        rotated_images = funcMap[func_name](image, label)
        for _i, (new_image, new_label) in enumerate(rotated_images):
            DataAugmentation.saveImage(new_image, os.path.join(img_des_path, f"{index:04d}_{func_name}_{_i:02d}_" + img_file_name))
            DataAugmentation.saveImage(new_label, os.path.join(label_des_path, f"{index:04d}_{func_name}_{_i:02d}_" + label_file_name))
    else:
        # For horizontal flip, only one image is generated
        new_image, new_label = funcMap[func_name](image, label)
        DataAugmentation.saveImage(new_image, os.path.join(img_des_path, f"{index:04d}_{func_name}_" + img_file_name))
        DataAugmentation.saveImage(new_label, os.path.join(label_des_path, f"{index:04d}_{func_name}_" + label_file_name))


def process_image(args):
    img_name, label_name, img_path, label_path, new_img_path, new_label_path, index = args

    image = DataAugmentation.openImage(os.path.join(img_path, img_name))
    label = DataAugmentation.openImage(os.path.join(label_path, label_name))

    # Apply horizontal flip first
    imageOps("horizontalFlip", image, label, new_img_path, new_label_path, img_name, label_name, index)

    # Apply random rotation next
    imageOps("randomRotation", image, label, new_img_path, new_label_path, img_name, label_name, index)


def threadOPS(img_path, new_img_path, label_path, new_label_path):
    # Get sorted list of image and label filenames
    img_names = sorted(os.listdir(img_path))
    label_names = sorted(os.listdir(label_path))

    img_num = len(img_names)
    label_num = len(label_names)

    assert img_num == label_num, "圖片和標籤數量不一致"
    num = img_num

    args_list = [(img_names[i], label_names[i], img_path, label_path, new_img_path, new_label_path, i) for i in range(num)]

    with Pool(cpu_count()) as p:
        list(tqdm(p.imap(process_image, args_list), total=num, desc="Processing images"))


# Please modify the path
if __name__ == '__main__':
    # CHASEDB1
    threadOPS("C:/new data/train/images",  # set your path of training images
              "C:/new data/train/aug/images",
              "C:/new data/train/label",  # set your path of training labels
              "C:/new data/train/aug/labels")
