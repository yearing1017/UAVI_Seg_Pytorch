import os
from PIL import Image
import cv2
images = os.listdir('data/dataset2/images_prepped_train')


for image_name in images:
    if not os.path.exists('data/dataset2/anno_prepped_train/' + image_name):
        print(image_name)

print("over")
