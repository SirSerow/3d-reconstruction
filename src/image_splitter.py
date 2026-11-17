import os 

import cv2
import numpy as np


class ImageSplitter:
    """
    Splits composite images (2x3 grid) into individual images.
    """

    def __init__(self, output_dir=".output"):

        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)


    def split_and_save(self, folder_path):

        index = 0

        # create cam subforlders
        for i in range(3):
            for j in range(2):
                if i == 2 and j == 1:
                    continue
                cam_folder = os.path.join(self.output_dir, f"cam_{i}_{j}")
                if not os.path.exists(cam_folder):
                    os.makedirs(cam_folder)

        for filename in os.listdir(folder_path):

            if filename.endswith((".png", ".jpg", ".jpeg")):

                image_path = os.path.join(folder_path, filename)

                image = cv2.imread(image_path)

                if image is None:
                    continue

                index = self._process_frame(image, index)

            elif filename.endswith((".mp4", ".avi", ".mov")):

                video_path = os.path.join(folder_path, filename)

                capture = cv2.VideoCapture(video_path)

                try:
                    while True:
                        success, frame = capture.read()

                        if not success:
                            break

                        index = self._process_frame(frame, index)
                        
                finally:
                    capture.release()


    def _process_frame(self, image, index):

        height, width, _ = image.shape

        sub_height = height // 3

        sub_width = width // 2

        for i in range(3):

            for j in range(2):

                if i == 2 and j == 1:
                    # skip empty sub-image in the bottom-right corner
                    continue

                sub_image = image[i*sub_height:(i+1)*sub_height, j*sub_width:(j+1)*sub_width]

                output_path = os.path.join(self.output_dir, f"cam_{i}_{j}", f"{index}.png")

                cv2.imwrite(output_path, sub_image)

        index += 1

        return index


if __name__ == "__main__":
    splitter = ImageSplitter()
    splitter.split_and_save(".input")
