import os
from PIL import Image
import numpy as np

class DataLoader:

    def __init__(self):
        pass

    def load_data_s2(self, base_dir = '/kaggle/input/sentinel12-image-pairs-segregated-by-terrain/v_2/', path_base="s2"):
        image_data = []

        for root, dirs, files in os.walk(base_dir):
            # Only include files if their immediate directory is named 's2'
            if os.path.basename(root).lower() == path_base:
                for file in files:
                    if file.lower().endswith('.png'):
                        image_path = os.path.join(root, file)
                        try:
                            with Image.open(image_path) as img:
                                img_array = np.array(img)
                            label = os.path.relpath(root, base_dir)
                            image_data.append({'image': img_array, 'label': label})
                        except Exception as e:
                            print(f"Error loading image {image_path}: {e}")

        return image_data