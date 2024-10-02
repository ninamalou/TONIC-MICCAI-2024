import pandas as pd
from PIL import Image
import numpy as np
import os

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from PIL import ImageOps



def load_data(csv_file_path, image_folder_path, num_samples=None, target_size=(224, 224)):
    """_summary_

    Args:
        csv_file_path (str): path to csv with IDs, labels and patient information
        image_folder_path (str): path to folder that contains the images
        num_samples (int, optional): number of images to load. Defaults to None.
        target_size (tuple, optional): resize the images to this target size. Defaults to (224, 224).

    Returns:
        images: array with images
        y: array with labels
    """
    # Load the CSV file
    print('busy')
    csv = pd.read_csv(csv_file_path)

    images = []
    labels = []
    cases = []

    # Iterate over the rows in the CSV
    for index, row in csv.iterrows():
        if num_samples is not None and index >= num_samples:
            break
        print(index, 'loaded')
        image = row['image']
        label = row['label']
        case = row['case']
        
        # Load the images
        image_path = os.path.join(image_folder_path, image)
        image = Image.open(image_path)
        image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)  # Resize image

        image_array = np.array(image)

        # Rearrange the channels (from HWC to CHW) as needed for resnet input
        image_array = np.transpose(image_array, (2, 0, 1))
        
        # Append the image and label to the lists
        images.append(image_array)
        labels.append(label)
        cases.append(case)

    # Convert lists to numpy arrays for feeding into Resnet
    images = np.array(images)
    y = np.array(labels)
    cases = np.array(cases)


    # Display the shape of the data
    print(f'Loaded {images.shape[0]} images with shape {images.shape[1:]}')
    print(f'Labels shape: {y.shape}')

    return images, y, cases