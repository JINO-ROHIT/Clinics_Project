# ====================================================
# Imports
# ====================================================
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import seaborn as sns
import random

meta_data = pd.read_csv('../input/csv-file/meta_data.csv')

corrected_labels = {broken_labels : i for i, broken_labels in enumerate(meta_data.target.unique())}
meta_data['corrected_labels'] = meta_data['target'].map(corrected_labels)

def get_path(image_id):
    return "../input/roi-images/roi_resized/{}".format(image_id)
    
meta_data['image_path'] = meta_data['image'].apply(get_path)



