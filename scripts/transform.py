# ====================================================
# Transforms
# ====================================================

import albumentations as A
from config import CFG

def get_transforms(*, data):
    if data == 'train' or data == 'valid':
        return A.Compose([
            A.Resize(CFG.size, CFG.size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])