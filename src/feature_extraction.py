
import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2hsv


IMAGE_SIZE = (224, 224)


def extract_features(image):
    if isinstance(image, Image.Image):
        image = image.convert("RGB").resize(IMAGE_SIZE)
    else:
        image = Image.fromarray(np.asarray(image).astype(np.uint8))
        image = image.convert("RGB").resize(IMAGE_SIZE)

    rgb = np.asarray(image, dtype=np.float32) / 255.0
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0

    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )

    hsv = rgb2hsv(rgb)

    hist_features = []

    for channel in range(3):
        hist, _ = np.histogram(
            hsv[:, :, channel],
            bins=16,
            range=(0, 1),
            density=True
        )
        hist_features.extend(hist.tolist())

    features = np.concatenate([
        hog_features,
        np.asarray(hist_features, dtype=np.float32)
    ])

    return features.reshape(1, -1).astype(np.float32)
