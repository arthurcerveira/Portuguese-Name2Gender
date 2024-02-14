import os

# Define environment variables to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from pathlib import Path
import random
import tensorflow as tf
import numpy as np

tf.get_logger().setLevel('ERROR')

# Set seeds for reproducibility
SEED = 1907

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

CURRENT_DIR = Path(__file__).resolve().parent
MODEL_DIR = CURRENT_DIR / "model"
DATA_DIR = CURRENT_DIR / "data"
