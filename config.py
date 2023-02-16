import torch
from math import log2

START_TRAIN_AT_IMG_SIZE = 32
DATASET = 'celeba_hq/train'  # FIXME: only use training data
VALIDATION_DATASET = 'celeba_hq/val'  # FIXME: only use training data
CHECKPOINT_GEN = "generator"
CHECKPOINT_CRITIC = "critic"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = True
LOAD_MODEL = True
LEARNING_RATE = 1e-3
#BATCH_SIZES = [32, 32, 32, 16, 16, 16, 16, 8, 4]  # change depending on vram
BATCH_SIZES = [64, 64, 64, 64, 32, 32, 32, 16, 8]  # change depending on vram
CHANNELS_IMG = 3
Z_DIM = 256  # should be 512 in original paper
IN_CHANNELS = 256  # should be 512 in original paper
CRITIC_ITERATIONS = 1
LAMBDA_GP = 10

# in paper they did 800k img per epoch; TODO: we're doing 30 ?!
PROGRESSIVE_EPOCHS = [30] * len(BATCH_SIZES)
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 8
