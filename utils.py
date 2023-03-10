import torch
import random
import numpy as np
import os
import torchvision
import torch.nn as nn
import config
from torchvision.utils import save_image
from scipy.stats import truncnorm

import psutil  # for CPU temp
import GPUtil  # for temp of graphics card
import time

# Print losses occasionally and print to tensorboard
def plot_to_tensorboard(
        writer, loss_critic, loss_gen, alpha, real, fake, tensorboard_step
):
    writer.add_scalar("Loss Critic", loss_critic, global_step=tensorboard_step)
    writer.add_scalar("Loss Generator", loss_gen, global_step=tensorboard_step)
    writer.add_scalar("alpha", alpha, global_step=tensorboard_step)

    with torch.no_grad():
        # take out (up to) 8 examples to plot
        img_grid_real = torchvision.utils.make_grid(real[:8], normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake[:8], normalize=True)
        writer.add_image("Real", img_grid_real, global_step=tensorboard_step)
        writer.add_image("Fake", img_grid_fake, global_step=tensorboard_step)


def cooldown():
    safety_factor = .95
    cooling_factor = .90
    max_gpu_temp = 93
    safe = True
    cooling_causes = []
    for temp_sensor in psutil.sensors_temperatures()['coretemp']:
        if temp_sensor.current > safety_factor * temp_sensor.high:
            safe = False
            cooling_causes.append(temp_sensor.label)
    for gpu in GPUtil.getGPUs():
        if gpu.temperature > safety_factor * max_gpu_temp:
            safe = False
            cooling_causes.append('GPU')
    # Cool to 80% of safe temperatures
    if not safe:
        time_before_cooling = time.time()
        while not safe:
            safe = True
            for temp_sensor in psutil.sensors_temperatures()['coretemp']:
                if temp_sensor.current > cooling_factor * temp_sensor.high:
                    safe = False
            for gpu in GPUtil.getGPUs():
                if gpu.temperature > cooling_factor * max_gpu_temp:
                    safe = False
            time.sleep(.1)
        return time.time() - time_before_cooling
    return 0


def gradient_penalty(critic, real, fake, alpha, train_step, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, alpha, train_step)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    return torch.mean((gradient_norm - 1) ** 2)


def save_checkpoint(model, optimizer, img_size, filename_prefix="my_checkpoint"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename_prefix + str(img_size) + ".pth")


def load_checkpoint(filename_prefix, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(filename_prefix + str(config.START_TRAIN_AT_IMG_SIZE // (1+config.START_ONE_BELOW)) + ".pth", map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_examples(gen, steps, truncation=0.7, n=100):
    """
    Tried using truncation trick here but not sure it actually helped anything, you can
    remove it if you like and just sample from torch.randn
    """
    gen.eval()
    alpha = 1.0
    for i in range(n):
        with torch.no_grad():
            noise = torch.tensor(truncnorm.rvs(-truncation, truncation, size=(1, config.Z_DIM, 1, 1)),
                                 device=config.DEVICE, dtype=torch.float32)
            img = gen(noise, alpha, steps)
            save_image(img * 0.5 + 0.5, f"saved_examples/img_{i}.png")
    gen.train()
