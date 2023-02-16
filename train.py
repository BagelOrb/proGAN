""" Training of ProGAN using WGAN-GP loss"""
import time

import GPUtil
import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import (
    gradient_penalty,
    plot_to_tensorboard,
    save_checkpoint,
    load_checkpoint,
    generate_examples,
)
from model import Discriminator, Generator
from math import log2
from tqdm import tqdm

import psutil

import config

# Gives additional performance benefits
torch.backends.cudnn.benchmarks = True


def get_loader(image_size, folder):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(
                [0.5 for _ in range(config.CHANNELS_IMG)],
                [0.5 for _ in range(config.CHANNELS_IMG)],
            ),
        ]
    )
    batch_size = config.BATCH_SIZES[int(log2(image_size / 4))]
    dataset = datasets.ImageFolder(root=folder, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    return loader, dataset


def train_fn(
        critic,
        gen,
        loader,
        dataset,
        val_loader,
        val_dataset,
        step,
        alpha,
        opt_critic,
        opt_gen,
        tensorboard_step,
        writer,
        scaler_gen,
        scaler_critic,
):
    loop = tqdm(loader, leave=True)
    time_cooled = 0
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(config.DEVICE)
        cur_batch_size = real.shape[0]

        # Train Critic: max E[critic(real)] - E[critic(fake)] <-> min -E[critic(real)] + E[critic(fake)]
        # which is equivalent to minimizing the negative of the expression
        noise = torch.randn(cur_batch_size, config.Z_DIM, 1, 1).to(config.DEVICE)

        with torch.cuda.amp.autocast():
            fake = gen(noise, alpha, step)
            critic_real = critic(real, alpha, step)
            critic_fake = critic(fake.detach(), alpha, step)
            gp = gradient_penalty(critic, real, fake, alpha, step, device=config.DEVICE)
            loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + config.LAMBDA_GP * gp
                    + (0.001 * torch.mean(critic_real ** 2))
            )

        opt_critic.zero_grad()
        scaler_critic.scale(loss_critic).backward()
        scaler_critic.step(opt_critic)
        scaler_critic.update()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        with torch.cuda.amp.autocast():
            gen_fake = critic(fake, alpha, step)
            loss_gen = -torch.mean(gen_fake)

        opt_gen.zero_grad()
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()

        # Update alpha and ensure less than 1
        alpha += cur_batch_size / (
                (config.PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset)
        )
        alpha = min(alpha, 1)

        if batch_idx % 240 == 0:
            with torch.no_grad():
                fixed_fakes = gen(config.FIXED_NOISE, alpha, step) * 0.5 + 0.5
            # TODO: add batch accuracy
            # TODO: add validation accuracy
            batch_accuracies = []
            print("Evaluating accuracy...")
            for (val_real, _) in val_loader:
                cur_val_batch_size = val_real.shape[0]
                noise = torch.randn(cur_val_batch_size, config.Z_DIM, 1, 1).to(config.DEVICE)
                # with torch.cuda.amp.autocast(): # TODO idk if this should be commented out or not
                with torch.no_grad():
                    fake = gen(noise, alpha, step)
                    critic_real = critic(real, alpha, step)
                    critic_fake = critic(fake.detach(), alpha, step)
                    batch_accuracies.append((
                                                    ((critic_real > .5).type(torch.float)).mean()
                                                    + ((critic_fake < .5).type(torch.float)).mean()
                                            ) / 2)  # / 2 because we add mean correct real and mean correct fake
            accuracy = torch.Tensor(batch_accuracies).mean()
            print("...done")

            plot_to_tensorboard(
                writer,
                loss_critic.item(),
                loss_gen.item(),
                accuracy,
                alpha,
                real.detach(),
                fixed_fakes.detach(),
                tensorboard_step,
            )
            tensorboard_step += 1

        # ensure safe temperatures
        safety_factor = .95
        cooling_factor = .80
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
            time_cooled += time.time() - time_before_cooling
            # print(f"Cooled for {time.time() - time_before_cooling: .1f} seconds because of {cooling_causes}.")

        loop.set_postfix(
            gp=gp.item(),
            loss_critic=loss_critic.item(),
        )

    print(f"Cooled for {time_cooled: .1f} seconds.")
    return tensorboard_step, alpha


def main():
    # initialize gen and disc, note: discriminator should be called critic,
    # according to WGAN paper (since it no longer outputs between [0, 1])
    # but really who cares..
    gen = Generator(
        config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG
    ).to(config.DEVICE)
    critic = Discriminator(
        config.IN_CHANNELS, img_channels=config.CHANNELS_IMG
    ).to(config.DEVICE)

    # initialize optimizers and scalers for FP16 training
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic = optim.Adam(
        critic.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99)
    )
    scaler_critic = torch.cuda.amp.GradScaler()
    scaler_gen = torch.cuda.amp.GradScaler()

    # for tensorboard plotting
    writer = SummaryWriter(f"logs/gan")

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC, critic, opt_critic, config.LEARNING_RATE,
        )

    gen.train()
    critic.train()

    tensorboard_step = 0
    # start at step that corresponds to img size that we set in config
    n_blocks = int(log2(config.START_TRAIN_AT_IMG_SIZE / 4))
    img_size = 4 * 2 ** n_blocks
    for n_epochs in config.PROGRESSIVE_EPOCHS[n_blocks:]:
        alpha = 1e-5  # start with very low alpha
        loader, dataset = get_loader(4 * 2 ** n_blocks, config.DATASET)  # 4->0, 8->1, 16->2, 32->3, 64 -> 4
        val_loader, val_dataset = get_loader(img_size,
                                             config.VALIDATION_DATASET)  # 4->0, 8->1, 16->2, 32->3, 64 -> 4
        print(f"Current image size: {img_size}")

        for epoch in range(n_epochs):
            print(f"Epoch [{epoch + 1}/{n_epochs}]")
            tensorboard_step, alpha = train_fn(
                critic,
                gen,
                loader,
                dataset,
                val_loader,
                val_dataset,
                n_blocks,
                alpha,
                opt_critic,
                opt_gen,
                tensorboard_step,
                writer,
                scaler_gen,
                scaler_critic,
            )

            if config.SAVE_MODEL:
                save_checkpoint(gen, opt_gen, img_size, filename_prefix=config.CHECKPOINT_GEN)
                save_checkpoint(critic, opt_critic, img_size, filename_prefix=config.CHECKPOINT_CRITIC)

        n_blocks += 1  # progress to the next img size


if __name__ == "__main__":
    main()
