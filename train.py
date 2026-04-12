"""Training entry point for PoDM diffusion model."""

import argparse
import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
import torchvision.utils as tvu
import yaml


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, torch.nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, torch.nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1.0 - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, torch.nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, torch.nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = torch.nn.DataParallel(module_copy, inner_module.args.dataparallel)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_optimizer(config, parameters):
    if config.optim.optimizer == "Adam":
        return optim.Adam(
            parameters,
            lr=config.optim.lr,
            weight_decay=config.optim.weight_decay,
            betas=(config.optim.beta1, 0.999),
            amsgrad=config.optim.amsgrad,
            eps=config.optim.eps,
        )
    elif config.optim.optimizer == "RMSProp":
        return optim.RMSprop(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == "SGD":
        return optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError(f"Optimizer {config.optim.optimizer} not understood.")


def noise_estimation_loss(net, images, sigma_start, sigma_end):
    rnd_normal = torch.randn([images.shape[0]], device=images.device)
    P_mean = np.log(sigma_start * sigma_end) / 2
    P_std = np.log(sigma_start / sigma_end) / 2
    sigma = (rnd_normal * P_std + P_mean).exp()
    reshaped_sigma = sigma.reshape(images.shape[0], 1, 1, 1)
    y = images
    n = torch.randn_like(y) * reshaped_sigma
    D_yn = net(y + n, sigma)
    loss = (D_yn - y) ** 2
    return loss.mean()


def edm_sampler(
    latents,
    num_steps,
    net,
    randn_like=torch.randn_like,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
):
    step_indices = torch.arange(num_steps, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])])

    x_next = [latents * t_steps[0]]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        t_cur = torch.ones(x_next[0].shape[0], device=t_cur.device) * t_cur
        t_next = torch.ones(x_next[0].shape[0], device=t_next.device) * t_next
        x_cur = x_next[-1]

        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur[0] <= S_max else 0
        t_hat = torch.as_tensor(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat[0] ** 2 - t_cur[0] ** 2).sqrt() * S_noise * randn_like(x_cur)

        denoised = net(x_hat, t_hat)
        d_cur = (x_hat - denoised) / t_hat[0]
        x_next_ = x_hat + (t_next[0] - t_hat[0]) * d_cur

        if i < num_steps - 1:
            denoised = net(x_next_, t_next)
            d_prime = (x_next_ - denoised) / t_next[0]
            x_next_ = x_hat + (t_next[0] - t_hat[0]) * (0.5 * d_cur + 0.5 * d_prime)

        x_next.append(x_next_)

    return x_next[-1].cpu()


def train(args, config, device, train_images_tensor, val_images_tensor):
    from diffusion_model import Model

    dataset = train_images_tensor
    val_dataset = val_images_tensor[: config.training.batch_size]
    train_loader = data.DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
    )

    model = Model(config)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters in the model: {total_params}")

    model = model.to(device)
    model = torch.nn.DataParallel(model, args.dataparallel)

    optimizer = get_optimizer(config, model.parameters())

    if config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(model)
    else:
        ema_helper = None

    start_epoch, step = 0, 0
    if args.resume_ckpt is not None:
        states = torch.load(args.resume_ckpt, map_location=device)
        model.load_state_dict(states[0])
        states[1]["param_groups"][0]["eps"] = config.optim.eps
        states[1]["param_groups"][0]["lr"] = config.optim.lr
        optimizer.load_state_dict(states[1])
        start_epoch = states[2]
        step = states[3]
        if config.model.ema:
            ema_helper.load_state_dict(states[4])
        print(f"Resumed from step {step}")

    loss_his = []
    val_loss_his = []
    val_loss_steps = []

    sigma_min = args.sigma_end**2 / args.sigma_start
    sigma_max = args.sigma_start**2 / args.sigma_end

    for epoch in range(start_epoch, config.training.n_epochs):
        data_start = time.time()
        data_time = 0
        for i, x in enumerate(train_loader):
            data_time += time.time() - data_start
            model.train()
            step += 1
            x = x.to(device)
            loss = noise_estimation_loss(model, x, args.sigma_start, args.sigma_end)

            print(f"step: {step}, loss: {loss.item()}, data time: {data_time / (i + 1)}")

            optimizer.zero_grad()
            loss.backward()

            try:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
            except Exception:
                pass

            optimizer.step()

            if config.model.ema:
                ema_helper.update(model)

            if step % config.training.snapshot_freq == 0 or step == 1:
                states = [model.state_dict(), optimizer.state_dict(), epoch, step]
                if config.model.ema:
                    states.append(ema_helper.state_dict())
                torch.save(states, os.path.join(args.log_dir, f"ckpt_{step}.pth"))
                torch.save(states, os.path.join(args.log_dir, "ckpt.pth"))

            if step % config.training.validation_freq == 0:
                model.eval()
                with torch.no_grad():
                    n = config.sampling.batch_size
                    x_sample = torch.randn(n, config.data.channels, config.data.image_size, config.data.image_size, device=device)
                    x_sample = edm_sampler(
                        x_sample, args.timesteps, model,
                        sigma_min=sigma_min, sigma_max=sigma_max, rho=args.rho,
                    )
                    x_sample = [(y + 1.0) / 2.0 for y in x_sample]
                    plt.figure(figsize=(16, 8))
                    for j in range(min(8, len(x_sample))):
                        plt.subplot(2, 4, j + 1)
                        img = x_sample[j].permute((1, 2, 0)).numpy()
                        plt.imshow(np.asarray(img), cmap="gray")
                        plt.axis("off")
                    plt.savefig(os.path.join(args.log_dir, f"sample_{step}.png"))
                    plt.close()

            data_start = time.time()

        val_dataset_gpu = val_dataset.to(device)
        val_loss = noise_estimation_loss(model, val_dataset_gpu, args.sigma_start, args.sigma_end)
        val_loss_his.append(val_loss.item())
        val_loss_steps.append(epoch)

        loss_his.append(loss.item())
        plt.figure()
        plt.plot(range(len(loss_his)), loss_his, label="loss")
        plt.plot(val_loss_steps, val_loss_his, label="val_loss")
        plt.legend()
        plt.savefig(os.path.join(args.log_dir, "loss_his.png"))
        plt.close()
        np.save(os.path.join(args.log_dir, "loss_his.npy"), loss_his)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train PoDM diffusion model")
    parser.add_argument("--config", type=str, default="biked_256.yml", help="YAML config file")
    parser.add_argument("--train_path", type=str, required=True, help="Training .npy file")
    parser.add_argument("--val_path", type=str, required=True, help="Validation .npy file")
    parser.add_argument("--test_path", type=str, default=None, help="Test .npy file")
    parser.add_argument("--log_dir", type=str, default="result_diffusion_model", help="Output directory")
    parser.add_argument("--gpu_ids", type=str, default="0", help="Comma-separated GPU IDs")
    parser.add_argument("--resume_ckpt", type=str, default=None, help="Checkpoint .pth to resume from")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--timesteps", type=int, default=18, help="Number of sampling timesteps")
    parser.add_argument("--rho", type=float, default=7, help="EDM rho parameter")
    parser.add_argument("--sigma_start", type=float, default=14.4, help="Sigma start")
    parser.add_argument("--sigma_end", type=float, default=1.64, help="Sigma end")
    return parser


def main():
    args = build_arg_parser().parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    args.dataparallel = list(range(len(args.gpu_ids.split(","))))

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config["tb_logger"] = None
    config = dict2namespace(config)

    device_str = f"cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")
    config.device = device

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    os.makedirs(args.log_dir, exist_ok=True)

    with open(os.path.join(args.log_dir, "config.yml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Load data
    train_images = np.load(args.train_path, mmap_mode="r")
    val_images = np.load(args.val_path, mmap_mode="r")
    train_images = train_images / 0.5 - 1
    val_images = val_images / 0.5 - 1
    print(f"train_images shape: {train_images.shape}, range: [{train_images.min():.2f}, {train_images.max():.2f}]")

    train_images_tensor = torch.tensor(train_images).permute((0, 3, 1, 2))
    val_images_tensor = torch.tensor(val_images).permute((0, 3, 1, 2))
    del train_images

    train(args, config, device, train_images_tensor, val_images_tensor)


if __name__ == "__main__":
    main()
