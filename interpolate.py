"""SLERP latent-space interpolation between two test images."""

import argparse
import os

import numpy as np
import torch
import torchvision.utils as tvu
import yaml

from diffusion_model import Model


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def slerp(z1, z2, alpha):
    theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
    return (
        torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
        + torch.sin(alpha * theta) / torch.sin(theta) * z2
    )


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


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Interpolate between two images with PoDM")
    parser.add_argument("--config", type=str, default="biked_256.yml", help="YAML config file")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Checkpoint .pth file")
    parser.add_argument("--test_path", type=str, required=True, help="Test .npy file")
    parser.add_argument("--output_dir", type=str, default="interpolation_output", help="Output directory")
    parser.add_argument("--idx_a", type=int, default=0, help="Index of first image")
    parser.add_argument("--idx_b", type=int, default=7, help="Index of second image")
    parser.add_argument("--num_steps", type=int, default=11, help="Number of interpolation steps")
    parser.add_argument("--timesteps", type=int, default=18, help="Number of sampling timesteps")
    parser.add_argument("--rho", type=float, default=7, help="EDM rho parameter")
    parser.add_argument("--sigma_start", type=float, default=14.4, help="Sigma start")
    parser.add_argument("--sigma_end", type=float, default=1.64, help="Sigma end")
    parser.add_argument("--gpu_ids", type=str, default="0", help="Comma-separated GPU IDs")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    return parser


def main():
    args = build_arg_parser().parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    dataparallel = list(range(len(args.gpu_ids.split(","))))

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config["tb_logger"] = None
    config = dict2namespace(config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    config.device = device

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    sigma_min = args.sigma_end**2 / args.sigma_start
    sigma_max = args.sigma_start**2 / args.sigma_end

    # Load test images
    test_images = np.load(args.test_path, mmap_mode="r")
    test_images = test_images / 0.5 - 1
    test_images_tensor = torch.tensor(test_images).permute((0, 3, 1, 2)).to(device)

    x1 = test_images_tensor[args.idx_a : args.idx_a + 1]
    x2 = test_images_tensor[args.idx_b : args.idx_b + 1]

    # Load model
    model = Model(config)
    states = torch.load(args.ckpt_path, map_location=device)
    model = model.to(device)
    model = torch.nn.DataParallel(model, dataparallel)
    model.load_state_dict(states[0], strict=True)
    print(f"Loaded checkpoint (step {states[3]})")
    model.eval()

    # Add noise to get latent codes
    step_indices = torch.arange(args.timesteps, device=device)
    t_steps = (
        sigma_max ** (1 / args.rho)
        + step_indices / (args.timesteps - 1) * (sigma_min ** (1 / args.rho) - sigma_max ** (1 / args.rho))
    ) ** args.rho
    t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])])
    t_steps_flip = t_steps.flip(0)

    z1, z2 = x1.clone(), x2.clone()
    for t_cur, t_next in zip(t_steps_flip[:-1], t_steps_flip[1:]):
        z1 = z1 + (t_next - t_cur) * torch.randn_like(z1)
        z2 = z2 + (t_next - t_cur) * torch.randn_like(z2)

    z1 = z1 / sigma_max
    z2 = z2 / sigma_max

    # SLERP interpolation
    alphas = torch.linspace(0.0, 1.0, args.num_steps).to(device)
    z_interp = []
    for a in alphas:
        z_interp.append(slerp(z1, z2, a))
    x = torch.cat(z_interp, dim=0)

    # Denoise
    xs = []
    with torch.no_grad():
        for i in range(0, x.size(0), 8):
            xs.append(edm_sampler(
                x[i : i + 8], args.timesteps, model,
                sigma_min=sigma_min, sigma_max=sigma_max, rho=args.rho,
            ))
    results = [(y + 1.0) / 2.0 for y in torch.cat(xs, dim=0)]

    # Save source images
    tvu.save_image((x1 + 1.0) / 2.0, os.path.join(args.output_dir, "source_1.png"))
    tvu.save_image((x2 + 1.0) / 2.0, os.path.join(args.output_dir, "source_2.png"))

    # Save interpolated images
    for i, img in enumerate(results):
        tvu.save_image(img, os.path.join(args.output_dir, f"interp_{i:02d}.png"))

    print(f"Saved {len(results)} interpolation steps to {args.output_dir}")


if __name__ == "__main__":
    main()
