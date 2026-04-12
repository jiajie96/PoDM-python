"""FID evaluation: generate samples and save for external FID computation."""

import argparse
import os

import numpy as np
import torch
import tqdm
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
    parser = argparse.ArgumentParser(description="Generate samples for FID evaluation")
    parser.add_argument("--config", type=str, default="biked_256.yml", help="YAML config file")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Checkpoint .pth file")
    parser.add_argument("--output_dir", type=str, default="fid_samples", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=1000, help="Total number of samples")
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

    # Load model
    model = Model(config)
    states = torch.load(args.ckpt_path, map_location=device)
    model = model.to(device)
    model = torch.nn.DataParallel(model, dataparallel)
    model.load_state_dict(states[0], strict=True)
    print(f"Loaded checkpoint (step {states[3]})")
    model.eval()

    # Generate samples
    batch_size = config.sampling.batch_size
    n_rounds = (args.num_samples + batch_size - 1) // batch_size
    img_id = 0

    with torch.no_grad():
        for _ in tqdm.tqdm(range(n_rounds), desc="Generating FID samples"):
            n = min(batch_size, args.num_samples - img_id)
            x = torch.randn(n, config.data.channels, config.data.image_size, config.data.image_size, device=device)

            x = edm_sampler(
                x, args.timesteps, model,
                sigma_min=sigma_min, sigma_max=sigma_max, rho=args.rho,
            )

            x = [(y + 1.0) / 2.0 for y in x]
            for i in range(n):
                tvu.save_image(x[i], os.path.join(args.output_dir, f"{img_id}.png"))
                img_id += 1

    print(f"Generated {img_id} samples in {args.output_dir}")


if __name__ == "__main__":
    main()
