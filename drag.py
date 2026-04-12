"""DragFDM: interactive latent-space manipulation via diffusion feature maps."""

import argparse
import copy
import os

import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
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


def forward_unet(latent, t_steps, start_step, end_step, model, timesteps,
                 stochastic=False, feature_map=True):
    if stochastic:
        S_churn, S_min, S_max, S_noise = 5, 0, float("inf"), 1.003
    else:
        S_churn, S_min, S_max, S_noise = 0, 0, float("inf"), 1

    t_steps_n2f = t_steps[start_step - 1 : end_step + 1]
    x_next = [latent]
    for i, (t_cur, t_next) in enumerate(zip(t_steps_n2f[:-1], t_steps_n2f[1:])):
        t_cur = torch.ones(x_next[0].shape[0], device=t_cur.device) * t_cur
        t_next = torch.ones(x_next[0].shape[0], device=t_next.device) * t_next
        x_cur = x_next[-1]

        gamma = min(S_churn / timesteps, np.sqrt(2) - 1) if S_min <= t_cur[0] <= S_max else 0
        t_hat = torch.as_tensor(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat[0] ** 2 - t_cur[0] ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

        for param in model.parameters():
            param.requires_grad = False
        denoised, features = model(x_hat, t_hat, return_intermediates=True)
        d_cur = (x_hat - denoised) / t_hat[0]
        x_next_ = x_hat + (t_next[0] - t_hat[0]) * d_cur

        if end_step != timesteps or i < end_step - start_step - 1:
            denoised = model(x_next_, t_next)
            d_prime = (x_next_ - denoised) / t_next[0]
            x_next_ = x_hat + (t_next[0] - t_hat[0]) * (0.5 * d_cur + 0.5 * d_prime)
        x_next.append(x_next_)

    if feature_map:
        return x_next[-1], features
    else:
        return x_next[-1]


def check_handle_reach_target(handle_points, target_points):
    all_dist = list(
        map(lambda p, q: (torch.tensor(p).float() - torch.tensor(q).float()).norm(), handle_points, target_points)
    )
    return (torch.tensor(all_dist) < 1.0).all()


def interpolate_feature_patch(feat, y, x, r):
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1

    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x1.float() - x) * (y - y0.float())
    wc = (x - x0.float()) * (y1.float() - y)
    wd = (x - x0.float()) * (y - y0.float())

    Ia = feat[:, :, y0 - r : y0 + r + 1, x0 - r : x0 + r + 1]
    Ib = feat[:, :, y1 - r : y1 + r + 1, x0 - r : x0 + r + 1]
    Ic = feat[:, :, y0 - r : y0 + r + 1, x1 - r : x1 + r + 1]
    Id = feat[:, :, y1 - r : y1 + r + 1, x1 - r : x1 + r + 1]

    return Ia * wa + Ib * wb + Ic * wc + Id * wd


def point_tracking(F0, F1, handle_points, handle_points_init, r_p):
    r_tracking_patch = 4
    with torch.no_grad():
        for i in range(len(handle_points)):
            pi0, pi = handle_points_init[i], handle_points[i]
            r1, r2 = int(pi[0]) - r_p, int(pi[0]) + r_p + 1
            c1, c2 = int(pi[1]) - r_p, int(pi[1]) + r_p + 1

            F1_neighbor = F1[:, :, r1:r2, c1:c2]
            f0 = F0[:, :, int(pi0[0]), int(pi0[1])]
            all_dist = (f0.unsqueeze(dim=-1).unsqueeze(dim=-1) - F1_neighbor).abs().sum(dim=1)

            row, col = divmod(all_dist.argmin().item(), all_dist.shape[-1])
            handle_points[i][0] = pi[0] - r_p + row
            handle_points[i][1] = pi[1] - r_p + col
    return handle_points


def build_arg_parser():
    parser = argparse.ArgumentParser(description="DragFDM: interactive latent-space manipulation")
    parser.add_argument("--config", type=str, default="biked_256.yml", help="YAML config file")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Checkpoint .pth file")
    parser.add_argument("--test_path", type=str, required=True, help="Test .npy file")
    parser.add_argument("--image_index", type=int, default=5, help="Test image index to manipulate")
    parser.add_argument("--output_dir", type=str, default="drag_output", help="Output directory")
    parser.add_argument("--timesteps", type=int, default=18, help="Number of sampling timesteps")
    parser.add_argument("--latent_step", type=int, default=3, help="Latent step for noise injection")
    parser.add_argument("--feature_step", type=int, default=10, help="Feature extraction step")
    parser.add_argument("--sigma_start", type=float, default=5.3, help="Sigma start")
    parser.add_argument("--sigma_end", type=float, default=0.6, help="Sigma end")
    parser.add_argument("--rho", type=float, default=7, help="EDM rho parameter")
    parser.add_argument("--r_p", type=int, default=5, help="Point tracking search radius")
    parser.add_argument("--r_n", type=int, default=5, help="Neighbourhood radius for motion supervision")
    parser.add_argument("--lam", type=float, default=1, help="Background regularisation weight")
    parser.add_argument("--n_steps", type=int, default=200, help="Number of optimisation steps")
    parser.add_argument("--lr", type=float, default=0.01, help="Optimiser learning rate")
    parser.add_argument("--handle_points", type=str, default="100,150", help="Handle point (row,col)")
    parser.add_argument("--target_points", type=str, default="80,150", help="Target point (row,col)")
    parser.add_argument("--mask_region", type=str, default="60,105,100,200", help="Mask region y1,y2,x1,x2")
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
    torch.backends.cudnn.benchmark = True

    os.makedirs(args.output_dir, exist_ok=True)

    # Parse points and mask
    hp = list(map(int, args.handle_points.split(",")))
    tp = list(map(int, args.target_points.split(",")))
    handle_points_init = [[hp[0], hp[1]]]
    target_points = [[tp[0], tp[1]]]
    mask_coords = list(map(int, args.mask_region.split(",")))
    y1, y2, x1, x2 = mask_coords

    # Load test image
    test_images = np.load(args.test_path, mmap_mode="r")
    test_images = test_images / 0.5 - 1
    test_img = torch.tensor(test_images[args.image_index : args.image_index + 1]).permute((0, 3, 1, 2)).to(device)

    # Create mask
    mask = torch.ones((1, config.data.image_size, config.data.image_size), dtype=torch.float32, device=device)
    for _x in range(x1, x2):
        for _y in range(y1, y2):
            mask[:, _y, _x] = 0

    # Load model
    model = Model(config)
    states = torch.load(args.ckpt_path, map_location=device)
    model = model.to(device)
    model = torch.nn.DataParallel(model, dataparallel)
    model.load_state_dict(states[0], strict=True)
    print(f"Loaded checkpoint (step {states[3]})")
    model.eval()

    # Compute noise schedule
    sigma_min = args.sigma_end**2 / args.sigma_start
    sigma_max = args.sigma_start**2 / args.sigma_end

    step_indices = torch.arange(args.timesteps, device=device)
    t_steps = (
        sigma_max ** (1 / args.rho)
        + step_indices / (args.timesteps - 1) * (sigma_min ** (1 / args.rho) - sigma_max ** (1 / args.rho))
    ) ** args.rho
    t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])])
    t_steps_flip = t_steps.flip(0)

    # Create initial latent code at latent_step
    init_code = test_img.clone()
    for i, (t_cur, t_next) in enumerate(
        zip(t_steps_flip[: -args.latent_step], t_steps_flip[1 : -args.latent_step + 1])
    ):
        init_code = init_code + (t_next - t_cur) * torch.randn_like(init_code)

    init_code_0 = copy.deepcopy(init_code)

    # Get initial features F0
    with torch.no_grad():
        x_prev, F0 = forward_unet(
            init_code, t_steps, args.latent_step, args.feature_step, model, args.timesteps,
            stochastic=False, feature_map=True,
        )

    # Optimise latent code
    init_code.requires_grad_(True)
    optimizer = torch.optim.Adam([init_code], lr=args.lr)
    handle_points = copy.deepcopy(handle_points_init)

    scaler = torch.cuda.amp.GradScaler()
    pbar = tqdm.tqdm(range(args.n_steps), desc="DragFDM", leave=False)

    loss = torch.tensor(0.0)
    for step_idx in pbar:
        with torch.cuda.amp.autocast():
            x_prev_updated, F1 = forward_unet(
                init_code, t_steps, args.latent_step, args.feature_step, model, args.timesteps,
                stochastic=False, feature_map=True,
            )

            if step_idx != 0:
                handle_points = point_tracking(F0, F1, handle_points, handle_points_init, args.r_p)
                pbar.set_description(f"handle: {handle_points}, loss: {loss.item():.4f}")

            if check_handle_reach_target(handle_points, target_points):
                break

            loss = 0.0
            for i in range(len(handle_points)):
                pi = torch.tensor(handle_points[i]).float()
                ti = torch.tensor(target_points[i]).float()
                if (ti - pi).norm() < 2:
                    continue
                di = (ti - pi) / (ti - pi).norm()
                f0_patch = F1[
                    :, :,
                    int(pi[0]) - args.r_n : int(pi[0]) + args.r_n + 1,
                    int(pi[1]) - args.r_n : int(pi[1]) + args.r_n + 1,
                ]
                f1_patch = interpolate_feature_patch(
                    F1, pi[0] + di[0], pi[1] + di[1], args.r_n,
                )
                loss += (f0_patch - f1_patch).abs().mean()

            loss += args.lam * ((init_code - init_code_0) * (1 - mask)).abs().mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # Denoise to full image and save
        with torch.no_grad():
            x_full = forward_unet(
                init_code, t_steps, args.latent_step, args.timesteps, model, args.timesteps,
                stochastic=True, feature_map=False,
            )
            x_img = (x_full + 1.0) / 2.0
            matplotlib.image.imsave(
                os.path.join(args.output_dir, f"{step_idx}.png"),
                x_img[0].permute(1, 2, 0).cpu().numpy().squeeze(),
                cmap="gray",
            )

    # Create GIF
    images = []
    for i in range(args.n_steps):
        fpath = os.path.join(args.output_dir, f"{i}.png")
        if os.path.exists(fpath):
            images.append(matplotlib.image.imread(fpath))
    if images:
        imageio.mimsave(os.path.join(args.output_dir, "drag_animation.gif"), images, duration=0.05)
        print(f"Saved {len(images)} frames and GIF to {args.output_dir}")


if __name__ == "__main__":
    main()
