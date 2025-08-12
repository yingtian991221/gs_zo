#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

# from zo_trainer import OurTrainer
from zo_wrapper import ZOWrapper

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def compute_loss(view_cam, gaussians, pipe, bg, opt, dataset):
    render_pkg = render(view_cam, gaussians, pipe, bg,
                        use_trained_exp=dataset.train_test_exp,
                        separate_sh=SPARSE_ADAM_AVAILABLE)
    image = render_pkg["render"]
    if view_cam.alpha_mask is not None:
        image *= view_cam.alpha_mask.cuda()
    gt = view_cam.original_image.cuda()
    Ll1 = l1_loss(image, gt)
    ssim_value = fused_ssim(image.unsqueeze(0), gt.unsqueeze(0)) if FUSED_SSIM_AVAILABLE else ssim(image, gt)
    return Ll1, (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value), render_pkg["radii"]


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    eps=1e-3
    lr=1e-1
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)

    zo = ZOWrapper(gaussians)

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0


    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        zo_params = [
            "_xyz",
            "_features_dc",
            "_features_rest",
            "_scaling",
            "_rotation",
            "_opacity",
        ]
        fixed_bg = background


        gaussians.optimizer.zero_grad(set_to_none=True)

        # 用“干净的”前向，和训练时一致但不做扰动
        Ll1_auto, loss_auto, _ = compute_loss(viewpoint_cam, gaussians, pipe, fixed_bg, opt, dataset)
        loss_auto.backward()  # 标准反传，所有需要 grad 的参数都会有 .grad

        # 收集关心的张量梯度
        params_to_check = {
            "_xyz":       getattr(gaussians, "_xyz", None),
            "_scaling":   getattr(gaussians, "_scaling", None),
            "_rotation":  getattr(gaussians, "_rotation", None),
            "_opacity":   getattr(gaussians, "_opacity", None),
            "_features_dc":   getattr(gaussians, "_features_dc", None),
            "_features_rest": getattr(gaussians, "_features_rest", None),
        }
        auto_grads = {n: (t.grad.clone() if (t is not None and t.grad is not None) else None)
                    for n,t in params_to_check.items()}

        # 清掉 autograd 的梯度，避免影响后面的 ZO
        gaussians.optimizer.zero_grad(set_to_none=True)


        # 2) 生成每个参数张量自己的 z，并做 +eps 渲染
        torch.manual_seed(torch.randint(0, 1_000_000_000, (1,)).item())
        seed = torch.initial_seed()
        if iteration<1000:
            param_names=["_xyz"]
        else:
            param_names = ["_xyz","_scaling","_rotation","_opacity","_features_dc","_features_rest"]  # 按需取舍
        tensors = {n: getattr(gaussians, n) for n in param_names if hasattr(gaussians, n)}
        orig = {n: t.data.clone() for n, t in tensors.items()}
        z    = {n: torch.randn_like(t) for n, t in tensors.items()}
        
        # +eps
        with torch.no_grad():
            for n,t in tensors.items():
                t.add_(eps * z[n])
        Ll1, loss1, radii = compute_loss(viewpoint_cam, gaussians, pipe, fixed_bg, opt, dataset)

        # -eps（从 +eps 直接减 2eps）
        with torch.no_grad():
            for n,t in tensors.items():
                t.add_(-2*eps * z[n])
        _,loss2,_ = compute_loss(viewpoint_cam, gaussians, pipe, fixed_bg, opt, dataset)

        # 复原
        with torch.no_grad():
            for n,t in tensors.items():
                t.copy_(orig[n])

        # 方向导数（标量）
        g_scalar = (loss1 - loss2) / (2*eps)

        zo_grads = {n: g_scalar * z[n] for n in tensors}  # 每张量的 ZO 伪梯度

        # 3) 逐张量做对比：范数 & 余弦相似度
        def flat(x): return x.reshape(1, -1)
        report_lines = []
        for n in tensors.keys():
            g_auto = auto_grads.get(n, None)
            g_zo   = zo_grads.get(n, None)
            if g_auto is None or g_zo is None: 
                continue
            # 为避免极端值影响，建议在计算相似度前把很小的值过滤或加个 eps
            auto_norm = g_auto.norm().item()
            zo_norm   = g_zo.norm().item()
            # 余弦相似度（加一个很小的 eps 防 NaN）
            cos = torch.nn.functional.cosine_similarity(
                flat(g_auto), flat(g_zo), dim=1, eps=1e-12
            ).item()
            report_lines.append(f"{n:14s}  ||g_auto||={auto_norm:.3e}  ||g_zo||={zo_norm:.3e}  cos={cos:.4f}")

        print("\n[GradCompare @ iter {:d}] loss_auto={:.6f}".format(iteration, loss_auto.item()))
        for l in report_lines: print("  " + l)

        # 3) 把“标量 × 各自的 z”作为伪梯度，逐张量 step（省显存）
        for n,t in tensors.items():
            # 重新用同一seed确保和上面一致（可选；这里我们直接用缓存的 z）
            pseudo_grad = g_scalar * z[n]*0.001
            t.grad = pseudo_grad
            gaussians.optimizer.step()              # 利用原来 per-group 的学习率
            t.grad = None

        # 如果有 rotation，记得单位化
        if "_rotation" in tensors:
            with torch.no_grad():
                r = getattr(gaussians, "_rotation")
                r.copy_(r / (r.norm(dim=-1, keepdim=True).clamp_min(1e-8)))
    
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss1.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            if iteration %2000==0:
                print("XYZ :", gaussians._xyz)
                print("Scaling shape:", gaussians._scaling.shape)
                print("Rotation shape:", gaussians._rotation.shape)
                print("Opacity shape:", gaussians._opacity.shape)
            # test_iter
            # Log and save
            training_report(tb_writer, iteration, Ll1, loss1, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                scene.save(iteration)

            # Densification
            # if iteration < opt.densify_until_iter:
            #     # Keep track of max radii in image-space for pruning
               
            #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #         gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
            #     if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            #         gaussians.reset_opacity()



            if (iteration in checkpoint_iterations):
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    # if iteration in testing_iterations:
    if iteration % 1000==0:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[5_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 1000,3000,5000,7000,12000,15000,20000,25000,30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
