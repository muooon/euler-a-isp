"""
euler-a-isp: Euler a based Inverse Scattering Problem Sampler
Author: muooon
Assistant: Copilot
License: Apache 2.0
"""

import torch
import tqdm
import k_diffusion.sampling
from modules import sd_samplers_common, sd_samplers_kdiffusion, sd_samplers
from tqdm.auto import trange
from k_diffusion import utils
from k_diffusion.sampling import to_d
import math

# A1111とReForgeの両方で統一
NAME = "euler-a-isp"
ALIAS = "euler-a-isp"

# device = x.device
# kernel = kernel.to(x.device)  # テンソルと同じデバイスに移動

import torch
from tqdm import trange

@torch.no_grad()
def wave_scattering_correction(x, alpha=0.1, beta=0.02, omega=5):
    """ 波動散乱逆問題によるAncestral補正を適用 """

    # CUDA に統一
    x = x.to(x.device)

    kernel = torch.tensor([[0.1, 0.2, 0.1], 
                           [0.2, 0.4, 0.2], 
                           [0.1, 0.2, 0.1]], dtype=torch.float32).to(x.device)

    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(x.shape[1], 1, 1, 1)

    # フィルタリング適用
    x_corrected = torch.nn.functional.conv2d(x, kernel, padding=1, groups=x.shape[1])

    # Ancestral補正適用
    tau = torch.arange(0, x_corrected.shape[-2], dtype=torch.float32).to(x.device)
    correction_factor = torch.exp(-beta * tau) * (1 + alpha * torch.sin(omega * tau))

    # サイズを統一
    correction_factor = correction_factor.view(1, 1, -1, 1).expand(x_corrected.shape).to(x.device)  # 修正：expand()

    x_corrected = x_corrected * correction_factor

    return x_corrected.to(x.device)

@torch.no_grad()
def spectral_transform(x):
    """ FFTを用いた周波数領域変換 """
    x_fft = torch.fft.fft2(x)
    return torch.real(torch.fft.ifft2(x_fft))  # 周波数から空間へ戻す

@torch.no_grad()
def optimize_tensor_shape(x):
    """ 行列分解 (SVD) によるサイズ整合性の確保 """
    batch_size = x.shape[0]
    U, S, V = torch.svd(x.view(batch_size, -1))  # 特異値分解
    x_optimized = U @ torch.diag_embed(S) @ V.T  # サイズ変換を補正
    return x_optimized.view_as(x)  # 元の形状に戻す

@torch.no_grad()
def isp_sampling_step(x, model, dt, sigma_hat, **extra_args):
    """ 修正: Ancestral 波動散乱補正を適用した Euler a サンプリングステップ """

    x = wave_scattering_correction(x)  # Ancestral の補正
    x = spectral_transform(x)  # FFT変換で安定化
    x = optimize_tensor_shape(x)  # SVD によりサイズ調整

    return x

@torch.no_grad()
def sample_euler_isp(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0.,
                     s_tmax=float("inf"), s_noise=1.):
    """ 修正: 波動逆補正を適用したオイラーAサンプリング """
    if model is None or x is None or sigmas is None:
        return torch.zeros_like(x) if x is not None else None

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = max(s_churn * (len(sigmas) - 1) ** -1, 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        dt = sigmas[i + 1] - sigma_hat

        if gamma > 0:
            x = x - eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5

        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)

        if sigmas[i + 1] > 0:
            if i // 2 == 1:
                x = isp_sampling_step(x, model, dt, sigma_hat, **extra_args)

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})

        x = x + d * dt

    return x

if NAME not in [x.name for x in sd_samplers.all_samplers]:
    isp_samplers = [
        sd_samplers_common.SamplerData(
            NAME,
            lambda model: sd_samplers_kdiffusion.KDiffusionSampler(sample_euler_isp, model) if model else None,
            [ALIAS],
            {}
        )
    ]
    
    # 既存のサンプラーに影響を与えないよう調整
    sd_samplers.all_samplers.extend([s for s in isp_samplers if s is not None])
    
    # 競合を防ぐために `.update()` を使用
    sd_samplers.all_samplers_map.update({x.name.lower(): x for x in isp_samplers if x is not None})