import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_blur(image, sigma):
    radius = int(math.ceil(3 * sigma))
    axis = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel = torch.exp(-(axis[:, None].square() + axis[None, :].square()) / (2 * sigma**2))
    kernel = (kernel / kernel.sum()).unsqueeze(0).unsqueeze(0)
    return F.conv2d(image, kernel, padding=radius)


def amplitude_mask(target_amplitude, relative_floor):
    mask = target_amplitude >= target_amplitude.amax() * relative_floor
    mask = mask.clone()
    mask[..., 0, 0] = False
    return mask


def normalized_log_amplitude_loss(prediction, target_amplitude, valid_mask):
    prediction_amplitude = torch.abs(torch.fft.fft2(prediction))
    scale = target_amplitude.amax().detach().clamp_min(1e-8)
    prediction_log = torch.log1p(prediction_amplitude / scale)
    target_log = torch.log1p(target_amplitude / scale)
    return F.mse_loss(prediction_log[valid_mask], target_log[valid_mask])


def soft_histogram_cdf_loss(prediction, target, bins, softness):
    centers = torch.linspace(0, 1, bins, device=prediction.device, dtype=prediction.dtype)
    prediction_values = prediction.flatten()
    target_values = target.flatten()
    prediction_cdf = torch.sigmoid((centers[:, None] - prediction_values[None, :]) / softness).mean(dim=1)
    target_cdf = torch.sigmoid((centers[:, None] - target_values[None, :]) / softness).mean(dim=1)
    return F.mse_loss(prediction_cdf, target_cdf)


def dynamic_contour(prediction, sigma, threshold):
    blurred = gaussian_blur(prediction, sigma)
    baseline = blurred.amin(dim=(-2, -1), keepdim=True)
    contrast = blurred - baseline
    peak = contrast.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return torch.sigmoid(12.0 * (contrast / peak - threshold))


def background_loss(prediction, contour):
    outside = 1.0 - contour.detach()
    return torch.mean((outside * prediction).square())


def contour_area_ratio_loss(contour, target_ratio):
    return F.mse_loss(contour.mean(), target_ratio)


class CalibratedWeights(nn.Module):
    """Fixed-weight loss combination with first-step auto-calibration.

        L = L_amp + β_hist·L_hist + β_bg·L_bg + β_area·L_area

    Amplitude is the anchor (weight 1). The other three are static penalties whose
    β is set once at the first forward pass so that each term's initial weighted
    contribution equals SHARE_i × L_amp,0. SHARE_i are global constants ordered by
    each constraint's credibility (amplitude > area > histogram > background), not
    per-image hyperparameters: when the image changes every L_i,0 changes, the β
    follow, but the contribution ratios stay fixed.

    No term carries an adaptive Lagrange multiplier. The earlier AL-PINNs scheme
    gave the area term a dual variable λ updated by gradient ascent every step;
    near convergence this primal-dual chasing produced a limit cycle (θ and λ
    hunting each other around the saddle point), so the area loss oscillated.
    Static penalties have no such feedback loop.
    """

    def __init__(self, share_histogram, share_background, share_area_ratio):
        super().__init__()
        self.shares = (share_histogram, share_background, share_area_ratio)
        self.register_buffer("betas", torch.zeros(3))   # order: histogram, background, area

    def total(self, amplitude, histogram, background, area_ratio):
        weighted = self.betas * torch.stack([histogram, background, area_ratio])
        return amplitude + weighted.sum()

    @torch.no_grad()
    def calibrate(self, amplitude, histogram, background, area_ratio):
        """First-step calibration: β_i = SHARE_i × L_amp,0 / L_i,0."""
        anchor = amplitude.detach().clamp_min(1e-12)
        initial = torch.stack([loss.detach() for loss in (histogram, background, area_ratio)])
        shares = torch.tensor(self.shares, device=initial.device, dtype=initial.dtype)
        self.betas.copy_(shares * anchor / initial.clamp_min(1e-12))
