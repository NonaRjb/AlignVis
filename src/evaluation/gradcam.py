import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                break

    def generate_heatmap(self, input_signal, target):
        self.model.zero_grad()
        output = self.model(input_signal)
        output = output - torch.mean(output, dim=-1, keepdim=True)
        output = F.normalize(output, p=2, dim=-1)
        target = target - torch.mean(target, dim=-1, keepdim=True)
        target = F.normalize(target, p=2, dim=-1)
        output.backward(target)

        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations

        # Compute weights
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        # Generate heatmap
        grad_cam_map = torch.sum(weights * activations, dim=1).squeeze()
        grad_cam_map = F.relu(grad_cam_map)
        grad_cam_map = grad_cam_map / grad_cam_map.max()  # Normalize
        grad_cam_map = grad_cam_map.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        grad_cam_map = F.interpolate(
            grad_cam_map,
            input_signal.shape[2:],
            mode='bicubic',
            align_corners=False)

        return grad_cam_map.detach().cpu().numpy()

def normalize(x: np.ndarray) -> np.ndarray:
    # Normalize to [0, 1].
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x

# Modified from: https://github.com/salesforce/ALBEF/blob/main/visualization.ipynb
def getAttMap(img, attn_map, blur=True):
    # attn_map = np.expand_dims(attn_map, axis=-1)
    if blur:
        attn_map = filters.gaussian_filter(attn_map, 0.02*max(img.shape))
    attn_map = normalize(attn_map)
    cmap = plt.get_cmap('jet')
    # attn_map_c = np.delete(cmap(attn_map), 3, 2)
    attn_map = 1*(1-attn_map**0.7)*img + \
            (attn_map**0.7)*cmap(attn_map)[:, :, 0]
    return attn_map

def viz_attn(img, attn_map, blur=True, ch_names=None, save_path=None, title='EEG Signal'):
    _, axes = plt.subplots(1, 1, figsize=(20, 18), dpi=300)
    # axes[0].imshow(img)
    img = img*1000
    plot_multivariate_time_series(
        img, 
        ax=axes, 
        attn_map=attn_map, 
        blur=blur, 
        vertical_offset=120, 
        ch_names=ch_names,
        title=title)
    # axes[1].imshow(getAttMap(img, attn_map, blur))
    # for ax in axes:
    # axes.axis("off")
    plt.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.05)
    if save_path:
        plt.savefig(save_path)
    else:  
        plt.show()


def plot_multivariate_time_series(data, ax=None, attn_map=None, blur=False, vertical_offset=10, ch_names=None, sampling_frequency=250, title='Multivariate Time Series'):
    """
    Plot multivariate time series with vertical offsets for visualization, optionally overlaying an attention map.

    Parameters:
        data (numpy.ndarray): 2D array of shape (channels, timepoints) representing the time series data.
        ax (matplotlib.axes.Axes): Matplotlib Axes object to plot on. If None, a new figure and axes are created.
        attn_map (numpy.ndarray): 2D array of shape (channels, timepoints) representing attention weights.
        blur (bool): Whether to apply Gaussian blur to the attention map.
        vertical_offset (float): Vertical offset between channels.
        title (str): Title of the plot.

    Returns:
        None
    """
    # Validate input
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array with shape (channels, timepoints).")
    print("vertical_offset:", vertical_offset)
    channels, timepoints = data.shape

    if attn_map is not None:
        if attn_map.shape != data.shape:
            raise ValueError("Attention map must have the same shape as the data.")

        # Process attention map
        if blur:
            attn_map = filters.gaussian_filter(attn_map, 0.02 * max(data.shape))
        attn_map = normalize(attn_map)

    # Create the plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    for i in range(channels):
        ax.plot(data[i] + i * vertical_offset, label=f'Channel {i+1}', linewidth=1.2, alpha=0.7)
        
        # Overlay attention map if provided
        if attn_map is not None:
            alpha = 1.0  # Transparency for the attention overlay
            ax.fill_between(
                range(timepoints),
                data[i] + i * vertical_offset - 0.7 * attn_map[i] * vertical_offset,  # Baseline
                data[i] + i * vertical_offset + 0.7 * attn_map[i] * vertical_offset,  # Signal
                where=attn_map[i] > 0,  # Highlight where attention is non-zero
                color='red',
                alpha=alpha * attn_map[i]
            )

    # Add labels, grid, and legend
    ax.set_xlabel('Time (ms)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Channels', fontsize=18, fontweight='bold')
    if ch_names is not None:
        ax.set_yticks(np.arange(channels) * vertical_offset)
        ax.set_yticklabels(ch_names, fontweight='bold')
    # Set x-axis to time in milliseconds
    time_in_ms = np.linspace(0, timepoints / sampling_frequency * 1000, timepoints)
    ax.set_xticks(np.linspace(0, timepoints, 11))  # 11 evenly spaced ticks
    ax.set_xticklabels([f"{t:.0f}" for t in np.linspace(0, time_in_ms[-1], 11)])
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.grid(True, alpha=0.5)


class Hook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, model, target_layer: str):
        self.data = None
        self.model = model
        for name, module in self.model.named_modules():
            if name == target_layer:
                self.hook = module.register_forward_hook(self.save_grad)
                break
        # self.hook = module.register_forward_hook(self.save_grad)
        
    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()
        
    @property
    def activation(self) -> torch.Tensor:
        return self.data
    
    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad


# Reference: https://arxiv.org/abs/1610.02391
def gradCAM(
    model: nn.Module,
    input: torch.Tensor,
    target: torch.Tensor,
    layer: nn.Module
) -> torch.Tensor:
    # Zero out any gradients at the input.
    if input.grad is not None:
        input.grad.data.zero_()
        
    # Disable gradient settings.
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)
        
    # Attach a hook to the model at the desired layer.
    # assert isinstance(layer, nn.Module)
    with Hook(model, layer) as hook:        
        # Do a forward and backward pass.
        output = model(input)
        output = output - torch.mean(output, dim=-1, keepdim=True)
        output = F.normalize(output, p=2, dim=-1)
        target = target - torch.mean(target, dim=-1, keepdim=True)
        target = F.normalize(target, p=2, dim=-1)
        output.backward(target)

        grad = hook.gradient.float()
        act = hook.activation.float()
    
        # Global average pool gradient across spatial dimension
        # to obtain importance weights.
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps over channel
        # dimension.
        gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
        # We only want neurons with positive influence so we
        # clamp any negative ones.
        gradcam = torch.clamp(gradcam, min=0)

    # Resize gradcam to input resolution.
    gradcam = F.interpolate(
        gradcam,
        input.shape[2:],
        mode='bicubic',
        align_corners=False)
    
    # Restore gradient settings.
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])
        
    return gradcam

