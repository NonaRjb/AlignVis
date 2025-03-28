import sys

sys.path.append("/proj/rep-learning-robotics/users/x_nonra/alignvis")

import itertools
import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from datetime import datetime
import wandb
import random
import argparse
import os
import warnings
import pickle

import src.utils as utils
from src.models.brain_encoder import BrainEncoder
from src.models.image_encoder import ImageEncoder
from src.training.trainer import EEGDiffusionTrainer
from src.training.training_utils import CLIPLoss, SoftCLIPLoss
from src import downstream
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import *

model_configs = {
        'eegnet': {},
        'eegconformer': {},
        'nice': {'emb_size': 40, 'embedding_dim': 1440, 'drop_proj': 0.5},
        'atms': {},
        'lstm': {'lstm_size': 128, 'lstm_layers': 1},
        'resnet1d': {},
        'resnet1d_subj': {},
        'resnet1d_subj_resblk': {},
        'brain-mlp': {},
        'dinov2_vit-l-14_noalign': {'embed_dim': 1024},
        'DINOv2_ViT-L14_noalign': {'embed_dim': 1024},
        'DINO_ViT-B8_noalign': {'embed_dim': 768},
        'DINOv2_ViT-B14_noalign': {'embed_dim': 768},
        'DINO_ViT-B16_noalign': {'embed_dim': 768},
        'CLIP_ViT-L14_noalign': {'embed_dim': 768},
        'CLIP_ViT-B32_noalign': {'embed_dim': 512},
        'OpenCLIP_ViT-L14_laion400m_noalign': {'embed_dim': 768},
        'OpenCLIP_ViT-L14_laion2b_noalign': {'embed_dim': 768},
        'OpenCLIP_ViT-B32_laion400m_noalign': {'embed_dim': 512},
        'OpenCLIP_ViT-H14_laion2b_noalign': {'embed_dim': 1024},
        'dreamsim_clip_vitb32_768': {'embed_dim': 768},
        'dreamsim_clip_vitb32': {'embed_dim': 512},
        'dreamsim_clip_vitb32_noalign': {'embed_dim': 512},
        'dreamsim_open_clip_vitb32': {'embed_dim': 512},
        'dreamsim_open_clip_vitb32_noalign': {'embed_dim': 512},
        'dreamsim_synclr_vitb16': {'embed_dim': 768},
        'dreamsim_synclr_vitb16_noalign': {'embed_dim': 768},
        'dreamsim_ensemble': {'embed_dim': 1792},
        'dreamsim_ensemble_noalign': {'embed_dim': 1792},
        'dreamsim_dino_vitb16': {'embed_dim': 768},
        'dreamsim_dino_vitb16_noalign': {'embed_dim': 768},
        'dreamsim_dinov2_vitb14': {'embed_dim': 768},
        'dreamsim_dinov2_vitb14_noalign': {'embed_dim': 768},
        'gLocal_openclip_vit-l-14_laion2b_s32b_b82k': {'embed_dim': 768},
        'gLocal_openclip_vit-l-14_laion2b_s32b_b82k_noalign': {'embed_dim': 768},
        'gLocal_openclip_vit-l-14_laion400m_e32': {'embed_dim': 768},
        'gLocal_openclip_vit-l-14_laion400m_e32_noalign': {'embed_dim': 768},
        'gLocal_clip_vit-l-14': {'embed_dim': 768},
        'gLocal_clip_vit-l-14_noalign': {'embed_dim': 768},
        'gLocal_dino-vit-base-p8': {'embed_dim': 768},
        'gLocal_dino-vit-base-p8_noalign': {'embed_dim': 768},
        'gLocal_dino-vit-base-p16': {'embed_dim': 768},
        'gLocal_dino-vit-base-p16_noalign': {'embed_dim': 768},
        'gLocal_dinov2-vit-base-p14': {'embed_dim': 768},
        'gLocal_dinov2-vit-base-p14_noalign': {'embed_dim': 768},
        'gLocal_dinov2-vit-large-p14': {'embed_dim': 1024},
        'gLocal_dinov2-vit-large-p14_noalign': {'embed_dim': 1024},
        'gLocal_clip_rn50': {'embed_dim': 1024},
        'gLocal_clip_rn50_noalign': {'embed_dim': 1024},
        'harmonization_vitb16': {'embed_dim': 768},
        'harmonization_vitb16_noalign': {'embed_dim': 768},
        'harmonization_resnet50': {'embed_dim': 2048},
        'harmonization_resnet50_noalign': {'embed_dim': 2048},
        'harmonization_convnext': {'embed_dim': 768},
        'harmonization_convnext_noalign': {'embed_dim': 768},
        'harmonization_levit': {'embed_dim': 384},
        'harmonization_levit_noalign': {'embed_dim': 384},
        'harmonization_vgg16': {'embed_dim': 4096},
        'harmonization_vgg16_noalign': {'embed_dim': 4096},
    }

def seed_everything(seed_val):
    np.random.seed(seed_val)
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return

@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def generate_ip_adapter_embeds(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    timesteps: List[int] = None,
    denoising_end: Optional[float] = None,
    guidance_scale: float = 5.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    ip_adapter_image: Optional[PipelineImageInput] = None,
    ip_adapter_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
    original_size: Optional[Tuple[int, int]] = None,
    crops_coords_top_left: Tuple[int, int] = (0, 0),
    target_size: Optional[Tuple[int, int]] = None,
    negative_original_size: Optional[Tuple[int, int]] = None,
    negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
    negative_target_size: Optional[Tuple[int, int]] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    **kwargs,
):
    r"""
    Function invoked when calling the pipeline for generation.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            instead.
        prompt_2 (`str` or `List[str]`, *optional*):
            The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
            used in both text-encoders
        height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The height in pixels of the generated image. This is set to 1024 by default for the best results.
            Anything below 512 pixels won't work well for
            [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
            and checkpoints that are not specifically fine-tuned on low resolutions.
        width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The width in pixels of the generated image. This is set to 1024 by default for the best results.
            Anything below 512 pixels won't work well for
            [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
            and checkpoints that are not specifically fine-tuned on low resolutions.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        timesteps (`List[int]`, *optional*):
            Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
            in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
            passed will be used. Must be in descending order.
        denoising_end (`float`, *optional*):
            When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
            completed before it is intentionally prematurely terminated. As a result, the returned sample will
            still retain a substantial amount of noise as determined by the discrete timesteps selected by the
            scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a
            "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
            Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)
        guidance_scale (`float`, *optional*, defaults to 5.0):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        negative_prompt_2 (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
            `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
            [`schedulers.DDIMScheduler`], will be ignored for others.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
            to make generation deterministic.
        latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor will ge generated by sampling using the supplied random `generator`.
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
            If not provided, pooled text embeddings will be generated from `prompt` input argument.
        negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
            input argument.
        ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
        ip_adapter_embeds: (`FloatTensor`, *optional*): Optional image embeddings to work with IP Adapters.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
            of a plain tuple.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
            [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
            Guidance rescale factor should fix overexposure when using zero terminal SNR.
        original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
            If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
            `original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning as
            explained in section 2.2 of
            [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
        crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
            `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
            `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
            `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
            [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
        target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
            For most cases, `target_size` should be set to the desired height and width of the generated image. If
            not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in
            section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
        negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
            To negatively condition the generation process based on a specific image resolution. Part of SDXL's
            micro-conditioning as explained in section 2.2 of
            [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
            information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
        negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
            To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
            micro-conditioning as explained in section 2.2 of
            [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
            information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
        negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
            To negatively condition the generation process based on a target image resolution. It should be as same
            as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
            [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
            information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
        callback_on_step_end (`Callable`, *optional*):
            A function that calls at the end of each denoising steps during the inference. The function is called
            with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
            callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
            `callback_on_step_end_tensor_inputs`.
        callback_on_step_end_tensor_inputs (`List`, *optional*):
            The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
            will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
            `._callback_tensor_inputs` attribute of your pipeline class.

    Examples:

    Returns:
        [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or `tuple`:
        [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
        `tuple`. When returning a tuple, the first element is a list with the generated images.
    """

    callback = kwargs.pop("callback", None)
    callback_steps = kwargs.pop("callback_steps", None)

    if callback is not None:
        deprecate(
            "callback",
            "1.0.0",
            "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
        )
    if callback_steps is not None:
        deprecate(
            "callback_steps",
            "1.0.0",
            "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
        )

    # 0. Default height and width to unet
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    original_size = original_size or (height, width)
    target_size = target_size or (height, width)

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        callback_steps,
        negative_prompt,
        negative_prompt_2,
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs,
    )

    self._guidance_scale = guidance_scale
    self._guidance_rescale = guidance_rescale
    self._clip_skip = clip_skip
    self._cross_attention_kwargs = cross_attention_kwargs
    self._denoising_end = denoising_end

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    # 3. Encode input prompt
    lora_scale = (
        self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
    )

    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        lora_scale=lora_scale,
        clip_skip=self.clip_skip,
    )

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

    # 5. Prepare latent variables
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7. Prepare added time ids & embeddings
    add_text_embeds = pooled_prompt_embeds
    if self.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

    add_time_ids = self._get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )
    if negative_original_size is not None and negative_target_size is not None:
        negative_add_time_ids = self._get_add_time_ids(
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
    else:
        negative_add_time_ids = add_time_ids

    if self.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

    if ip_adapter_image is not None:
        image_embeds, negative_image_embeds = self.encode_image(ip_adapter_image, device, num_images_per_prompt)
        if self.do_classifier_free_guidance:
            image_embeds = torch.cat([negative_image_embeds, image_embeds])
            image_embeds = image_embeds.to(device)
    
    if ip_adapter_embeds is not None:
        image_embeds = ip_adapter_embeds.to(device=device, dtype=prompt_embeds.dtype)
        if self.do_classifier_free_guidance:
            negative_image_embeds = torch.zeros_like(image_embeds)
            image_embeds = torch.cat([negative_image_embeds, image_embeds])
            image_embeds = image_embeds.to(device)

    # 8. Denoising loop
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

    # 8.1 Apply denoising_end
    if (
        self.denoising_end is not None
        and isinstance(self.denoising_end, float)
        and self.denoising_end > 0
        and self.denoising_end < 1
    ):
        discrete_timestep_cutoff = int(
            round(
                self.scheduler.config.num_train_timesteps
                - (self.denoising_end * self.scheduler.config.num_train_timesteps)
            )
        )
        num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
        timesteps = timesteps[:num_inference_steps]

    # 9. Optionally get Guidance Scale Embedding
    timestep_cond = None
    if self.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
        timestep_cond = self.get_guidance_scale_embedding(
            guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
        ).to(device=device, dtype=latents.dtype)

    self._num_timesteps = len(timesteps)
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            if ip_adapter_image is not None or ip_adapter_embeds is not None:
                added_cond_kwargs["image_embeds"] = image_embeds
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=self.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                negative_pooled_prompt_embeds = callback_outputs.pop(
                    "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                )
                add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, latents)

            if XLA_AVAILABLE:
                xm.mark_step()

    if not output_type == "latent":
        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

        if needs_upcasting:
            self.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
    else:
        image = latents

    if not output_type == "latent":
        # apply watermark if available
        if self.watermark is not None:
            image = self.watermark.apply_watermark(image)

        image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    return StableDiffusionXLPipelineOutput(images=image)

def return_dataloaders(
    dataset_nm, 
    data_pth, 
    sid, 
    subj_training_ratio,
    batch, 
    num_workers, 
    seed_val, 
    device_type, 
    separate_test=False, 
    **kwargs):

    data, ds_configs = utils.load_dataset(
        dataset_name=dataset_nm, 
        data_path=data_path, 
        sid=sid, 
        subj_training_ratio=subj_training_ratio,
        split='train',
        **kwargs
        )
    
    print(ds_configs)
    
    g = torch.Generator().manual_seed(seed_val)

    if not separate_test:
        train_data, val_data, test_data = torch.utils.data.random_split(
            data, [0.8, 0.1, 0.1], generator=g)
    else:
        train_data, val_data = torch.utils.data.random_split(
            data, [0.9, 0.1], generator=g)
        test_data, _ = utils.load_dataset(
            dataset_name=dataset_nm, 
            data_path=data_path, 
            sid=kwargs['test_subject'], 
            split='test',  
            subj_training_ratio=1.0, 
            **kwargs)
    train_dl = DataLoader(train_data, batch_size=batch, shuffle=True,
                            drop_last=True,
                            num_workers=num_workers,
                            pin_memory=True if 'cuda' in device_type else False,
                            generator=g)
    val_dl = DataLoader(val_data, batch_size=64, shuffle=False,
                            drop_last=False,
                            num_workers=num_workers,
                            pin_memory=True if 'cuda' in device_type else False,
                            generator=g)
    test_dl = DataLoader(test_data, batch_size=1, shuffle=False,
                            drop_last=False,
                            pin_memory=True if 'cuda' in device_type else False,
                            generator=g)
    return train_dl, val_dl, test_dl, ds_configs


def encode_image(image, image_encoder, feature_extractor, num_images_per_prompt=1, device='cuda'):
    dtype = next(image_encoder.parameters()).dtype

    if not isinstance(image, torch.Tensor):
        image = feature_extractor(image, return_tensors="pt").pixel_values # [1, 3, 224, 224]
    
    image = image.to(device=device, dtype=dtype)
    image_embeds = image_encoder(image).image_embeds # (1, 1024)
    image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0) # (num_images_per_prompt, 1024)

    return image_embeds

class Generator4Embeds:

    def __init__(self, num_inference_steps=1, device='cuda') -> None:

        self.num_inference_steps = num_inference_steps
        self.dtype = torch.float16
        self.device = device
        
        # path = '/home/weichen/.cache/huggingface/hub/models--stabilityai--sdxl-turbo/snapshots/f4b0486b498f84668e828044de1d0c8ba486e05b'
        # path = "/home/ldy/Workspace/sdxl-turbo/f4b0486b498f84668e828044de1d0c8ba486e05b"
        pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16", cache_dir="/proj/rep-learning-robotics/users/x_nonra/.cache/diffusers_cache/")
        # pipe = DiffusionPipeline.from_pretrained(path, torch_dtype=torch.float16, variant="fp16")
        pipe.to(device)
        pipe.generate_ip_adapter_embeds = generate_ip_adapter_embeds.__get__(pipe)
        # load ip adapter
        pipe.load_ip_adapter(
            "h94/IP-Adapter", subfolder="sdxl_models", 
            weight_name="ip-adapter_sdxl_vit-h.safetensors", 
            torch_dtype=torch.float16)
        # set ip_adapter scale (defauld is 1)
        pipe.set_ip_adapter_scale(1)
        self.pipe = pipe

    def generate(self, image_embeds, text_prompt='', generator=None):
        image_embeds = image_embeds.to(device=self.device, dtype=self.dtype)
        pipe = self.pipe

        # generate image with image prompt - ip_adapter_embeds
        image = pipe.generate_ip_adapter_embeds(
            prompt=text_prompt, 
            ip_adapter_embeds=image_embeds, 
            num_inference_steps=self.num_inference_steps,
            guidance_scale=0.0,
            generator=generator,
        ).images[0]

        return image
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None)
    parser.add_argument('--data_path', type=str, help="Path to the EEG data")
    parser.add_argument('--save_path', type=str, help="Path to save the model")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to the pretrained model")
    parser.add_argument('--dataset', type=str, default="things-eeg-2")
    parser.add_argument('--subject_id', type=int, nargs='+', default=[0], help="Subject ID(s). Provide one or more subject IDs.")
    parser.add_argument('--test_subject', type=int, default=None)
    parser.add_argument('--subj_training_ratio', type=float, default=1, help="a ratio between 0 and 1 determining how much of participants training samples to be used")
    parser.add_argument('--channels', type=int, nargs='+', default=None)
    parser.add_argument('--interpolate', type=int, default=None, help="Resampling rate for EEG data")
    parser.add_argument('--window', type=float, nargs=2, default=None, help="Window start and end for EEG data")
    parser.add_argument('--eeg_enc', type=str, default="resnet1d", help="EEG Encoder")
    parser.add_argument('--img_enc', type=str, default="CLIP_IMG", help="Image Encoder")
    parser.add_argument('--img_enc_model', type=str, default=None, help="Image Encoder Model")
    parser.add_argument('--loss', type=str, default="clip-loss", help="Loss function")
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--net_filter_size', type=int, nargs='+', default=None)
    parser.add_argument('--net_seq_length', type=int, nargs='+', default=None)
    parser.add_argument('--img', type=str, default="embedding")
    parser.add_argument('--epoch', type=int, default=1000, help="Number of epochs for pretraining")
    parser.add_argument('--finetune_epoch',  type=int, default=50, help="Number of epochs for finetuning (if the downstream task is classification)")
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--scheduler', type=str, default="plateau")
    parser.add_argument('--patience', type=int, default=25, help="Patience for the reduce_lr_on_plateau scheduler")
    parser.add_argument('--temperature', type=float, default=0.04)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--downstream', type=str, default=None)
    parser.add_argument('--separate_test', action="store_true")
    parser.add_argument('--return_subject_id', action="store_true")
    parser.add_argument('--subj_spec_epochs', type=int, default=0)
    parser.add_argument('-b', '--batch', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--experiment', type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    with wandb.init():
        # args = wandb.config
        seed = args.seed
        dataset_name = args.dataset
        subject_id = args.subject_id
        if len(subject_id) == 1:    #TODO to be compatible with Spampinato until I fix it
            subject_id = subject_id[0]
        test_subject = args.test_subject
        brain_enc_name = args.eeg_enc
        img_enc_name = args.img_enc
        batch_size = args.batch
        lr = args.lr
        epochs = args.epoch
        finetune_epochs = args.finetune_epoch
        data_path = args.data_path
        save_path = args.save_path
        downstream_task = args.downstream
        separate_test_set = args.separate_test
        channels=args.channels

        if args.subj_training_ratio == 0:
            epochs=0

        if args.net_filter_size:
            model_configs['resnet1d']['net_filter_size'] = args.net_filter_size
            model_configs['resnet1d_subj']['net_filter_size'] = args.net_filter_size
            model_configs['resnet1d_subj_resblk']['net_filter_size'] = args.net_filter_size

        if args.net_seq_length:
            model_configs['resnet1d']['net_seq_length'] = args.net_seq_length
            model_configs['resnet1d_subj']['net_seq_length'] = args.net_seq_length
            model_configs['resnet1d_subj_resblk']['net_seq_length'] = args.net_seq_length

        if args.dataset == "things-meg":
            model_configs['nice']['embedding_dim'] = 217360
            model_configs['nice']['emb_size'] = 40
        elif args.dataset == "things-eeg-preprocessed":
            model_configs['nice']['embedding_dim'] = 240
            model_configs['nice']['emb_size'] = 40

        if args.checkpoint:
            model_configs[brain_enc_name]['subject_ids'] = [str(s) for s in range(1, 11)]
        else:
            model_configs[brain_enc_name]['subject_ids'] = [str(s) for s in subject_id] if isinstance(subject_id, list) else [str(subject_id)]

        if separate_test_set and downstream_task == "classification":
            warnings.warn("The test set won't be used to finetune the classifier. seperate_test will be set to False")
            separate_test_set = False
        
        print("training subjects: ", subject_id)
        print("test subjects: ", test_subject if test_subject is not None else subject_id)

        # constants
        min_lr = 1e-07
        warmup_epochs = args.warmup
        weight_decay=0.1
        
        seed_everything(seed)
        if args.experiment is not None:
            save_path = os.path.join(save_path, args.experiment)
            os.makedirs(save_path, exist_ok=True)
        paths = {"brain_data": data_path, "save_path": save_path}
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("device = ", device)

        print("**********************************************************************************************")
        print(f"Starting a run on {dataset_name} with {brain_enc_name}")

        start_str = "scratch" if args.checkpoint is None else "pretrained"

        if test_subject is None:
            directory_name = f"{img_enc_name}"
        else:
            directory_name = f"{img_enc_name}"
        
        current_datetime = datetime.now()
        # directory_name += current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(os.path.join(paths["save_path"], directory_name), exist_ok=True)
        if len(args.subject_id) == 1:
            os.makedirs(os.path.join(paths["save_path"], directory_name, f"sub-{subject_id:02}"), exist_ok=True)
            paths["save_path"] = os.path.join(paths["save_path"], directory_name, f"sub-{subject_id:02}")
        else:
            os.makedirs(os.path.join(paths["save_path"], directory_name, f"sub-{test_subject:02}"), exist_ok=True)
            paths["save_path"] = os.path.join(paths["save_path"], directory_name, f"sub-{test_subject:02}")
        
        print(f"Directory '{directory_name}' created.")
        utils.save_config(args, root_path=paths['save_path'])
        print(vars(args))

        train_data_loader, val_data_loader, test_data_loader, data_configs = return_dataloaders(
            dataset_nm=dataset_name, 
            data_pth=paths['brain_data'], sid=subject_id, 
            test_subject=test_subject if test_subject is not None else subject_id,
            batch=batch_size, 
            num_workers=args.n_workers,
            seed_val=seed, 
            load_img=args.img,
            separate_test=separate_test_set,
            select_channels=channels,
            return_subject_id=args.return_subject_id,
            subj_training_ratio=args.subj_training_ratio if args.subj_training_ratio > 0 else 0.01,
            img_encoder=img_enc_name,
            interpolate=args.interpolate,
            window=args.window,
            device_type=device)   
        
        embedding_size_retrieve = model_configs[img_enc_name]['embed_dim']
        
        print("eeg embedding size: ", embedding_size_retrieve, "image embedding size: 1024")
        brain_encoder = BrainEncoder(
            embed_dim=embedding_size_retrieve,
            backbone=brain_enc_name,
            n_channels=data_configs["n_channels"],
            n_samples=data_configs["n_samples"],
            n_classes=data_configs["n_classes"],
            model_path=None,
            device=device, 
            **model_configs[brain_enc_name]
            )
        brain_encoder = brain_encoder.float()
        brain_encoder.to(device)

        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint)['model_state_dict']
            brain_encoder.load_state_dict(checkpoint, strict=False)
            brain_encoder.to(device)

        sdxl = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            cache_dir="/proj/rep-learning-robotics/users/x_nonra/.cache/diffusers_cache/",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(device)
        # sdxl.generate_ip_adapter_embeds = generate_ip_adapter_embeds.__get__(sdxl)
        sdxl.load_ip_adapter(
            "h94/IP-Adapter", subfolder="sdxl_models", 
            weight_name="ip-adapter_sdxl_vit-h.bin", 
            torch_dtype=torch.float16)
        sdxl.set_ip_adapter_scale(1.0)  # focus only on image prompt

        # from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

        # image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        #     "h94/IP-Adapter", 
        #     subfolder="models/image_encoder",  # this is the ViT-H one
        #     torch_dtype=torch.float16,
        # ).to(device)

        # image_processor = CLIPImageProcessor()


        # from diffusers.utils import load_image
        # image = load_image("/proj/rep-learning-robotics/users/x_nonra/alignvis/data/things_eeg_2/images/test_images/00001_aircraft_carrier/aircraft_carrier_06s.jpg")
        # pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device, dtype=torch.float16)
        # image_embeds = image_encoder(pixel_values).image_embeds  # (1, 1024)
        # image_embeds = image_embeds.unsqueeze(0).unsqueeze(0)
        # do_cfg = True
        # if do_cfg:
        #     negative_embeds = torch.zeros_like(image_embeds)
        #     image_embeds = torch.cat([negative_embeds, image_embeds], dim=0)

        # print(image_embeds[0][0])
        # print(type(image_embeds))
        # print(image_embeds[0].shape)

        # images = sdxl(
        #     prompt="",
        #     ip_adapter_image_embeds=[image_embeds],
        #     negative_prompt="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
        #     num_inference_steps=100,
        #     generator=None,
        # ).images
        # images[0].save("test.jpg")
        # exit()

        trainer = EEGDiffusionTrainer( 
                brain_encoder=brain_encoder, 
                generator=sdxl,
                eeg_embed_dim=embedding_size_retrieve, 
                recon_dim=1024, 
                retrieve_dim=embedding_size_retrieve,
                save_path=paths["save_path"], 
                filename=f'{brain_enc_name}_{dataset_name}_seed{seed}', 
                epochs=epochs,
                lr=lr,
                scheduler=args.scheduler,
                min_lr=min_lr,
                warmup_epochs=warmup_epochs,
                lr_patience=args.patience, 
                es_patience=30,
                return_subject_id=False,
                precompute_img_emb=True,
                device=device,
                )
        best_model = trainer.train(train_data_loader, val_data_loader)
        trainer.brain_encoder.load_state_dict(best_model['brain_state_dict'])
        trainer.projector_recon.load_state_dict(best_model['projector_state_dict'])
        trainer.generate(test_data_loader)

        