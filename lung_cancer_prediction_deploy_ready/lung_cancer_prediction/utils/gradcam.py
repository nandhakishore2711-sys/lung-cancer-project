"""
Grad-CAM explainability.

Radiologist support isn't just "here's a label" — a heatmap showing WHICH
region of the CT slice drove the prediction is standard in medical-imaging
papers and makes the demo far more convincing. This module implements
Grad-CAM for both backends behind one function, get_gradcam_overlay(),
so app.py doesn't care which framework produced the model.

Returns a PIL Image (the original scan with a translucent heatmap overlay)
ready to be base64-encoded and sent to the browser.
"""

import numpy as np
from PIL import Image


def _overlay_heatmap(original_rgb_float, heatmap, alpha=0.4):
    """
    original_rgb_float: (H, W, 3) float32 in [0,1]
    heatmap: (h, w) float32 in [0,1], smaller spatial size than the image
    Returns a PIL Image of the blended result at the original resolution.
    """
    h_img, w_img = original_rgb_float.shape[:2]

    heat_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
        (w_img, h_img), Image.BILINEAR
    )
    heat_arr = np.asarray(heat_img, dtype=np.float32) / 255.0

    # Simple "jet-like" colormap without a hard matplotlib dependency:
    # red channel rises with intensity, blue falls, green peaks in the middle.
    r = np.clip(1.5 * heat_arr - 0.2, 0, 1)
    g = np.clip(1.5 - np.abs(heat_arr - 0.5) * 3.0, 0, 1)
    b = np.clip(1.2 - 1.5 * heat_arr, 0, 1)
    heat_rgb = np.stack([r, g, b], axis=-1)

    blended = (1 - alpha * heat_arr[..., None]) * original_rgb_float + \
              (alpha * heat_arr[..., None]) * heat_rgb
    blended = np.clip(blended, 0, 1)

    return Image.fromarray((blended * 255).astype(np.uint8))


def gradcam_tensorflow(model, image_batch, class_index, last_conv_layer_name):
    """
    model: compiled Keras model
    image_batch: (1, H, W, 3) float32 in [0,1]
    class_index: int, which class's score to explain
    """
    import tensorflow as tf

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(image_batch)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    original = image_batch[0]
    return _overlay_heatmap(original, heatmap)


def gradcam_pytorch(model, image_tensor, class_index):
    """
    model: LungCancerCNN instance (forward returns logits, features)
    image_tensor: torch tensor (1, 3, H, W) float32 in [0,1], requires_grad handled internally
    """
    import torch

    model.eval()
    image_tensor = image_tensor.clone().detach().requires_grad_(True)

    logits, features = model(image_tensor)
    features.retain_grad()

    score = logits[0, class_index]
    model.zero_grad()
    score.backward()

    grads = features.grad[0]              # (C, H, W)
    fmap = features[0].detach()            # (C, H, W)
    weights = grads.mean(dim=(1, 2))       # (C,)

    heatmap = torch.relu((weights[:, None, None] * fmap).sum(0))
    heatmap = heatmap / (heatmap.max() + 1e-8)
    heatmap = heatmap.detach().cpu().numpy()

    original = image_tensor[0].detach().cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
    return _overlay_heatmap(original, heatmap)


def get_gradcam_overlay(backend, model, preprocessed_input, class_index, last_conv_layer_name=None):
    """Unified entry point used by app.py."""
    if backend == "tensorflow":
        return gradcam_tensorflow(model, preprocessed_input, class_index, last_conv_layer_name)
    elif backend == "pytorch":
        import torch
        tensor = torch.from_numpy(preprocessed_input).float()
        return gradcam_pytorch(model, tensor, class_index)
    else:
        raise ValueError(f"Unknown backend: {backend}")
