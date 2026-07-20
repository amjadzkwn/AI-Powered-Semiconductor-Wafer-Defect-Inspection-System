import cv2
import numpy as np
import torch


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output
        output.register_hook(self._save_gradient)

    def _save_gradient(self, gradient):
        self.gradients = gradient

    def generate(self, input_tensor: torch.Tensor, target_index: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        logits[:, target_index].sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1))[0]
        cam = cam.detach().cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam

    def close(self):
        self.forward_handle.remove()


def make_overlay(wafer_map: np.ndarray, cam: np.ndarray) -> np.ndarray:
    cam = cv2.resize(cam, (wafer_map.shape[1], wafer_map.shape[0]))
    heat = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    base = np.zeros((*wafer_map.shape, 3), dtype=np.uint8)
    base[wafer_map == 1] = [120, 120, 120]
    base[wafer_map == 2] = [255, 255, 255]
    return cv2.addWeighted(base, 0.55, heat, 0.45, 0)
