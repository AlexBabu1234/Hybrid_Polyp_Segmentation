import torch
import numpy as np
import cv2


class GradCAM:

    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate(self, input_tensor):

        self.model.zero_grad()

        output = self.model(input_tensor)

        # For segmentation, we want to explain the model's positive predictions.
        # We only take the sum of logits where the model actually predicts 'polyp'
        # to prevent background negative logits from destroying the gradient signal.
        predicted_mask = (output > 0.0).float()
        
        if predicted_mask.sum() == 0:
            # If no polyp is predicted, just explain the most confident pixel
            score = output.max()
        else:
            score = (output * predicted_mask).sum()

        score.backward()

        gradients = self.model.get_gradients()

        activations = self.model.get_activations()

        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()

        heatmap = heatmap.cpu().detach().numpy()

        heatmap = np.maximum(heatmap, 0)

        # Normalize the heatmap
        heatmap_max = np.max(heatmap)
        if heatmap_max > 0:
            heatmap /= heatmap_max

        return heatmap


def generate_gradcam(model, input_tensor, original_frame):
    """Convenience function: returns a BGR overlay image with GradCAM heatmap.

    Args:
        model: AttentionUNet model (must support get_gradients / get_activations).
        input_tensor: preprocessed image tensor [1, 3, H, W] with requires_grad=True.
        original_frame: original BGR frame (numpy array).

    Returns:
        overlay: BGR numpy array with heatmap blended onto the original frame.
    """

    cam = GradCAM(model)
    heatmap = cam.generate(input_tensor)

    heatmap_resized = cv2.resize(heatmap, (original_frame.shape[1], original_frame.shape[0]))

    # Use VIRIDIS colormap which matches the reference image style
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_VIRIDIS)

    # Increase heatmap opacity (0.7) so it looks more like the pure heatmap reference
    overlay = cv2.addWeighted(original_frame, 0.3, heatmap_colored, 0.7, 0)

    return overlay