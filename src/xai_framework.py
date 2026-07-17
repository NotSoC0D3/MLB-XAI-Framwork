import os
import cv2
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# Import SmallCNN architecture
from src.models import SmallCNN

# Define device globally as expected by the functions
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(image_path):
    """Loads and preprocesses the leaf image."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image)
    return tensor

def integrated_gradients(model, input_tensor, target_class, baseline=None, steps=50):
    """Computes Integrated Gradients for a given input."""
    if baseline is None:
        baseline = torch.zeros_like(input_tensor).to(DEVICE)
    
    input_tensor = input_tensor.to(DEVICE)
    
    # Scale inputs
    scaled_inputs = [
        (baseline + i/steps * (input_tensor - baseline)).clone().detach().requires_grad_(True)
        for i in range(steps + 1)
    ]

    grads = []
    for scaled in scaled_inputs:
        output = model(scaled)
        loss = output[0, target_class]

        model.zero_grad()
        loss.backward()

        grads.append(scaled.grad.detach().clone())

    avg_grads = torch.mean(torch.stack(grads), dim=0)
    integrated = (input_tensor - baseline) * avg_grads
    return integrated

def overlay_integrated_gradients(img, ig_attr):
    """Visualizes the Integrated Gradients overlay."""
    # Convert CHW → HWC for image
    img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # normalize for display

    # Process IG tensor
    ig = ig_attr.squeeze().cpu().detach().numpy()   # (3,H,W)
    ig = np.sum(np.abs(ig), axis=0)                 # → (H,W)
    ig = (ig - ig.min()) / (ig.max() - ig.min() + 1e-8)    # normalize 0-1

    # Plot overlay
    plt.figure(figsize=(5, 5))
    plt.imshow(img_np)
    plt.imshow(ig, cmap="hot", alpha=0.45)
    plt.title("Integrated Gradients Overlay")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    
    return ig

def gradcam_visualize(model, image, true_label, pred_label, target_layer):
    """Visualizes Grad-CAM and returns the CAM array for fusion."""
    model.eval()
    gradients, activations = [], []

    def save_gradient(grad):
        gradients.append(grad)

    def forward_hook(module, input, output):
        activations.append(output)
        output.register_hook(save_gradient)

    # hook the target layer
    hook = target_layer.register_forward_hook(forward_hook)

    # forward + backward
    image = image.unsqueeze(0).to(DEVICE)
    output = model(image)
    score = output[:, pred_label]
    score.backward()

    # compute Grad-CAM
    grads = gradients[0].mean(dim=(2,3), keepdim=True)
    cams = (activations[0] * grads).sum(dim=1).squeeze().cpu().detach().numpy()
    cams = np.maximum(cams, 0)
    cams = cams / (cams.max() + 1e-8)

    hook.remove()

    # convert image to numpy
    img_np = image.squeeze().permute(1,2,0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

    # plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(8,4))

    # original image
    axes[0].imshow(img_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Grad-CAM overlay
    axes[1].imshow(img_np)
    axes[1].imshow(cams, cmap="jet", alpha=0.5)
    axes[1].set_title(f"Grad-CAM\nTrue: {true_label}, Pred: {pred_label}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
    
    return cams, img_np

def fusion_heatmap(img_np, cams, ig_attr_processed):
    """Fuses Grad-CAM and Integrated Gradients heatmaps together."""
    # Resize GradCAM heatmap to match input image resolution (H, W)
    cams_resized = cv2.resize(cams, (ig_attr_processed.shape[1], ig_attr_processed.shape[0]))

    # Normalize GradCAM
    gc = (cams_resized - cams_resized.min()) / (cams_resized.max() - cams_resized.min() + 1e-8)

    # Normalize IG
    ig = (ig_attr_processed - ig_attr_processed.min()) / (ig_attr_processed.max() - ig_attr_processed.min() + 1e-8)

    # Fusion heatmap (combined)
    fusion = gc * ig
    fusion = (fusion - fusion.min()) / (fusion.max() - fusion.min() + 1e-8)

    plt.figure(figsize=(6,6))
    plt.imshow(img_np)
    plt.imshow(fusion, cmap="jet", alpha=0.45)
    plt.title("Fusion Grad-CAM + Integrated Gradients")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate XAI Explanations using SmallCNN")
    parser.add_argument("--image", type=str, required=True, help="Path to the target leaf image")
    parser.add_argument("--weights", type=str, required=True, help="Path to the saved SmallCNN .pth weights")
    parser.add_argument("--true_class", type=int, default=0, help="True class label index")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of output classes in the model")
    
    args = parser.parse_args()
    
    print(f"Executing XAI Framework on device: {DEVICE}")
    
    # 1. Load SmallCNN model
    model = SmallCNN(num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.weights, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # The final convolution layer
    target_layer = model.conv3
    
    # 2. Load and preprocess the image
    img_tensor = load_image(args.image)
    img_tensor_batch = img_tensor.unsqueeze(0).to(DEVICE)
    
    # 3. Get the Model Prediction
    with torch.no_grad():
        output = model(img_tensor_batch)
        pred_class = output.argmax(dim=1).item()
        
    print(f"True Label: {args.true_class} | Predicted Label: {pred_class}")
    
    # 4. Generate and Visualize Grad-CAM
    cams, img_np = gradcam_visualize(model, img_tensor, args.true_class, pred_class, target_layer)
    
    # 5. Generate and Visualize Integrated Gradients
    img_tensor_ig = img_tensor_batch.clone().requires_grad_(True)
    ig_attr = integrated_gradients(model, img_tensor_ig, pred_class)
    ig_processed = overlay_integrated_gradients(img_tensor, ig_attr)
    
    # 6. Generate and Visualize Fusion Heatmap
    fusion_heatmap(img_np, cams, ig_processed)