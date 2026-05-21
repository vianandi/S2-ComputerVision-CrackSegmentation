import matplotlib
matplotlib.use('Agg')  # Set backend untuk non-interactive environment
import matplotlib.pyplot as plt
import torch
import os
from utils.model_output import extract_logits

def visualize_prediction(model, dataset, device, idx=0, threshold=0.5, save_path='results'):
    model.eval()

    image, mask = dataset[idx]

    image = image.unsqueeze(0).to(device)
    mask = mask.squeeze().cpu().numpy()

    with torch.no_grad():
        output = model(image)
        prob = torch.sigmoid(extract_logits(output))
        pred = (prob > threshold).float()

    image = image.squeeze().permute(1, 2, 0).cpu().numpy()
    pred = pred.squeeze().cpu().numpy()
    prob = prob.squeeze().cpu().numpy()

    print("Pred min/max:", pred.min(), pred.max())
    print("Mask min/max:", mask.min(), mask.max())
    print("Prob min/max:", prob.min(), prob.max())

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(pred, cmap="gray")
    plt.title(f"Prediction (thr={threshold})")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(prob, cmap="hot")
    plt.title("Probability Map")
    plt.axis("off")

    plt.tight_layout()
    
    # Create save directory if not exists
    os.makedirs(save_path, exist_ok=True)
    output_file = os.path.join(save_path, f'prediction_idx{idx}_thr{threshold}.png')
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    print(f"Saved visualization to: {output_file}")
    plt.close()