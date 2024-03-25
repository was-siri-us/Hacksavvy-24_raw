from flask import Flask, request, jsonify, send_file
import os
import PIL
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp

app = Flask(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image sent'}), 400

    img = request.files['image']
    pil_img = PIL.Image.open(img)

    # Convert the image to RGB format and resize it
    pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize((224, 224))

    torch_img = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])(pil_img).to(device)
    torch_img = torch.unsqueeze(torch_img, 0)
    normed_torch_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)

    densenet = models.densenet161(pretrained=True)
    densenet.load_state_dict(torch.load("your.pth.tar"), strict=False)

    configs = [
        dict(model_type='densenet', arch=densenet, layer_name='features_norm5'),
    ]

    for config in configs:
        config['arch'].to(device).eval()

    cams = [
        [cls.from_config(**config) for cls in (GradCAM, GradCAMpp)]
        for config in configs
    ]

    images = []
    for gradcam, gradcam_pp in cams:
        mask, _ = gradcam(normed_torch_img)
        heatmap, result = visualize_cam(mask, torch_img)

        mask_pp, _ = gradcam_pp(normed_torch_img)
        heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)

        # Remove the batch dimension for visualization tensors
        heatmap = heatmap.squeeze(0)
        result = result.squeeze(0)
        heatmap_pp = heatmap_pp.squeeze(0)
        result_pp = result_pp.squeeze(0)

        # Remove the batch dimension for input image tensor
        torch_img_cpu = torch_img.cpu().squeeze(0)

        images.extend([torch_img_cpu, heatmap, heatmap_pp, result, result_pp])

    grid_image = make_grid(images[3:], nrow=5)
    save_image(grid_image, 'grid_image.png')

    return send_file('grid_image.png', mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
