import torch
import torchvision.transforms.functional as TF
from torchvision.io import read_image
import matplotlib.pyplot as plt

# Load image (C, H, W) and normalize
img = read_image("./irl-images/corner-setup.jpg").float() / 255.0

# 🔄 Rotate 90° clockwise (i.e., counter the left rotation)

# Resize to 3024 x 3024
img_resized = TF.resize(img, size=[3072, 3072], antialias=True)

img_rotated = TF.rotate(img_resized, angle=-90)
# Convert to HWC format for display
img_disp = img_rotated.permute(1, 2, 0).clamp(0, 1).cpu().numpy()

# Show image
plt.imshow(img_disp)
plt.axis("off")
plt.title("Corrected Orientation (3072x3072)")
plt.show()
