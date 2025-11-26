import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import wiener
from scipy.ndimage import uniform_filter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def add_gaussian_noise(image, mean=0, sigma=50):
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy = np.clip(image + gauss, 0, 255).astype(np.uint8)
    return noisy

def custom_identity(image):
    kernel = np.zeros((3,3), np.float32)
    kernel[1, 1] = 1.0
    return cv2.filter2D(image, -1, kernel)

def custom_lowpass(image):
    kernel = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]], np.float32) / 10
    return cv2.filter2D(image, -1, kernel)

def anisotropic_diffusion(image, num_iter=20, kappa=15, gamma=0.2):
    img = image.astype(np.float32)
    for i in range(num_iter):
        for channel in range(img.shape[2]):
            nablaN = np.roll(img[:,:,channel], -1, axis=0) - img[:,:,channel]
            nablaS = np.roll(img[:,:,channel], +1, axis=0) - img[:,:,channel]
            nablaE = np.roll(img[:,:,channel], -1, axis=1) - img[:,:,channel]
            nablaW = np.roll(img[:,:,channel], +1, axis=1) - img[:,:,channel]
            cN = np.exp(-(nablaN / kappa)**2)
            cS = np.exp(-(nablaS / kappa)**2)
            cE = np.exp(-(nablaE / kappa)**2)
            cW = np.exp(-(nablaW / kappa)**2)
            img[:,:,channel] += gamma * (cN*nablaN + cS*nablaS + cE*nablaE + cW*nablaW)
    return np.clip(img, 0, 255).astype(np.uint8)

image = cv2.imread('peppers.png')
if image is None:
    image = cv2.imread('image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

noisy = add_gaussian_noise(image, sigma=50)
average_blur = cv2.blur(noisy, (13, 13))
gaussian_blur = cv2.GaussianBlur(noisy, (13, 13), 4.0)
uniform_blur = uniform_filter(noisy, size=(13,13,1))
wiener_blur = np.zeros_like(noisy)
for ch in range(3):
    wiener_blur[:,:,ch] = np.clip(wiener(noisy[:,:,ch], mysize=(13,13)), 0, 255)
identity = custom_identity(noisy)
lowpass = custom_lowpass(noisy)
anisodiff = anisotropic_diffusion(noisy, num_iter=20, kappa=15, gamma=0.2)

def quality_metrics(base, filtered):
    psnr = peak_signal_noise_ratio(base, filtered)
    ssim = structural_similarity(base, filtered, channel_axis=2)
    return psnr, ssim

filters = [
    ("Original", image_rgb),
    (f"Noisy (σ=50)", cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)),
    ("Average (Box 13x13)", cv2.cvtColor(average_blur, cv2.COLOR_BGR2RGB)),
    ("Gaussian (13x13, σ=4.0)", cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2RGB)),
    ("Uniform (Mean 13x13)", cv2.cvtColor(uniform_blur, cv2.COLOR_BGR2RGB)),
    ("Wiener (Adaptive 13x13)", cv2.cvtColor(wiener_blur, cv2.COLOR_BGR2RGB)),
    ("Identity", cv2.cvtColor(identity, cv2.COLOR_BGR2RGB)),
    ("Custom Lowpass", cv2.cvtColor(lowpass, cv2.COLOR_BGR2RGB)),
    ("Anisotropic Diffusion (20 iter)", cv2.cvtColor(anisodiff, cv2.COLOR_BGR2RGB)),
]

ncols = 3
nrows = int(np.ceil(len(filters) / ncols))

plt.figure(figsize=(16, nrows * 5))
for idx, (name, img) in enumerate(filters):
    if name == "Original":
        psnr_str = "-"
        ssim_str = "-"
    else:
        psnr, ssim = quality_metrics(image_rgb, img)
        psnr_str = f"{psnr:.2f}"
        ssim_str = f"{ssim:.3f}"
    ax = plt.subplot(nrows, ncols, idx + 1)
    ax.imshow(img)
    ax.set_title(f"{name}\nPSNR={psnr_str} SSIM={ssim_str}", fontsize=14)
    ax.axis('off')
plt.tight_layout()
plt.show()
