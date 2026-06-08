import torch
import numpy as np
import cv2
import random
from torchvision import transforms


class ElasticTransform:
    def __init__(self, alpha=34.0, sigma=4.0, p=0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img

        if img.dim() == 3:
            img_np = img.squeeze(0).cpu().numpy()
            need_unsqueeze = True
        else:
            img_np = img.cpu().numpy()
            need_unsqueeze = False

        h, w = img_np.shape
        dx = np.random.rand(h, w).astype(np.float32) * 2 - 1
        dy = np.random.rand(h, w).astype(np.float32) * 2 - 1
        dx = cv2.GaussianBlur(dx, (0, 0), self.sigma) * self.alpha
        dy = cv2.GaussianBlur(dy, (0, 0), self.sigma) * self.alpha

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        transformed = cv2.remap(
            img_np, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        result = torch.from_numpy(transformed).float()
        if need_unsqueeze:
            result = result.unsqueeze(0)
        return result


class RandomErasing:
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0.0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img

        img = img.clone()
        if img.dim() == 3:
            img_2d = img.squeeze(0)
            need_unsqueeze = True
        else:
            img_2d = img
            need_unsqueeze = False

        h, w = img_2d.shape
        area = h * w

        for _ in range(20):
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)
            rw = int(np.sqrt(target_area * aspect_ratio))
            rh = int(np.sqrt(target_area / aspect_ratio))

            if rw <= w and rh <= h:
                x = random.randint(0, h - rh)
                y = random.randint(0, w - rw)
                img_2d[x:x + rh, y:y + rw] = self.value
                break

        if need_unsqueeze:
            img_2d = img_2d.unsqueeze(0)
        return img_2d


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.05, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img
        noise = torch.randn_like(img) * self.std + self.mean
        return torch.clamp(img + noise, 0.0, 1.0)


def get_train_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(degrees=15, fill=0),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=0),
        ElasticTransform(alpha=34, sigma=4, p=0.5),
        RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0.0),
        AddGaussianNoise(std=0.05, p=0.5),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
    ])


def get_tta_transforms():
    base_norm = transforms.Normalize((0.1307,), (0.3081,))
    return [
        transforms.Compose([transforms.ToTensor(), base_norm]),
        transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=1.0),
            base_norm
        ]),
        transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=5, fill=0),
            base_norm
        ]),
        transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            base_norm
        ]),
    ]