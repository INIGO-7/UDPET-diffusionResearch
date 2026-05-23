import torchvision.transforms.functional as TF
from torchvision import transforms

from config import config


class PadToSquare:
    """Pad image to a square with white fill, centering the content."""
    def __init__(self, size, fill=255):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        pad_w = self.size - w
        pad_h = self.size - h
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
        return TF.pad(img, [left, top, right, bottom], fill=self.fill)


preprocess = transforms.Compose(
    [
        PadToSquare(config.image_size, fill=255),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}


def transform_ds(dataset):
    dataset.set_transform(transform)
    return dataset
