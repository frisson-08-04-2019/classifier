from PIL import Image
import torchvision


class MakeSquare:
    def __init__(self, size=224, fill_color=(255, 255, 255)):
        self.size = size
        self.fill_color = fill_color

    def __call__(self, sample):
        sample.thumbnail((self.size, self.size), Image.ANTIALIAS)
        x, y = sample.size
        new_img = Image.new('RGB', (self.size, self.size), self.fill_color)
        new_img.paste(sample, ((self.size - x) // 2, (self.size - y) // 2))
        return new_img

    def __repr__(self):
        return self.__class__.__name__ + '(size=%d, fill_color=%s)' % (self.size, str(self.fill_color))


normalize = torchvision.transforms.Compose(
    [
        MakeSquare(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
