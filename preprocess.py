from torchvision import transforms
import albumentations as A
import albumentations.pytorch


transforms_train = A.Compose([
    # A.ElasticTransform(p=0.5),
    # A.GridDistortion(p=0.5),
    A.Resize(640, 640),
    A.RandomRotate90(p=0.5),
    A.Normalize(),
    A.pytorch.transforms.ToTensor()
])

transforms_test = A.Compose([
    A.Normalize(),
    A.pytorch.transforms.ToTensor()
])