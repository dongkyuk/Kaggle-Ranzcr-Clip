import torch
import pandas as pd
import numpy as np

from torchinfo import summary
from torch.utils.data import DataLoader

from model import MnistModel
from dataset import MnistDataset
from preprocess import transforms_train, transforms_test
from tta import TTA

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Get Model
model0 = MnistModel().to(device)
model0.load_state_dict(torch.load('data/fold-0_best.pth', map_location=device))

model1 = MnistModel().to(device)
model1.load_state_dict(torch.load('data/fold-1_best.pth', map_location=device))

model2 = MnistModel().to(device)
model2.load_state_dict(torch.load('data/fold-2_best.pth', map_location=device))

model3 = MnistModel().to(device)
model3.load_state_dict(torch.load('data/fold-3_best.pth', map_location=device))

model4 = MnistModel().to(device)
model4.load_state_dict(torch.load('data/fold-4_best.pth', map_location=device))



model0.eval()
model1.eval()
model2.eval()
model3.eval()
model4.eval()

# Prepare Data
submit = pd.read_csv('data/sample_submission.csv')
testset = MnistDataset(
    'data/test', 'data/sample_submission.csv', transforms_test)
test_loader = DataLoader(testset, batch_size=8, num_workers=4)

# Test time augmentation
conf = '{"augs":["NO",\
                "ROT90",\
                "ROT180",\
                "ROT270"],\
        "mean":"ARITH"}'
model0 = TTA(model0, device, conf)
model1 = TTA(model1, device, conf)
model2 = TTA(model2, device, conf)
model3 = TTA(model3, device, conf)
model4 = TTA(model4, device, conf)


# Inference
batch_size = test_loader.batch_size
batch_index = 0
for i, (images, targets) in enumerate(test_loader):
    # images.to(device)
    images = images.permute(0, 2, 3, 1)
    images = images.numpy()

    outputs0 = model0.predict_images(images)
    outputs1 = model1.predict_images(images)
    outputs2 = model2.predict_images(images)
    outputs3 = model3.predict_images(images)
    outputs4 = model4.predict_images(images)

    outputs = (outputs0 + outputs1 + outputs2 + outputs3 + outputs4) / 5

    outputs = outputs > 0.4
    outputs = torch.tensor(outputs)
    batch_index = i * batch_size
    submit.iloc[batch_index:batch_index+batch_size, 1:] = \
        outputs.long().squeeze(0).detach().cpu().numpy()

# Make submission file
submit.to_csv('data/submit-fold-all-0.4.csv', index=False)
