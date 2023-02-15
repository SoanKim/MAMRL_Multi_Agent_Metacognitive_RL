import torch.utils.data as data
from environment import TripletDataset

# Generate stimuli
n_prb = 10
train_ratio = 0.5
validate_ratio = 0.3
test_ratio = 0.2

train_starting_idx = 0
train_ending_idx = 9 * train_ratio

validate_starting_idx = train_ending_idx
validate_ending_idx = 9 * (train_ratio + validate_ratio)

test_starting_idx = validate_ending_idx
test_ending_idx = 9

train_loader = data.DataLoader(TripletDataset(n_prb, train_starting_idx, train_ending_idx),
                               batch_size=10, shuffle=False, drop_last=True, pin_memory=True)

val_loader = data.DataLoader(TripletDataset(n_prb, validate_starting_idx, validate_ending_idx),
                             batch_size=10, shuffle=False, drop_last=True, pin_memory=True)

test_loader = data.DataLoader(TripletDataset(n_prb, test_starting_idx, test_ending_idx),
                              batch_size=10, shuffle=False, drop_last=True, pin_memory=True)


import PIL
import numpy as np
import matplotlib.pyplot as plt

# Load sample image
from PIL import Image
from matplotlib import cm
import torchvision.transforms as T
from skimage import io
import torchvision
data = TripletDataset(10, 0, 2)
print("data", data[0][0])
print("answer", data[0][1])
import itertools
combi = list(itertools.combinations(list(range(5)), 3))
print(combi)
for i, each_card in enumerate(data[0][0]):

    each_card = np.array(each_card)

    img = Image.fromarray((each_card * 255).astype(np.uint8))
    plt.matshow(img)
    plt.title(str(i))
    plt.show()


    # image = torchvision.transforms.ToPILImage()(each_card.unsqueeze(0))
    # return_image = io.BytesIO()
    # image.save(return_image, "JPEG")
    # return_image.seek(0)





