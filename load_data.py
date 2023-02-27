import torch.utils.data as data
from environment import TripletDataset
import os
import glob
import torch

saving_dir = os.path.join(os.getcwd())

train_data = glob.glob(os.path.join(os.getcwd(), "img_data", "*train.png"))
train_loader = torch.utils.data.DataLoader(train_data)

val_data = glob.glob(os.path.join(os.getcwd(), "img_data", "*validation.png"))
val_loader = torch.utils.data.DataLoader(val_data)

test_data = glob.glob(os.path.join(os.getcwd(), "img_data", "*test.png"))
test_loader = torch.utils.data.DataLoader(test_data)

# Text Version
# train_loader = data.DataLoader(TripletDataset(n_prb, train_starting_idx, train_ending_idx),
#                                batch_size=10, shuffle=False, drop_last=True, pin_memory=True)
#
# val_loader = data.DataLoader(TripletDataset(n_prb, validate_starting_idx, validate_ending_idx),
#                              batch_size=10, shuffle=False, drop_last=True, pin_memory=True)
#
# test_loader = data.DataLoader(TripletDataset(n_prb, test_starting_idx, test_ending_idx),
#                               batch_size=10, shuffle=False, drop_last=True, pin_memory=True)

