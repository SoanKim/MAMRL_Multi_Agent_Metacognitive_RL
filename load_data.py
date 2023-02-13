import torch.utils.data as data
from environment import TripletDataset

# Generate stimuli
n_prb = 5000
train_ratio = 0.5
validate_ratio = 0.3
test_ratio = 0.2

train_starting_idx = 0
train_ending_idx = 3000 * train_ratio

validate_starting_idx = train_ending_idx
validate_ending_idx = 3000 * (train_ratio + validate_ratio)

test_starting_idx = validate_ending_idx
test_ending_idx = 3000

train_loader = data.DataLoader(TripletDataset(n_prb, train_starting_idx, train_ending_idx),
                               batch_size=10, shuffle=False, drop_last=True, pin_memory=True)

val_loader = data.DataLoader(TripletDataset(n_prb, validate_starting_idx, validate_ending_idx),
                             batch_size=10, shuffle=False, drop_last=True, pin_memory=True)

test_loader = data.DataLoader(TripletDataset(n_prb, test_starting_idx, test_ending_idx),
                              batch_size=10, shuffle=False, drop_last=True, pin_memory=True)

