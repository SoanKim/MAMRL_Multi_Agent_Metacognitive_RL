from load_data import TripletDataset

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import itertools

n_prb = 10
data = TripletDataset(n_prb, 0, 2)
print("Number of stimuli", len(data))


combi = list(itertools.combinations(list(range(5)), 3))

for i_prb, each_card in enumerate(data):
    answer = combi[data[int(i_prb)][1]]
    each_card = np.array(data[i_prb][0])
    img = Image.fromarray((each_card * 255).astype(np.uint8))

    plt.matshow(img)
    plt.title("Problem Number "+str(i_prb)+" (Answer "+str(answer)+")")
    plt.show()