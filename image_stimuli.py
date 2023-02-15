from load_data import TripletDataset

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import itertools

data = TripletDataset(10, 0, 2)
print("data", data[0][0])
print("answer", data[0][1])

combi = list(itertools.combinations(list(range(5)), 3))
print(combi)
for i, each_card in enumerate(data[0][0]):

    each_card = np.array(each_card)

    img = Image.fromarray((each_card * 255).astype(np.uint8))
    plt.matshow(img)
    plt.title(str(i))
    plt.show()