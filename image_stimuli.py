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

five_cards = np.zeros((5, 4, 3))
for i, each_card in enumerate(data[0][0]):
    each_card = np.array(each_card)
    five_cards[i] = each_card
    img = Image.fromarray((each_card * 255).astype(np.uint8))
    plt.matshow(img)
    plt.title(str(i))
    plt.show()

five_cards = np.transpose(five_cards, (1, 0, 2))
five_cards = np.reshape(five_cards, (4, 15))
plt.matshow(five_cards)
plt.show()
# img = Image.fromarray((five_cards * 255).astype(np.uint8))
# plt.matshow(img)
# #plt.title(str(i))
# plt.show()