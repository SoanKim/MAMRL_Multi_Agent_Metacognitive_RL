from load_data import TripletDataset
import os
import numpy as np
import json

from PIL import Image
import itertools
# Generate stimuli

# Hyper parameters for generating data
n_prb = 1000
train_ratio = 0.5
validate_ratio = 0.3
test_ratio = 0.2

train_starting_idx = 0
train_ending_idx = n_prb*0.9 * train_ratio

validate_starting_idx = train_ending_idx
validate_ending_idx = n_prb*0.9 * (train_ratio + validate_ratio)

test_starting_idx = validate_ending_idx
test_ending_idx = n_prb*0.9

data = TripletDataset(n_prb, 0, n_prb*0.9)
print("Number of stimuli: ", len(data))

saving_dir = os.path.join(os.getcwd())
combi = list(itertools.combinations(list(range(5)), 3))

# Scalar labels
for i in list(combi):
    combi.append(tuple(i))
combi_dic = dict(zip(combi, list(range(10))))

if not os.path.exists('img_data'):
    os.makedirs('img_data/')

# Saving stimuli
label_file = {"train": [], "validation": [], "test": []}

for i_prb, each_card in enumerate(data):
    answer = combi[data[int(i_prb)][1]]
    each_card = np.array(data[i_prb][0])
    # print(each_card)

    img = Image.fromarray((each_card * 255).astype(np.uint8), 'RGB')
    scalar_answer = combi_dic[answer]

    # Divide the training, validation, and test
    if i_prb < validate_starting_idx:
        file_name = "Problem_" + str(i_prb).zfill(3) + "_Answer_" + str(answer[0]) + str(answer[1]) + str(answer[2])
        img.save("img_data/" + file_name + '_train.png')
        label_file['train'].append(scalar_answer)

    elif validate_starting_idx < i_prb < test_starting_idx:
        file_name = "Problem_" + str(i_prb).zfill(3) + "_Answer_" + str(answer[0]) + str(answer[1]) + str(answer[2])
        img.save("img_data/" + file_name + '_validation.png')
        label_file['validation'].append(scalar_answer)

    else:
        file_name = "Problem_" + str(i_prb).zfill(3) + "_Answer_" + str(answer[0]) + str(answer[1]) + str(answer[2])
        img.save("img_data/" + file_name + '_test.png')
        label_file['test'].append(scalar_answer)

with open("img_data/labels.json", "w") as output_file:
    json.dump(label_file, output_file)
    #label_file.write("%s\n" % str(scalar_answer))

    # img_arr = np.array(img)
    # img_arr = np.asarray(img)
    # print(img_arr)

    # plt.matshow(img)
    # plt.title("Problem Number "+str(i_prb)+" (Answer "+str(answer)+")")
    # plt.show()
