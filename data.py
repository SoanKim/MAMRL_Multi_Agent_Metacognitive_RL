import numpy as np
from collections import Counter

n_prb = 1000

problem_mat = np.zeros((5, 4, 3))


problem_mat = np.zeros((n_prb, 5, 4, 3))
answer_attribute = []
non_answer_attribute = []

for random_value in range(3000):

    random_attribute = np.random.choice(list(range(3)), 3, replace=True)

    if np.sum(random_attribute) % 3 == 0:
        answer_attribute.append(random_attribute)
    else:
        non_answer_attribute.append(random_attribute)


answer_li = []
for prb in range(n_prb):

    triplet = sorted(np.random.choice(list(range(5)), 3, replace=False))
    three_answers = sorted([list(range(5))[i] for i in triplet])
    answer_li.append(tuple(three_answers))
    # two_distractors = sorted([list(range(5))[item] for item in list(range(5)) if item not in triplet])
    # print("answers", three_answers)
    for card in range(5):
        for attribute in range(4):
            for value in range(3):
                if card in three_answers:
                    problem_mat[prb, card, attribute] = answer_attribute[np.random.choice(list(range(len(answer_attribute))))]
                elif card not in three_answers:
                    problem_mat[prb, card, attribute] = non_answer_attribute[np.random.choice(list(range(len(non_answer_attribute))))]

print(Counter(answer_li))