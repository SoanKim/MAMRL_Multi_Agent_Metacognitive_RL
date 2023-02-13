import numpy as np
import itertools
from collections import Counter

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

n_prb = 100

# Making problems
problem_mat = np.zeros((n_prb, 5, 4))
answer_attribute = []
non_answer_attribute = []

for random_value in range(n_prb * 10):
    random_attribute = np.random.choice(list(range(3)), 3, replace=True)
    if np.sum(random_attribute) % 3 == 0:
        answer_attribute.append(random_attribute)
    else:
        non_answer_attribute.append(random_attribute[:-1])

answer_li = []
for prb in range(n_prb):
    triplet = sorted(np.random.choice(list(range(5)), 3, replace=False))
    three_answers = sorted([list(range(5))[i] for i in triplet])
    answer_li.append(tuple(three_answers))

    for attribute in range(4):
        problem_mat[prb, [three_answers], attribute] = answer_attribute[np.random.choice(list(range(len(answer_attribute))))]
        problem_mat[prb, list(set(list(range(5)))-set(three_answers)), attribute] = non_answer_attribute[np.random.choice(list(range(len(non_answer_attribute))))]

# Checking Duplicates
combi = list(itertools.combinations(list(range(5)), 3))
duplicated = []
for prb in range(n_prb):
    candidates = []
    for c in combi:
        for attribute in range(4):
            three_values = problem_mat[prb, list(c), attribute]
            if np.sum(three_values) % 3 == 0:
                candidates.append(c)

    counts = Counter(candidates).most_common(2)

    if counts[0][1] == counts[1][1]:
        duplicated.append(prb)

problem_mat = np.delete(problem_mat, duplicated, axis=0)
answer_li = [elem for i, elem in enumerate(answer_li) if i not in duplicated]

# One-Hot
# one_hot_problem_mat = np.zeros((problem_mat.shape[0], 5, 12))
# for prb in range(problem_mat.shape[0]):
#     for card in range(5):
#         for attribute in range(4):
#             arr = np.array(problem_mat[prb, card].astype(int))
#             one_hot_attribute = np.eye(3, dtype=np.int32)[arr].flatten()
#             one_hot_problem_mat[prb, card, :] = one_hot_attribute


