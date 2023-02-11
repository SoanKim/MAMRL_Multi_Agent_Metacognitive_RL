import numpy as np
from collections import Counter

n_prb = 1
def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def answer_gen(n_prb):
    answer_li = []
    for prb in range(n_prb):
        temp_li = []
        answer = np.random.choice(list(range(5)), 3, replace=False)
        for num in answer:
            temp_li.append(num)
        answer_li.append(tuple(sorted(temp_li)))
    return answer_li

answer = answer_gen(n_prb)
print(answer)

total_mat = np.zeros((5 *4 *3))

# #def problem_gen(n_prb):
# total_prb = np.zeros((n_prb, 5, 4, 3))
# for prb in range(n_prb):
#     one_prb_li = []
#     for card in range(5):
#         one_att_li = []
#         for attribute in range(4):
#             value = np.random.choice(list(range(3)))
#             print("####", prb, card, attribute, value)
#             value = one_hot(value, 3)
#             total_prb[prb, card, attribute] = value
#
# print(total_prb)

