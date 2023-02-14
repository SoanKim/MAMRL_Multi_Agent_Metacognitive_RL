import numpy as np
import itertools
import torch
from collections import Counter
import torch.utils.data as data


class TripletDataset(data.Dataset):

    def __init__(self, n_prb, starting_idx, ending_idx):
        super(TripletDataset, self).__init__()
        self.n_prb = n_prb
        self.starting_idx = int(starting_idx)
        self.ending_idx = int(ending_idx)

        problem_mat = np.zeros((self.n_prb, 5, 4))
        answer_attribute = []
        non_answer_attribute = []

        for random_value in range(self.n_prb * 10):
            random_attribute = np.random.choice(list(range(3)), 3, replace=True)
            if np.sum(random_attribute) % 3 == 0:
                answer_attribute.append(random_attribute)
            else:
                non_answer_attribute.append(random_attribute[:-1])

        answer_li = []
        for prb in range(self.n_prb):
            triplet = sorted(np.random.choice(list(range(5)), 3, replace=False))
            three_answers = sorted([list(range(5))[i] for i in triplet])
            answer_li.append(tuple(three_answers))

            for attribute in range(4):
                problem_mat[prb, [three_answers], attribute] = answer_attribute[
                    np.random.choice(list(range(len(answer_attribute))))]
                problem_mat[prb, list(set(list(range(5))) - set(three_answers)), attribute] = non_answer_attribute[
                    np.random.choice(list(range(len(non_answer_attribute))))]

        # Checking Duplicates
        combi = list(itertools.combinations(list(range(5)), 3))
        duplicated = []
        for prb in range(self.n_prb):
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
        one_hot_problem_mat = np.zeros((problem_mat.shape[0], 5, 12))
        for prb in range(problem_mat.shape[0]):
            for card in range(5):
                for attribute in range(4):
                    arr = np.array(problem_mat[prb, card].astype(int))
                    one_hot_attribute = np.eye(3, dtype=np.int32)[arr].flatten()
                    one_hot_problem_mat[prb, card, :] = one_hot_attribute

        print(one_hot_problem_mat.shape)
        env = torch.Tensor(one_hot_problem_mat)
        labels = torch.Tensor(answer_li)

        self.env = env[self.starting_idx:self.ending_idx, :, :]
        self.labels = labels[self.starting_idx:self.ending_idx]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        states = self.env[idx]
        labels = self.labels[idx]

        return states, labels
