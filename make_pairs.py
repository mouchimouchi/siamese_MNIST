import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim

import numpy as np
import random


def categorize_data(dataset, num_each, mode):  # ラベルごとにnum_each ずつとってくる
    count = [0 for i in range(10)]
    if mode == "train":
        shuffle_index = list(range(10000))  # 0~10000 for training
    elif mode == "valid":
        shuffle_index = list(range(10001, 20000))  # 10001~20000 for validation
    elif mode == "test":
        shuffle_index = list(range(10000))  # MNIST のテスト用データは10000 しかない。
    else:
        print("mode invalid")
        return

    random.shuffle(shuffle_index)
    # print(shuffle_index)
    i = 0
    img_list = [[] for i in range(10)]
    while True:
        label = dataset[shuffle_index[i]][1]
        img_tensor = dataset[shuffle_index[i]][0]
        i += 1
        if count[label] < num_each:
            img_list[label].append(img_tensor)
            count[label] += 1

        if min(count) >= num_each:
            break
    return img_list


def make_pairs_index(num):
    all_pairs = []
    for i in range(num - 1):
        for j in range(i + 1, num):
            all_pairs.append([i, j])
    return all_pairs


def make_same_pairs(img_list):
    num_each = len(img_list[0])
    same_list_index = [[] for i in range(10)]
    same_list = [[] for i in range(10)]

    for i in range(10):
        index = make_pairs_index(num_each)
        # pair_index = [] #for debug
        pair = []
        for j in range(len(index)):
            # pair_index.append([index[j][0],index[j][1]]) #for debug
            pair.append([img_list[i][index[j][0]], img_list[i][index[j][1]]])
        # same_list_index[i] = pair_index  #for debug
        same_list[i] = pair

        # print(same_list_index)
    same_list_8 = []
    # same_list_8_index = [] #for debug
    for i in range(8):  # 1 shot learnig のため、8,9 のデータは、なかったことにする。
        same_list_8.extend(same_list[i])
        # same_list_8_index.extend(same_list_index[i]) #for debug
    # print(same_list_8_index)
    return same_list_8


def make_different_pairs(img_list):
    num_each = len(img_list[0])
    num_pairs = 8 * num_each * (num_each - 1) / 2  # same_list と同じ数になるように。
    num_pairs = int(num_pairs)

    pairs = []
    for i in range(num_pairs):
        category1 = random.randint(0, 9)
        category2 = random.randint(0, 9)
        while category1 == category2:
            category2 = random.randint(0, 9)

        img1 = random.choice(img_list[category1])
        img2 = random.choice(img_list[category2])
        pairs.append([img1, img2])

    return pairs


def make_pairloader(dataset, num_each, mode):
    img_list = categorize_data(dataset, num_each, mode)
    img_size = img_list[0][0].size()
    # print(img_list)
    same_pairs = make_same_pairs(img_list)

    different_pairs = make_different_pairs(img_list)

    # print(same_pairs[0])
    # print(different_pairs[0])
    assert len(same_pairs) == len(different_pairs)

    tensor_pairs_data = torch.zeros([len(same_pairs) * 2, 2, img_size[0],
                                     img_size[1], img_size[2]])
    # number of pairs (same + different),2,channel,height,width
    pairs_label = []

    for i in range(len(same_pairs)):
        tensor_pairs_data[2 * i][0] = same_pairs[i][0]
        tensor_pairs_data[2 * i][1] = same_pairs[i][1]
        pairs_label.append(1.0)  # 1 for same_pair
        tensor_pairs_data[2 * i + 1][0] = different_pairs[i][0]
        tensor_pairs_data[2 * i + 1][1] = different_pairs[i][1]
        pairs_label.append(0.0)  # 0 for different_pair

    # print(tensor_pairs_data.size())
    tensor_pairs_label = torch.tensor(pairs_label)
    # print(tensor_pairs_label.size())

    pairset = torch.utils.data.TensorDataset(
        tensor_pairs_data, tensor_pairs_label)
    pairloader = torch.utils.data.DataLoader(
        pairset, batch_size=4, shuffle=False, num_workers=2)

    return pairset, pairloader