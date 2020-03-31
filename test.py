import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim

import numpy as np
import random

from make_pairs import *


def oneshot(model, testset):
    # 学習していない 8,9 の画像について、1枚ずつサンプル画像をとってきて、8と9 の画像を分類させる。 (デタラメだと、accuracy 0.5)
    model.eval()
    num_test = 700  # 700 images for each class
    imgs = categorize_data(testset, num_test, "test")
    with torch.no_grad():
        num_accurate = np.array([0, 0])
        # 8と9のサンプル画像を 1枚ずつとってきて、testset の8,9 のそれぞれの画像がどちらに近いかを判断する。
        sample_img_8 = imgs[8][0]
        # print(sample_img_8)
        sample_img_9 = imgs[9][0]
        for i in range(len(imgs[8]) - 1):
            for j in range(8, 10):
                p_8 = model(imgs[j][i], sample_img_8)  # 8のサンプル画像 との近さ
                p_9 = model(imgs[j][i], sample_img_9)  # 9のサンプル画像 との近さ
                pred = 8 + np.argmax([p_8, p_9], axis=0)  # どちらに近いか
                # print(pred)

                if pred == j:
                    # print("accurate")
                    num_accurate[j - 8] += 1

    accuracy = num_accurate / num_test
    print("accuracy when '8' is input: {}, accuracy when '9' is input: {}".format(
        accuracy[0], accuracy[1]))
    return accuracy


# ブレが大きいので、とりあえずaccuracyを10回出したあとに平均をとった。
def oneshot_mean(model, testset):
    print("start one-shot learning for '8','9' images \n \
'8'.'9' were not used for training \n only 1 teacher data for each class will be used")

    accuracy_mean = np.array([0.0, 0.0])
    for i in range(10):
        accuracy = oneshot(model, testset)
        accuracy_mean += accuracy
    accuracy_mean /= 10
    print("average accuracy")
    print(accuracy_mean)
    return accuracy_mean