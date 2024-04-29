import pickle
import sys

import keras
import numpy as np

def createOneHot(train_label, test_label):
    maxlen = int(max(train_label.max(), test_label.max()))

    train = np.zeros((train_label.shape[0], train_label.shape[1], maxlen + 1))
    test = np.zeros((test_label.shape[0], test_label.shape[1], maxlen + 1))

    for i in range(train_label.shape[0]):
        for j in range(train_label.shape[1]):
            train[i, j, train_label[i, j]] = 1

    for i in range(test_label.shape[0]):
        for j in range(test_label.shape[1]):
            test[i, j, test_label[i, j]] = 1

    return train, test

def createOneHotMosei3way(train_label, test_label):
    one = 0
    two = 0
    three = 0
    print("调用createOneHotMosei3way函数")
    print(train_label.shape)    # (2250, 98, 3)
    print(test_label.shape)     # (678, 98, 3)
    maxlen = 2

    train = np.zeros((train_label.shape[0], train_label.shape[1], maxlen + 1))
    print(train_label.shape)  # (2250, 98, 3)
    test = np.zeros((test_label.shape[0], test_label.shape[1], maxlen + 1))
    print(test.shape)        # (678, 98, 3)

    for i in range(train_label.shape[0]):   # 2250
        pass
        # for j in range(train_label.shape[1]):  # 98
        #     # print(type(train_label[i, j]), train_label[i,j])
        #     # print(train_label[i, j].any())
        #     # print("11111")
        #     if train_label[i, j].any() > 0:   # 对应每一个值
        #         train[i, j, 1] = 1
        #         one = one + 1
        #     else:
        #         if train_label[i, j].any() < 0:
        #             train[i, j, 0] = 1
        #             # print(train_label[i, j].any())
        #             # print("-111111")
        #             two = two + 1
        #         else:
        #             if train_label[i, j].any() == 0:
        #                 train[i, j, 2] = 1
        #                 three = three + 1

    for i in range(test_label.shape[0]):
        for j in range(test_label.shape[1]):
            print(test_label[i, j])
            if test_label[i, j].any() > 0:
                test[i, j, 1] = 1
            else:
                if test_label[i, j].any() < 0:
                    test[i, j, 0] = 1
                else:
                    if test_label[i, j].any() == 0:
                        test[i, j, 2] = 1
    print("one:", one)
    print("two:", one)
    print("three:", one)
    return train, test

def createOneHotMosei2way(train_label, test_label):
    maxlen = 1
    # print(maxlen)

    train = np.zeros((train_label.shape[0], train_label.shape[1], maxlen + 1))
    test = np.zeros((test_label.shape[0], test_label.shape[1], maxlen + 1))

    for i in range(train_label.shape[0]):
        for j in range(train_label.shape[1]):
            if train_label[i, j] > 0:
                train[i, j, 1] = 1
            else:
                if train_label[i, j] <= 0:
                    train[i, j, 0] = 1

    for i in range(test_label.shape[0]):
        for j in range(test_label.shape[1]):
            if test_label[i, j] > 0:
                test[i, j, 1] = 1
            else:
                if test_label[i, j] <= 0:
                    test[i, j, 0] = 1

    return train, test

data="mosei"
classes = 3
def get_raw_data(data, classes):
    mode = 'audio'
    with open('./dataset/{0}/raw/{1}_{2}way.pickle'.format(data, mode, classes), 'rb') as handle:
        u = pickle._Unpickler(handle)
        u.encoding = 'latin1'
        (audio_train, train_label, _, _, audio_test, test_label, _, train_length, _, test_length, _, _,_) = u.load()
        print("\n audio训练标签shape")
        print(train_label.shape)   # (2250, 98, 3)
        print("\n audio测试标签shape")
        print(test_label.shape)  # (678, 98, 3)

    mode = 'text'
    with open('./dataset/{0}/raw/{1}_{2}way.pickle'.format(data, mode, classes), 'rb') as handle:
        u = pickle._Unpickler(handle)
        u.encoding = 'latin1'
        (text_train, train_label, _, _, text_test, test_label, _, train_length, _, test_length, _, _, _) = u.load()
        print("\n text训练标签shape")
        print(train_label.shape)  # (2250, 98, 3)
        print("\n text测试标签shape")
        print(test_label.shape)  # (678, 98, 3)

    mode = 'video'
    with open('./dataset/{0}/raw/{1}_{2}way.pickle'.format(data, mode, classes), 'rb') as handle:
        u = pickle._Unpickler(handle)
        u.encoding = 'latin1'
        (video_train, train_label, _, _, video_test, test_label, _, train_length, _, test_length, _, _,_) = u.load()
        print("\n video训练标签shape")
        print(train_label.shape)  # (2250, 98, 3)
        print("\n video测试标签shape")
        print(test_label.shape)  # (678, 98, 3)

    train_data = np.concatenate((audio_train, video_train, text_train), axis=-1)
    print("---------------------")
    print("train_data：", train_data.shape)    #  (2250, 98, 409)
    test_data = np.concatenate((audio_test, video_test, text_test), axis=-1)
    print("test_data:", test_data.shape)      # (678, 98, 409)

    train_label = train_label.astype('int')
    test_label = test_label.astype('int')
    print(train_data.shape)   # (2250, 98, 409)
    print(test_data.shape)    # (678, 98, 409)
    train_mask = np.zeros((train_data.shape[0], train_data.shape[1]), dtype='float')
    for i in range(len(train_length)):
        train_mask[i, :train_length[i]] = 1.0

    test_mask = np.zeros((test_data.shape[0], test_data.shape[1]), dtype='float')
    for i in range(len(test_length)):
        test_mask[i, :test_length[i]] = 1.0

    # train_label, test_label = createOneHotMosei3way(train_label, test_label)
    print("--------------------------------------")
    print(train_label.shape)
    print(test_label.shape)
    print(train_label[3][4])

    print('train_mask', train_mask.shape)

    seqlen_train = train_length
    seqlen_test = test_length

    return train_data, test_data, audio_train, audio_test, text_train, text_test, video_train, video_test, train_label, test_label, seqlen_train, seqlen_test, train_mask, test_mask





train_data, test_data, audio_train, audio_test, text_train, text_test, video_train, video_test, train_label, test_label, \
seqlen_train, seqlen_test, train_mask, test_mask = get_raw_data(data, classes)

