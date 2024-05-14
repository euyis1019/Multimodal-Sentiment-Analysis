import gc, numpy as np, pickle
import tensorflow as tf
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Bidirectional, GRU, Masking, Dense, Dropout, TimeDistributed, Lambda, Activation, dot, \
    multiply, concatenate
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# 计算测试准确率等结果
def calc_test_result(result, test_label, test_mask, print_detailed_results=False):
    '''
    # Arguments
        predicted test labels, gold test labels and test mask

    # Returns
        accuracy of the predicted labels
    '''
    true_label = []
    predicted_label = []

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if test_mask[i, j] == 1:
                true_label.append(np.argmax(test_label[i, j]))
                predicted_label.append(np.argmax(result[i, j]))

    if print_detailed_results:
        print("Confusion Matrix :")
        print(confusion_matrix(true_label, predicted_label))
        print("Classification Report :")
        print(classification_report(true_label, predicted_label))
    print("Accuracy ", accuracy_score(true_label, predicted_label))
    return accuracy_score(true_label, predicted_label)


# 将标签转为one-hot形式
def create_one_hot_labels(train_label, test_label):
    '''
    # Arguments
        train and test labels (2D matrices)

    # Returns
        one hot encoded train and test labels (3D matrices)
    '''

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


# mask
def create_mask(train_data, test_data, train_length, test_length):
    '''
    # Arguments
        train, test data (any one modality (text, audio or video)), utterance lengths in train, test videos

    # Returns
        mask for train and test data
    '''

    train_mask = np.zeros((train_data.shape[0], train_data.shape[1]), dtype='float')
    for i in range(len(train_length)):
        train_mask[i, :train_length[i]] = 1.0

    test_mask = np.zeros((test_data.shape[0], test_data.shape[1]), dtype='float')
    for i in range(len(test_length)):
        test_mask[i, :test_length[i]] = 1.0

    return train_mask, test_mask


# 加载pickle数据
(train_text, train_label, test_text, test_label, max_utt_len, train_len, test_len) = pickle.load(
    open('./input/text.pickle', 'rb'))
(train_audio, _, test_audio, _, _, _, _) = pickle.load(open('./input/audio.pickle', 'rb'))
(train_video, _, test_video, _, _, _, _) = pickle.load(open('./input/video.pickle', 'rb'))

train_label, test_label = create_one_hot_labels(train_label.astype('int'), test_label.astype('int'))

train_mask, test_mask = create_mask(train_text, test_text, train_len, test_len)

# 划分训练集和测试集
num_train = int(len(train_text) * 0.8)

train_text, dev_text = train_text[:num_train, :, :], train_text[num_train:, :, :]
train_audio, dev_audio = train_audio[:num_train, :, :], train_audio[num_train:, :, :]
train_video, dev_video = train_video[:num_train, :, :], train_video[num_train:, :, :]
train_label, dev_label = train_label[:num_train, :, :], train_label[num_train:, :, :]
train_mask, dev_mask = train_mask[:num_train, :], train_mask[num_train:, :]


# 跨模态注意力
def bi_modal_attention(x, y):  # x,y分别是两种不同的模态

    '''
    .  stands for dot product
    *  stands for elemwise multiplication
    {} stands for concatenation

    m1 = x . transpose(y) ||  m2 = y . transpose(x)
    n1 = softmax(m1)      ||  n2 = softmax(m2)
    o1 = n1 . y           ||  o2 = m2 . x
    a1 = o1 * x           ||  a2 = o2 * y

    return {a1, a2}

    '''

    m1 = dot([x, y], axes=[2, 2])
    # print("\n m1")
    # print(m1.shape)   # (?, 63, 63)
    n1 = Activation('softmax')(m1)
    o1 = dot([n1, y], axes=[2, 1])
    # print("\n o1")
    # print(o1.shape)  # (?, 63, 100)
    a1 = multiply([o1, x])
    # print("\n a1")
    # print(a1.shape)  # (?, 63, 100)

    m2 = dot([y, x], axes=[2, 2])
    n2 = Activation('softmax')(m2)
    o2 = dot([n2, x], axes=[2, 1])
    a2 = multiply([o2, y])

    return concatenate([a1, a2])


def tri_model_attention(x, y, z):
    '''
        .  stands for dot product
        *  stands for elemwise multiplication
        {} stands for concatenation

        m1 = x . transpose(y) ||  m2 = y . transpose(x)
        n1 = softmax(m1)      ||  n2 = softmax(m2)
        o1 = n1 . y           ||  o2 = m2 . x
        a1 = o1 * x           ||  a2 = o2 * y

        return {a1, a2}



        '''

    m1 = dot([x, y], axes=[2, 2])
    # print(m1.shape)  # (?, 63, 63)
    m1 = dot([m1, z], axes=[2, 1])
    # print(m1.shape)  # (?, 63, 100)
    n1 = Activation('softmax')(m1)
    # print(n1.shape)  # (?, 63, 100)

    m2 = dot([x, z], axes=[2, 2])
    m2 = dot([m2, y], axes=[2, 1])
    n2 = Activation('softmax')(m2)

    m3 = dot([y, z], axes=[2, 2])
    m3 = dot([m3, x], axes=[2, 1])
    n3 = Activation('softmax')(m3)

    return concatenate([n1, n2, n3])


# 自注意力机制
def self_attention(x):
    '''
    .  stands for dot product
    *  stands for elemwise multiplication

    m = x . transpose(x)
    n = softmax(m)
    o = n . x
    a = o * x

    return a

    '''

    m = dot([x, x], axes=[2, 2])
    n = Activation('softmax')(m)
    o = dot([n, x], axes=[2, 1])
    a = multiply([o, x])

    return a


# 上下文注意力模型
def contextual_attention_model(mode):  # 参数分别代表了不同的模型

    ########### Input Layer ############

    in_text = Input(shape=(train_text.shape[1], train_text.shape[2]))
    in_audio = Input(shape=(train_audio.shape[1], train_audio.shape[2]))
    in_video = Input(shape=(train_video.shape[1], train_video.shape[2]))

    ########### Masking Layer ############

    masked_text = Masking(mask_value=0)(in_text)
    masked_audio = Masking(mask_value=0)(in_audio)
    masked_video = Masking(mask_value=0)(in_video)

    ########### Recurrent Layer ############

    drop_rnn = 0.7
    gru_units = 300

    rnn_text = Bidirectional(GRU(gru_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5),
                             merge_mode='concat')(masked_text)
    rnn_audio = Bidirectional(GRU(gru_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5),
                              merge_mode='concat')(masked_audio)
    rnn_video = Bidirectional(GRU(gru_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5),
                              merge_mode='concat')(masked_video)

    rnn_text = Dropout(drop_rnn)(rnn_text)
    rnn_audio = Dropout(drop_rnn)(rnn_audio)
    rnn_video = Dropout(drop_rnn)(rnn_video)

    ########### Time-Distributed Dense Layer ############

    drop_dense = 0.7
    dense_units = 100

    dense_text = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_text))
    dense_audio = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_audio))
    dense_video = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_video))

    ########### Attention Layer ############

    ## Multi Modal Multi Utterance Bi-Modal attention ##
    if mode == 'MMMU_BA':

        vt_att = bi_modal_attention(dense_video, dense_text)
        av_att = bi_modal_attention(dense_audio, dense_video)
        ta_att = bi_modal_attention(dense_text, dense_audio)

        merged = concatenate([vt_att, av_att, ta_att, dense_video, dense_audio, dense_text])


    ## Multi Modal Uni Utterance Self Attention ##
    elif mode == 'MMUU_SA':

        attention_features = []

        for k in range(max_utt_len):
            # extract multi modal features for each utterance #
            m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_video)
            m2 = Lambda(lambda x: x[:, k:k + 1, :])(dense_audio)
            m3 = Lambda(lambda x: x[:, k:k + 1, :])(dense_text)

            utterance_features = concatenate([m1, m2, m3], axis=1)
            attention_features.append(self_attention(utterance_features))

        merged_attention = concatenate(attention_features, axis=1)
        merged_attention = Lambda(lambda x: K.reshape(x, (-1, max_utt_len, 3 * dense_units)))(merged_attention)

        merged = concatenate([merged_attention, dense_video, dense_audio, dense_text])


    ## Multi Utterance Self Attention ##
    elif mode == 'MU_SA':

        vv_att = self_attention(dense_video)
        tt_att = self_attention(dense_text)
        aa_att = self_attention(dense_audio)

        merged = concatenate([aa_att, vv_att, tt_att, dense_video, dense_audio, dense_text])


    ## No Attention ##
    elif mode == 'None':

        merged = concatenate([dense_video, dense_audio, dense_text])

    else:
        print("Mode must be one of 'MMMU-BA', 'MMUU-SA', 'MU-SA' or 'None'.")
        return

    ########### Output Layer ############

    output = TimeDistributed(Dense(2, activation='softmax'))(merged)  # 2分类输出
    model = Model([in_text, in_audio, in_video], output)  # 输入是3个模态,输出是一个模态

    return model


# TODO:修改的上下文注意力模型
def update_contextual_attention_model(mode):  # 参数分别代表了不同的模型

    ########### Input Layer ############

    in_text = Input(shape=(train_text.shape[1], train_text.shape[2]))
    in_audio = Input(shape=(train_audio.shape[1], train_audio.shape[2]))
    in_video = Input(shape=(train_video.shape[1], train_video.shape[2]))

    ########### Masking Layer ############

    masked_text = Masking(mask_value=0)(in_text)
    masked_audio = Masking(mask_value=0)(in_audio)
    masked_video = Masking(mask_value=0)(in_video)

    ########### Recurrent Layer ############

    drop_rnn = 0.7
    gru_units = 300

    print("\n 未处理之前的特征")
    print(in_text.shape)  # (?, 63, 100)
    print(in_audio.shape)  # (?, 63, 73)
    print(in_video.shape)  # (?, 63, 100)

    print("\n 经过掩码之后的特征")
    print(masked_text.shape)  # (?, 63, 100)
    print(masked_audio.shape)  # (?, 63, 73)
    print(masked_video.shape)  # (?, 63, 100)

    rnn_text = Bidirectional(GRU(gru_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5),
                             merge_mode='concat')(masked_text)
    rnn_audio = Bidirectional(GRU(gru_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5),
                              merge_mode='concat')(masked_audio)
    # 可以把视觉特征经过自己的图像分类提取网络，来进行图像特征的提取
    rnn_video = Bidirectional(GRU(gru_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5),
                              merge_mode='concat')(masked_video)

    print("\n 经过Bidirectional GRU之后的特征")
    print(rnn_text.shape)  # (?, ?, 600)
    print(rnn_audio.shape)  # (?, ?, 600)
    print(rnn_video.shape)  # (?, ?, 600)

    # 用了这个好像效果特别差
    # rnn_text = Bidirectional(GRU(gru_units, return_sequences=True),
    #                          merge_mode='concat')(masked_text)
    # rnn_audio = Bidirectional(GRU(gru_units, return_sequences=True),
    #                           merge_mode='concat')(masked_audio)
    # rnn_video = Bidirectional(GRU(gru_units, return_sequences=True),
    #                           merge_mode='concat')(masked_video)

    # rnn_text = Dropout(drop_rnn)(rnn_text)
    # rnn_audio = Dropout(drop_rnn)(rnn_audio)
    # rnn_video = Dropout(drop_rnn)(rnn_video)

    ########### Time-Distributed Dense Layer ############

    # drop_dense = 0.7
    dense_units = 100

    # dense_text = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='relu'))(rnn_text))
    # dense_audio = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='relu'))(rnn_audio))
    # dense_video = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='relu'))(rnn_video))

    # 尝试不用这个
    dense_text = TimeDistributed(Dense(dense_units, activation='relu'))(rnn_text)
    dense_audio = TimeDistributed(Dense(dense_units, activation='relu'))(rnn_audio)
    dense_video = TimeDistributed(Dense(dense_units, activation='relu'))(rnn_video)

    print("\n 经过TimeDistributed Dense之后的特征")
    print(dense_text.shape)  # (?, 63, 100)
    print(dense_audio.shape)  # (?, 63, 100)
    print(dense_video.shape)  # (?, 63, 100)

    # 得统一维度
    # dense_text = rnn_text
    # dense_audio = rnn_audio
    # dense_video = rnn_video

    # TODO:方案一  每个语句的自注意力机制  自己改一下代码方案
    # 每个语句的自注意力机制   获取每个语句内部单词的重要程度
    # 好像不是这个
    attention_features = []
    for k in range(max_utt_len):
        # extract multi modal features for each utterance #
        m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_video)  # m1是每一条语句的video特征
        attention_features.append(self_attention(m1))
        m2 = Lambda(lambda x: x[:, k:k + 1, :])(dense_audio)  # m2是每一条语句的audio特征
        attention_features.append(self_attention(m2))
        m3 = Lambda(lambda x: x[:, k:k + 1, :])(dense_text)  # m3是每一条语句的text特征
        attention_features.append(self_attention(m3))
        # print("\n 每条语句的特征")
        # print(m1.shape)  # (?, 1, 100)
        # print(m2.shape)  # (?, 1, 100)
        # print(m3.shape)  # (?, 1, 100)
        # utterance_features = concatenate([m1, m2, m3], axis=1)
        # print("\n 将每条语句的每种模态结合起来进行自注意力机制")
        # print(utterance_features.shape)  # (?, 3, 100)
        # attention_features.append(self_attention(utterance_features))
    # 将列表的注意力机制结合起来
    merged_attention = concatenate(attention_features, axis=1)
    # reshape
    merged_attention = Lambda(lambda x: K.reshape(x, (-1, max_utt_len, 3 * dense_units)))(merged_attention)

    # # TODO:方案一
    # # 每个语句的自注意力机制   获取每个语句内部单词的重要程度
    # # 好像不是这个
    # attention_features = []
    # for k in range(max_utt_len):
    #     # extract multi modal features for each utterance #
    #     m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_video)    # m1是每一条语句的video特征
    #     m2 = Lambda(lambda x: x[:, k:k + 1, :])(dense_audio)    # m2是每一条语句的audio特征
    #     m3 = Lambda(lambda x: x[:, k:k + 1, :])(dense_text)     # m3是每一条语句的text特征
    #     # print("\n 每条语句的特征")
    #     # print(m1.shape)  # (?, 1, 100)
    #     # print(m2.shape)  # (?, 1, 100)
    #     # print(m3.shape)  # (?, 1, 100)
    #     utterance_features = concatenate([m1, m2, m3], axis=1)
    #     # print("\n 将每条语句的每种模态结合起来进行自注意力机制")
    #     # print(utterance_features.shape)  # (?, 3, 100)
    #     attention_features.append(self_attention(utterance_features))
    # # 将列表的注意力机制结合起来
    # merged_attention = concatenate(attention_features, axis=1)
    # # reshape
    # merged_attention = Lambda(lambda x: K.reshape(x, (-1, max_utt_len, 3 * dense_units)))(merged_attention)

    # 每个模态内的自注意力机制
    vv_att = self_attention(concatenate([dense_video, merged_attention]))
    tt_att = self_attention(concatenate([dense_text, merged_attention]))
    aa_att = self_attention(concatenate([dense_audio, merged_attention]))

    # # 每个模态间的自注意力机制
    # vvv_att = self_attention(concatenate([vv_att, merged_attention]))
    # ttt_att = self_attention(concatenate([tt_att, merged_attention]))
    # aaa_att = self_attention(concatenate([aa_att, merged_attention]))

    # 跨模态的两两注意力机制
    vt_att = bi_modal_attention(dense_video, dense_text)
    av_att = bi_modal_attention(dense_audio, dense_video)
    ta_att = bi_modal_attention(dense_text, dense_audio)

    # 三模态的跨模态注意力机制
    a_v_t_atten = tri_model_attention(dense_video, dense_text, dense_audio)

    # 结合在一起进行分类
    merged = concatenate(
        [vt_att, av_att, ta_att, a_v_t_atten, vv_att, tt_att, aa_att, dense_audio, dense_text,
         dense_video])  # ,dense_audio, dense_text, dense_video

    ########### Attention Layer ############

    ## Multi Modal Multi Utterance Bi-Modal attention ##

    # TODO：跨模态注意力机制  --两两结合
    # print("\n经过两两跨模态注意力之前的shape")
    # print(dense_video.shape)  # (?, 63, 100)
    # print(dense_audio.shape)  # (?, 63, 100)
    # print(dense_text.shape)  # (?, 63, 100)

    # vt_att = bi_modal_attention(dense_video, dense_text)
    # av_att = bi_modal_attention(dense_audio, dense_video)
    # ta_att = bi_modal_attention(dense_text, dense_audio)

    # merged = concatenate([vt_att, av_att, ta_att, dense_video, dense_audio, dense_text])

    # print("\n经过两两跨模态注意力之后的shape")
    # print(vt_att.shape)   # (?, 63, 200)
    # print(av_att.shape)   # (?, 63, 200)
    # print(ta_att.shape)   # (?, 63, 200)

    # TODO：跨模态注意力机制  --三个模态结合   每个模态的重要性不同，直接学习一个矩阵表明重要性程度
    # a_v_t_atten = tri_model_attention(dense_video, dense_text, dense_audio)
    # print("跨模态注意力机制  --三个模态结合之后的shape")
    # print(a_v_t_atten.shape)   # (?, 63, 300)

    # merged = concatenate([vt_att, av_att, ta_att,  a_v_t_atten])
    # print("\n merge shape")
    # print(merged.shape)  # (?, 63, 1200)

    # TODO:经过自注意力机制   每个模态内部重要程度也不同
    # vv_att = self_attention(dense_video)
    # tt_att = self_attention(dense_text)
    # aa_att = self_attention(dense_audio)
    # merged = concatenate([vt_att, av_att, ta_att, a_v_t_atten])

    # # ## Multi Modal Uni Utterance Self Attention ##
    # # TODO：单一语句的自注意力机制   每一个语句的词汇重要性也不同
    # attention_features = []
    #
    # for k in range(max_utt_len):
    #     # extract multi modal features for each utterance #
    #     m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_video)
    #     m2 = Lambda(lambda x: x[:, k:k + 1, :])(dense_audio)
    #     m3 = Lambda(lambda x: x[:, k:k + 1, :])(dense_text)
    #
    #     utterance_features = concatenate([m1, m2, m3], axis=1)
    #     attention_features.append(self_attention(utterance_features))
    #
    # merged_attention = concatenate(attention_features, axis=1)
    # merged_attention = Lambda(lambda x: K.reshape(x, (-1, max_utt_len, 3 * dense_units)))(merged_attention)
    # #     merged = concatenate([merged_attention, dense_video, dense_audio, dense_text])
    #
    # merged = concatenate(
    #     [vt_att, av_att, ta_att, dense_video, dense_audio, dense_text, a_v_t_atten, vv_att, tt_att, aa_att, merged_attention])
    # print("\n merge shape")
    # print(merged.shape)  # (?, 63, 1500)

    ########### Output Layer ############
    output = TimeDistributed(Dense(2, activation='softmax'))(merged)  # 2分类输出
    model = Model([in_text, in_audio, in_video], output)  # 输入是3个模态,输出是一个模态

    return model


# 训练模型
def train(mode):
    runs = 10  # 求五次结果平均
    accuracy = []

    for j in range(runs):
        # np.random.seed(j)
        # tf.set_random_seed(j)   # 不设置随机种子

        # compile model #
        model = contextual_attention_model(mode)
        # 打印出模型summary
        # model.summary()
        model.compile(optimizer='adam', loss='categorical_crossentropy', sample_weight_mode='temporal',
                      metrics=['accuracy'])

        # set callbacks #
        path = 'weights/Mosi_Trimodal_' + mode + '_Run_' + str(j) + '.hdf5'  # 报错，需要建立weights目录

        early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=0)
        check = ModelCheckpoint(path, monitor='val_acc', save_best_only=True, mode='max', verbose=0)

        # train model #
        history = model.fit([train_text, train_audio, train_video], train_label,
                            epochs=64,  # 感觉可以加大epoch
                            batch_size=32,  # 改成64好像效果不是很好
                            sample_weight=train_mask,
                            shuffle=True,
                            callbacks=[early_stop, check],
                            # validation_data=([dev_text, dev_audio, dev_video], dev_label, dev_mask),
                            validation_data=([test_text, test_audio, test_video], test_label, test_mask),
                            verbose=1)

        # test results #
        model.load_weights(path)
        test_predictions = model.predict([test_text, test_audio, test_video])
        test_accuracy = calc_test_result(test_predictions, test_label, test_mask)
        accuracy.append(test_accuracy)

        # release gpu memory #
        K.clear_session()
        del model, history
        gc.collect()

    # summarize test results #

    avg_accuracy = sum(accuracy) / len(accuracy)
    max_accuracy = max(accuracy)

    print('Mode: ', mode)
    print('Avg Test Accuracy:', '{0:.4f}'.format(avg_accuracy), '|| Max Test Accuracy:', '{0:.4f}'.format(max_accuracy))
    print('-' * 55)


if __name__ == "__main__":
    for mode in ['None']:   # 'MMMU_BA', 'MMUU_SA', 'MU_SA',
        train(mode)
