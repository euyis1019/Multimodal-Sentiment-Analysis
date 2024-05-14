import gc, numpy as np, pickle
import tensorflow as tf
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Bidirectional, GRU, Masking, Dense, Dropout, TimeDistributed, Lambda, Activation, dot, \
    multiply, concatenate, MaxPool2D, MaxPooling3D, MaxPooling1D
# sklearn.metrics里的包
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os


# 计算测试准确率等结果
def calc_test_result(result, test_label, test_mask, print_detailed_results=True):  # 这里改改，改成打印细节
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
        print("Confusion Matrix 混淆矩阵:")
        print(confusion_matrix(true_label, predicted_label))
        print("Classification Report 分类报告 :")
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
    n1 = Activation('softmax')(m1)
    o1 = dot([n1, y], axes=[2, 1])
    a1 = multiply([o1, x])

    m2 = dot([y, x], axes=[2, 2])
    n2 = Activation('softmax')(m2)
    o2 = dot([n2, x], axes=[2, 1])
    a2 = multiply([o2, y])

    return concatenate([a1, a2])


def tri_model_attention(v, t, a):     # dense_video, dense_text, dense_audio

    Ftv = Dense(100, activation='tanh')(concatenate([t, v],  axis=2))
    Fta = Dense(100, activation='tanh')(concatenate([t, a], axis=2))
    Fav = Dense(100, activation='tanh')(concatenate([a, v], axis=2))

    c1 = dot([a, Ftv], axes=[2, 2])
    c2 = dot([v, Fta], axes=[2, 2])
    c3 = dot([t, Fav], axes=[2, 2])

    p1 = Activation('softmax')(c1)
    p2 = Activation('softmax')(c2)
    p3 = Activation('softmax')(c3)

    t1 = dot([p1, a], axes=[1, 1])
    t2 = dot([p2, v], axes=[1, 1])
    t3 = dot([p3, t], axes=[1, 1])

    Oatv = multiply([t1, Ftv])
    Ovta = multiply([t2, Fta])
    Otav = multiply([t3, Fav])

    return concatenate([Oatv, Oatv, Oatv], axis=2)

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
    #print(n)
    return a


# 上下文注意力模型

# TODO:修改的上下文注意力模型

def update_contextual_attention_model2(mode):  # 参数分别代表了不同的模型
    in_text = Input(shape=(train_text.shape[1], train_text.shape[2]))
    in_audio = Input(shape=(train_audio.shape[1], train_audio.shape[2]))
    in_video = Input(shape=(train_video.shape[1], train_video.shape[2]))
    masked_text = Masking(mask_value=0)(in_text)
    masked_audio = Masking(mask_value=0)(in_audio)
    masked_video = Masking(mask_value=0)(in_video)
    drop_rnn = 0.7
    gru_units = 300
    rnn_text = Bidirectional(GRU(gru_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5),
                             merge_mode='concat')(masked_text)
    rnn_audio = Bidirectional(GRU(gru_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5),
                              merge_mode='concat')(masked_audio)
    # 可以把视觉特征经过自己的图像分类提取网络，来进行图像特征的提取
    rnn_video = Bidirectional(GRU(gru_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5),
                              merge_mode='concat')(masked_video)
    rnn_text = Dropout(drop_rnn)(rnn_text)
    rnn_audio = Dropout(drop_rnn)(rnn_audio)
    rnn_video = Dropout(drop_rnn)(rnn_video)
    drop_dense = 0.7
    dense_units = 100
    #这里就是很标准的时间不变的全连接层，广播至每个时间步
    dense_text = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_text))
    dense_audio = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_audio))
    dense_video = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_video))

    # TODO:方案一  只考虑文本模态中每句话的自注意力机制
    # 文本每个语句的自注意力机制   获取每个语句内部单词的重要程度
    text_attention_features = []
    for k in range(max_utt_len):#对63句子
        m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_text)  # m1是每一条语句的video特征  这里是dense_text还是in_text呢？
        text_attention_features.append(self_attention(m1))
    merged_text_attention = concatenate(text_attention_features, axis=1)

    # 视频每个语句的自注意力机制   获取每个语句内部单词的重要程度
    video_attention_features = []
    for k in range(max_utt_len):
        m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_video)  # m1是每一条语句的video特征  这里是dense_text还是in_text呢？
        video_attention_features.append(self_attention(m1))
    merged_video_attention = concatenate(video_attention_features, axis=1)

    # 音频每个语句的自注意力机制   获取每个语句内部单词的重要程度
    audio_attention_features = []
    for k in range(max_utt_len):
        m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_audio)  # m1是每一条语句的video特征  这里是dense_text还是in_text呢？
        audio_attention_features.append(self_attention(m1))
    merged_audio_attention = concatenate(audio_attention_features, axis=1)

    # TODO:方案二  跨模态语句级注意力 每个语句的自注意力机制  语句级别的融合  对应相应的重要程度 各模态之间的交互信息
    # 每个语句的自注意力机制   获取每个语句对应的三种模态特征的重要程度
    attention_features1 = []
    for k in range(max_utt_len):
        m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_video)  # m1是某一条语句的video特征  dense_video
        m2 = Lambda(lambda x: x[:, k:k + 1, :])(dense_audio)  # m2是某一条语句的audio特征  dense_audio
        m3 = Lambda(lambda x: x[:, k:k + 1, :])(dense_text)  # m3是某一条语句的text特征  dense_text
        utterance_features = concatenate([m1, m2, m3], axis=1)
        attention_features1.append(self_attention(utterance_features))
    merged_attention1 = concatenate(attention_features1, axis=1)#(3*63,)
    merged_attention1 = Lambda(lambda x: K.reshape(x, (-1, max_utt_len, 3 * dense_units)))(
        merged_attention1)

    # TODO 方案三 由于文本所蕴含信息更为重要，考虑文本和其他两种模态的跨模态注意力机制
    vt_att = bi_modal_attention(dense_video, dense_text)
    av_att = bi_modal_attention(dense_audio, dense_video)
    ta_att = bi_modal_attention(dense_text, dense_audio)

    # # TODO:对比实验
    # vt_at_concat1 = self_attention(av_att)
    # # TODO 方案四 在方案三的基础之上进行自注意力机制  承接方案三
    # vt_at_concat = concatenate([vt_att, ta_att])
    # vt_at_concat_self = self_attention(vt_at_concat)
    # vt_self_att = self_attention(vt_att)
    # at_self_att = self_attention(ta_att)

    # TODO 方案五 三模态的跨模态注意力机制  (?, 126, 300)
    a_v_t_atten = tri_model_attention(dense_video, dense_text, dense_audio)

    # TODO：方案6 每个模态内的自注意力机制
    vv_att = self_attention(dense_video)
    tt_att = self_attention(dense_text)
    aa_att = self_attention(dense_audio)


    # 维度太高   可以考虑最大池化来降维


    unit = 64
    Bi = concatenate([vt_att, ta_att, av_att], axis=2)
    Ci = Dense(unit, activation="tanh")(Bi)
    Ci = Dense(unit)(Ci)
    a = Activation("softmax")(Ci)
    Bi = multiply([a, Ci])

    Di = concatenate([merged_text_attention, merged_video_attention, merged_audio_attention], axis=2)
    Ti = Dense(unit, activation="tanh")(Di)
    Ti = Dense(unit)(Ti)
    b = Activation("softmax")(Ti)
    Di = multiply([b, Ti])

    Ai = Dense(unit, activation="tanh")(a_v_t_atten)
    Ei = Dense(unit)(Ai)
    c = Activation("softmax")(Ei)
    Ai = multiply([c, Ei])

    # dense_audio = multiply([dense_audio, merged_audio_attention])
    # dense_video = multiply([dense_video, merged_video_attention])
    # dense_text = multiply([dense_text, merged_text_attention])

    merged = concatenate([
        dense_audio, dense_video, dense_text,
        merged_text_attention, merged_video_attention, merged_audio_attention,
        # Di,
        # vt_att, ta_att,  # av_att,
        Bi ,
        tt_att,
        # Ai,
        a_v_t_atten,
    ], axis=2)

    output = TimeDistributed(Dense(2, activation='softmax'))(merged)  # 2分类输出
    model = Model([in_text, in_audio, in_video], output)  # 输入是3个模态,输出是一个模态

    return model



# 训练模型
def train(mode):
    runs = 5  # 求五次结果平均
    accuracy = []

    for j in range(runs):
        # compile model #
        # 原始论文模型
        # model = contextual_attention_model(mode)
        # 第一次小论文修改模型
        # model = update_contextual_attention_model(mode)
        # 第二次小论文修改模型
        model = update_contextual_attention_model2(mode)
        # 打印出模型summary
        # model.summary()

        model.compile(optimizer='adam', loss='categorical_crossentropy', sample_weight_mode='temporal',
                      metrics=['accuracy'])

        # set callbacks #
        path = 'weightsformodel/Mosi_Trimodal_' + mode + '_Run_' + str(j) + '.hdf5'  # 报错，需要建立weights目录
        path = os.path.join(current_dir, path)
        early_stop = EarlyStopping(monitor='val_accuracy', patience=20, verbose=0)  # 将val_loss 改成了 val_acc
        check = ModelCheckpoint(path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=0)

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

    current_dir = os.path.dirname(os.path.abspath(__file__))
    text_path = os.path.join(current_dir, 'input/text.pickle')
    audio_path = os.path.join(current_dir, 'input/audio.pickle')
    video_path = os.path.join(current_dir, 'input/video.pickle')
    print(text_path)
    #Train 
    (train_text, train_label, test_text, test_label, max_utt_len, train_len, test_len) = pickle.load(open(text_path, 'rb'))
    (train_audio, _, test_audio, _, _, _, _) = pickle.load(open(audio_path, 'rb'))
    (train_video, _, test_video, _, _, _, _) = pickle.load(open(video_path, 'rb'))
    # log_tensor_shapes(
    #     train_text=train_text,
    #     train_label=train_label,
    #     test_text=test_text,
    #     test_label=test_label,
    #     max_utt_len=max_utt_len,
    #     train_len=train_len,
    #     test_len=test_len,
    #     train_audio=train_audio,
    #     test_audio=test_audio,
    #     train_video=train_video,
    #     test_video=test_video
    # )
    #Batch size is 62, 63 is utterance(the actual figure is defined by len list), 100is vector(maximum)
    # train_text: (62, 63, 100)
    # train_label: (62, 63)
    # test_text: (31, 63, 100)
    # test_label: (31, 63)
    # max_utt_len: Not a tensor, the value is63
    # train_len: Not a tensor, the value is[14, 30, 24, 12, 14, 19, 39, 23, 26, 25, 33, 22, 30, 26, 29, 34, 22, 29, 18, 24, 25, 13, 12, 18, 14, 15, 17, 55, 32, 22, 11, 9, 28, 30, 21, 34, 25, 15, 33, 29, 19, 43, 15, 19, 30, 15, 14, 27, 31, 30, 10, 24, 14, 16, 21, 22, 18, 16, 30, 24, 23, 35]
    # test_len: Not a tensor, the value is[13, 25, 30, 63, 30, 25, 12, 31, 31, 31, 44, 31, 18, 21, 18, 39, 16, 20, 13, 32, 16, 22, 9, 34, 16, 24, 18, 16, 20, 12, 22]
    # train_audio: (62, 63, 73)
    # test_audio: (31, 63, 73)
    # train_video: (62, 63, 100)
    # test_video: (31, 63, 100)
    
    train_label, test_label = create_one_hot_labels(train_label.astype('int'), test_label.astype('int'))

    train_mask, test_mask = create_mask(train_text, test_text, train_len, test_len)

    # 划分训练集和测试集
    num_train = int(len(train_text) * 0.8)

    # train_text, dev_text = train_text[:num_train, :, :], train_text[num_train:, :, :]
    # train_audio, dev_audio = train_audio[:num_train, :, :], train_audio[num_train:, :, :]
    # train_video, dev_video = train_video[:num_train, :, :], train_video[num_train:, :, :]
    # train_label, dev_label = train_label[:num_train, :, :], train_label[num_train:, :, :]
    # train_mask, dev_mask = train_mask[:num_train, :], train_mask[num_train:, :]
    # for mode in ['MMMU_BA', 'MMUU_SA', 'MU_SA', 'None']:
    #     train(mode)
    mode = "sdq"
    train(mode)