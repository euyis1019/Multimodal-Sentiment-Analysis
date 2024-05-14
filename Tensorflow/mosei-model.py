import gc, numpy as np, pickle
import tensorflow as tf
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Bidirectional, GRU, Masking, Dense, Dropout, TimeDistributed, Lambda, Activation, dot, \
    multiply, concatenate
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from keras.optimizers import SGD,Adam


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# TODO:添加的内容
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

def get_raw_data(data, classes):
    mode = 'audio'
    with open('./dataset/{0}/raw/{1}_{2}way.pickle'.format(data, mode, classes), 'rb') as handle:
        u = pickle._Unpickler(handle)
        u.encoding = 'latin1'
        (audio_train, train_label, _, _, audio_test, test_label, _, train_length, _, test_length, _, _,_) = u.load()

    mode = 'text'
    with open('./dataset/{0}/raw/{1}_{2}way.pickle'.format(data, mode, classes), 'rb') as handle:
        u = pickle._Unpickler(handle)
        u.encoding = 'latin1'
        (text_train, train_label, _, _, text_test, test_label, _, train_length, _, test_length, _, _, _) = u.load()

    mode = 'video'
    with open('./dataset/{0}/raw/{1}_{2}way.pickle'.format(data, mode, classes), 'rb') as handle:
        u = pickle._Unpickler(handle)
        u.encoding = 'latin1'
        (video_train, train_label, _, _, video_test, test_label, _, train_length, _, test_length, _, _,_) = u.load()
        # print("text_train", text_train.shape)
        # print('audio_train', audio_train.shape)
        # print('video_train', video_train.shape)
        # print('text_test', text_test.shape)
        # print('audio_test', audio_test.shape)
        # print('video_test', video_test.shape)

    train_data = np.concatenate((audio_train, video_train, text_train), axis=-1)
    test_data = np.concatenate((audio_test, video_test, text_test), axis=-1)

    train_label = train_label.astype('int')
    test_label = test_label.astype('int')
    # print(train_data.shape)
    # print(test_data.shape)
    train_mask = np.zeros((train_data.shape[0], train_data.shape[1]), dtype='float')
    for i in range(len(train_length)):
        train_mask[i, :train_length[i]] = 1.0

    test_mask = np.zeros((test_data.shape[0], test_data.shape[1]), dtype='float')
    for i in range(len(test_length)):
        test_mask[i, :test_length[i]] = 1.0

    # train_label, test_label = createOneHot(train_label, test_label)

    # print('train_mask', train_mask.shape)

    seqlen_train = train_length
    seqlen_test = test_length

    return train_data, test_data, audio_train, audio_test, text_train, text_test, video_train, video_test, train_label, test_label, seqlen_train, seqlen_test, train_mask, test_mask

# 计算测试准确率等结果
def calc_test_result(result, test_label, test_mask, print_detailed_results=True):
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

# 加载pickle数据
# TODO： 加载CMU-MOSEI数据集
# 要先将数据集拿到
# (train_text, train_label, test_text, test_label, max_utt_len, train_len, test_len) = pickle.load(
#     open('./input/text.pickle', 'rb'))
# (train_audio, _, test_audio, _, _, _, _) = pickle.load(open('./input/audio.pickle', 'rb'))
# (train_video, _, test_video, _, _, _, _) = pickle.load(open('./input/video.pickle', 'rb'))

# train_label, test_label = create_one_hot_labels(train_label.astype('int'), test_label.astype('int'))

# train_mask, test_mask = create_mask(train_text, test_text, train_len, test_len)

train_data, test_data, train_audio, test_audio, train_text, test_text, train_video, test_video, train_label, test_label, \
max_utt_len, seqlen_test, train_mask, test_mask = get_raw_data('mosei', 3)
# max_utt_len

# print("\n text")
# print(train_text.shape)   # (62, 63, 100)    (2250, 98, 300)
# print(train_label.shape)  # (62, 63)   ???    (2250, 98, 2)
# print(test_text.shape)    # (31, 63, 100)     (678, 98, 300)
# print(test_label.shape)   # (31, 63)   ???    (678, 98, 2)
#
# print("\n video")
# print(train_video.shape)  # (62, 63, 100)     (2250, 98, 35)
# print(test_video.shape)   # (31, 63, 100)     (678, 98, 35)
#
# print("\n audio")
# print(train_audio.shape)  # (62, 63, 73)      (2250, 98, 74)
# print(test_audio.shape)  # (31, 63, 73)       (678, 98, 74)

print("max_utt_len:")
print(max_utt_len)
print(max_utt_len.shape)    # (2250,)
print(max_utt_len.shape[0])


# 划分训练集和测试集
# TODO:使用CMU-MOSEI数据集
num_train = int(len(train_text) * 0.8)

train_text, dev_text = train_text[:num_train, :, :], train_text[num_train:, :, :]
train_audio, dev_audio = train_audio[:num_train, :, :], train_audio[num_train:, :, :]
train_video, dev_video = train_video[:num_train, :, :], train_video[num_train:, :, :]
train_label, dev_label = train_label[:num_train, :, :], train_label[num_train:, :, :]
train_mask, dev_mask = train_mask[:num_train, :], train_mask[num_train:, :]


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
        print("  \n MMUU_SA ")
        print(dense_video.shape)      #  (?, 98, 100)
        print(dense_audio.shape)      #  (?, 98, 100)
        print(dense_text.shape)       #  (?, 98, 100)

        for k in range(98):      # 把max_len改成98
            # extract multi modal features for each utterance #
            m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_video)
            m2 = Lambda(lambda x: x[:, k:k + 1, :])(dense_audio)
            m3 = Lambda(lambda x: x[:, k:k + 1, :])(dense_text)

            utterance_features = concatenate([m1, m2, m3], axis=1)
            attention_features.append(self_attention(utterance_features))


        merged_attention = concatenate(attention_features, axis=1)
        print(merged_attention.shape)
        merged_attention = Lambda(lambda x: K.reshape(x, (-1, 98, 3 * dense_units)))(merged_attention)

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

    output = TimeDistributed(Dense(3, activation='softmax'))(merged)  # 2分类输出
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

    ########### Time-Distributed Dense Layer ############

    drop_dense = 0.7
    dense_units = 100

    # 尝试不用这个
    dense_text = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_text))
    dense_audio = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_audio))
    dense_video = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_video))


    # TODO:方案一  只考虑文本模态中每句话的自注意力机制
    # 每个语句的自注意力机制   获取每个语句内部单词的重要程度
    text_attention_features = []
    for k in range(98):
        # extract multi modal features for each utterance #
        m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_text)  # m1是每一条语句的video特征  这里是dense_text还是in_text呢？
        text_attention_features.append(self_attention(m1))
    # 将列表的注意力机制结合起来
    merged_text_attention = concatenate(text_attention_features, axis=1)

    # TODO:方案一 只考虑图像模态中每句话的自注意力机制
    # 每个语句的自注意力机制   获取每个语句内部单词的重要程度
    video_attention_features = []
    for k in range(98):
        # extract multi modal features for each utterance #
        m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_video)  # m1是每一条语句的video特征  这里是dense_text还是in_text呢？
        video_attention_features.append(self_attention(m1))
    # 将列表的注意力机制结合起来
    merged_video_attention = concatenate(video_attention_features, axis=1)

    # TODO:方案一  只考虑语音模态中每句话的自注意力机制
    # 每个语句的自注意力机制   获取每个语句内部单词的重要程度
    audio_attention_features = []
    for k in range(98):
        # extract multi modal features for each utterance #
        m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_audio)  # m1是每一条语句的video特征  这里是dense_text还是in_text呢？
        audio_attention_features.append(self_attention(m1))
    # 将列表的注意力机制结合起来
    merged_audio_attention = concatenate(audio_attention_features, axis=1)

    # TODO:方案二  跨模态语句级注意力 每个语句的自注意力机制  语句级别的融合  对应相应的重要程度 各模态之间的交互信息
    # 每个语句的自注意力机制   获取每个语句对应的三种模态特征的重要程度
    attention_features1 = []
    for k in range(98):
        # extract multi modal features for each utterance #
        m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_video)  # m1是每一条语句的video特征  dense_video
        m2 = Lambda(lambda x: x[:, k:k + 1, :])(dense_audio)  # m2是每一条语句的audio特征  dense_audio
        m3 = Lambda(lambda x: x[:, k:k + 1, :])(dense_text)  # m3是每一条语句的text特征  dense_text
        utterance_features = concatenate([m1, m2, m3], axis=1)
        attention_features1.append(self_attention(utterance_features))
    # 将列表的注意力机制结合起来
    merged_attention1 = concatenate(attention_features1, axis=1)
    # reshape
    merged_attention1 = Lambda(lambda x: K.reshape(x, (-1, 98, 3 * dense_units)))(merged_attention1)

    # TODO 方案三 由于文本所蕴含信息更为重要，考虑文本和其他两种模态的跨模态注意力机制
    vt_att = bi_modal_attention(dense_video, dense_text)
    # av_att = bi_modal_attention(dense_audio, dense_video)
    ta_att = bi_modal_attention(dense_text, dense_audio)

    # TODO 方案四 在方案三的基础之上进行自注意力机制  承接方案三
    vt_at_concat = concatenate([vt_att, ta_att])
    vt_at_concat_self = self_attention(vt_at_concat)
    vt_self_att = self_attention(vt_att)
    at_self_att = self_attention(ta_att)

    # TODO 方案五 三模态的跨模态注意力机制
    a_v_t_atten = tri_model_attention(dense_video, dense_text, dense_audio)

    # TODO:方案三  考虑文本video组合的自注意力机制
    # 每个语句的自注意力机制   获取每个语句对应的三种模态特征的重要程度
    attention_features_text_to_audio = []
    attention_features_text_to_video = []
    for k in range(98):
        # extract multi modal features for each utterance #
        m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_video)  # m1是每一条语句的video特征
        m2 = Lambda(lambda x: x[:, k:k + 1, :])(dense_audio)  # m2是每一条语句的audio特征
        m3 = Lambda(lambda x: x[:, k:k + 1, :])(dense_text)  # m3是每一条语句的text特征
        utterance_features_tv = concatenate([m1, m3], axis=1)
        attention_features_text_to_video.append(self_attention(utterance_features_tv))
    # 将列表的注意力机制结合起来
    merged_attention_two_combine = concatenate(attention_features_text_to_video, axis=1)
    # reshape
    merged_attention_two_combine = Lambda(lambda x: K.reshape(x, (-1, 98, 2 * dense_units)))(
        merged_attention_two_combine)

    # TODO:方案五  考虑文本audio组合的自注意力机制
    # 每个语句的自注意力机制   获取每个语句对应的三种模态特征的重要程度
    attention_features_text_to_audio = []
    for k in range(98):
        # extract multi modal features for each utterance #
        m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_video)  # m1是每一条语句的video特征
        m2 = Lambda(lambda x: x[:, k:k + 1, :])(dense_audio)  # m2是每一条语句的audio特征
        m3 = Lambda(lambda x: x[:, k:k + 1, :])(dense_text)  # m3是每一条语句的text特征
        utterance_features_ta = concatenate([m2, m3], axis=1)
        attention_features_text_to_audio.append(self_attention(utterance_features_ta))
    # 将列表的注意力机制结合起来
    merged_attention_two_combine_ta = concatenate(attention_features_text_to_audio, axis=1)
    # reshape
    merged_attention_two_combine_ta = Lambda(lambda x: K.reshape(x, (-1, 98, 2 * dense_units)))(
        merged_attention_two_combine_ta)

    # TODO：方案6 每个模态内的自注意力机制
    vv_att = self_attention(dense_video)
    tt_att = self_attention(dense_text)
    aa_att = self_attention(dense_audio)

    # TODO 方案8 三模态的跨模态注意力机制
    a_v_t_atten = tri_model_attention(dense_video, dense_text, dense_audio)
    merged = concatenate([
        dense_text, dense_audio, dense_video,  # 经过dense层的文本、视觉、语音特征
        merged_text_attention,  # 经过dense层的文本特征中每句话的自注意力机制
        merged_video_attention,  # 经过dense层的图像特征中每句话的自注意力机制
        merged_audio_attention,  # 经过dense层的语音特征中每句话的自注意力机制
        merged_attention1,  # 跨模态语句级别自注意力机制
        vt_att, ta_att,  # 文本与其他模态的跨模态注意力机制
        # tt_att,                                 # 文本模态内的自注意力机制
        # vt_self_att, at_self_att,                # 先文本与其他模态的跨模态注意力机制，再进行自注意力机制
        # a_v_t_atten,                             # 三模态跨模态自注意力机制
        vt_at_concat_self,  # 先文本与其他模态的跨模态注意力机制，concat 再进行自注意力机制
        # merged_text_attention,  # 单一文本模态的注意力机制
    ])  # ,dense_audio, dense_text, dense_video

    output = TimeDistributed(Dense(3, activation='softmax'))(merged)  # 2分类输出
    model = Model([in_text, in_audio, in_video], output)  # 输入是3个模态,输出是一个模态

    return model


# 训练模型
def train(mode):
    runs = 1  # 求五次结果平均
    accuracy = []

    for j in range(runs):
        # np.random.seed(j)
        # tf.set_random_seed(j)   # 不设置随机种子

        # compile model #
        model = contextual_attention_model(mode)
        # model = update_contextual_attention_model(mode)
        # 打印出模型summary
        # model.summary()
        optimizer = Adam(lr=0.001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', sample_weight_mode='temporal',
                      metrics=['accuracy'])

        # set callbacks #
        path = 'weights/Mosei_Trimodal_' + mode + '_Run_' + str(j) + '.hdf5'  # 报错，需要建立weights目录

        early_stop = EarlyStopping(monitor='val_acc', patience=20, verbose=0)  # 将val_loss 改成了 val_acc
        check = ModelCheckpoint(path, monitor='val_acc', save_best_only=True, mode='max', verbose=0)

        # train model #
        history = model.fit([train_text, train_audio, train_video], train_label,
                            epochs=64,  # 感觉可以加大epoch
                            batch_size=64,  # 改成64好像效果不是很好
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
    for mode in ['MMMU_BA', 'MMUU_SA', 'MU_SA', 'None']:   # 'MMMU_BA', 'MMUU_SA', 'MU_SA',   None : Avg Test Accuracy: 0.6153 || Max Test Accuracy: 0.6153
        train(mode)
    # mode = "None"
    # train(mode)