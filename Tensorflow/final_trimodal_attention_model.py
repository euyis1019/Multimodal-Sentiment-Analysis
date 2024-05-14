import gc, numpy as np, pickle
import tensorflow as tf
from keras.models import Model
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.layers import Dense,Flatten,Conv2D,Activation,MaxPooling2D,BatchNormalization,\
    GlobalAveragePooling2D,Reshape,Multiply,Concatenate,Input, GlobalAveragePooling1D, Conv1D, Conv3D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Bidirectional, GRU, Masking, Dense, Dropout, TimeDistributed, Lambda, Activation, dot, \
    multiply, concatenate, MaxPool2D, MaxPooling3D, MaxPooling1D
# sklearn.metrics里的包
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 设置GPU环境
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


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
        # merged = concatenate([vt_att, av_att, ta_att, dense_video, dense_audio, dense_text])
        merged = concatenate([ta_att, dense_video, dense_audio, dense_text])
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
    print("--------output-------------")
    print(output.shape)
    model = Model([in_text, in_audio, in_video], output)  # 输入是3个模态,输出是一个模态
    return model

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

    # print("\n 未处理之前的特征")
    # print(in_text.shape)   # (?, 63, 100)
    # print(in_audio.shape)  # (?, 63, 73)
    # print(in_video.shape)  # (?, 63, 100)

    # print("\n 经过掩码之后的特征")
    # print(masked_text.shape)  # (?, 63, 100)
    # print(masked_audio.shape)  # (?, 63, 73)
    # print(masked_video.shape)  # (?, 63, 100)

    # 这里是否可以直接加注意力机制了？  不行，因为特征数目不一致

    rnn_text = Bidirectional(GRU(gru_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5),
                             merge_mode='concat')(masked_text)
    rnn_audio = Bidirectional(GRU(gru_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5),
                              merge_mode='concat')(masked_audio)
    # 可以把视觉特征经过自己的图像分类提取网络，来进行图像特征的提取
    rnn_video = Bidirectional(GRU(gru_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5),
                              merge_mode='concat')(masked_video)

    # print("\n 经过Bidirectional GRU之后的特征")
    # print(rnn_text.shape)  # (?, ?, 600)
    # print(rnn_audio.shape)  # (?, ?, 600)
    # print(rnn_video.shape)  # (?, ?, 600)

    rnn_text = Dropout(drop_rnn)(rnn_text)
    rnn_audio = Dropout(drop_rnn)(rnn_audio)
    rnn_video = Dropout(drop_rnn)(rnn_video)

    # rnn_text = Dropout(drop_rnn)(rnn_text)
    # rnn_audio = Dropout(drop_rnn)(rnn_audio)
    # rnn_video = Dropout(drop_rnn)(rnn_video)

    ########### Time-Distributed Dense Layer ############

    drop_dense = 0.7
    dense_units = 100

    # dense_text = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='relu'))(rnn_text))
    # dense_audio = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='relu'))(rnn_audio))
    # dense_video = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='relu'))(rnn_video))

    # 尝试不用这个
    dense_text = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_text))
    dense_audio = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_audio))
    dense_video = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_video))

    # print("\n 经过TimeDistributed Dense之后的特征")
    # print(dense_text.shape)  # (?, 63, 100)
    # print(dense_audio.shape)  # (?, 63, 100)
    # print(dense_video.shape)  # (?, 63, 100)

    # TODO:方案一  只考虑文本模态中每句话的自注意力机制
    # 每个语句的自注意力机制   获取每个语句内部单词的重要程度
    text_attention_features = []
    for k in range(max_utt_len):
        # extract multi modal features for each utterance #
        m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_text)  # m1是每一条语句的video特征  这里是dense_text还是in_text呢？
        text_attention_features.append(self_attention(m1))
    # 将列表的注意力机制结合起来
    merged_text_attention = concatenate(text_attention_features, axis=1)

    # TODO:方案一 只考虑图像模态中每句话的自注意力机制
    # 每个语句的自注意力机制   获取每个语句内部单词的重要程度
    video_attention_features = []
    for k in range(max_utt_len):
        # extract multi modal features for each utterance #
        m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_video)  # m1是每一条语句的video特征  这里是dense_text还是in_text呢？
        video_attention_features.append(self_attention(m1))
    # 将列表的注意力机制结合起来
    merged_video_attention = concatenate(video_attention_features, axis=1)

    # TODO:方案一  只考虑语音模态中每句话的自注意力机制
    # 每个语句的自注意力机制   获取每个语句内部单词的重要程度
    audio_attention_features = []
    for k in range(max_utt_len):
        # extract multi modal features for each utterance #
        m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_audio)  # m1是每一条语句的video特征  这里是dense_text还是in_text呢？
        audio_attention_features.append(self_attention(m1))
    # 将列表的注意力机制结合起来
    merged_audio_attention = concatenate(audio_attention_features, axis=1)

    # TODO:方案二  跨模态语句级注意力 每个语句的自注意力机制  语句级别的融合  对应相应的重要程度 各模态之间的交互信息
    # 每个语句的自注意力机制   获取每个语句对应的三种模态特征的重要程度
    attention_features1 = []
    for k in range(max_utt_len):
        # extract multi modal features for each utterance #
        m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_video)  # m1是每一条语句的video特征  dense_video
        m2 = Lambda(lambda x: x[:, k:k + 1, :])(dense_audio)  # m2是每一条语句的audio特征  dense_audio
        m3 = Lambda(lambda x: x[:, k:k + 1, :])(dense_text)  # m3是每一条语句的text特征  dense_text
        # print("\n 每条语句的特征")
        # print(m1.shape)  # (?, 1, 100)
        # print(m2.shape)  # (?, 1, 100)
        # print(m3.shape)  # (?, 1, 100)
        # TODO:对比试验
        utterance_features = concatenate([m1, m2, m3], axis=1)
        # utterance_features = concatenate([m1, m2], axis=1)
        # print("\n 将每条语句的每种模态结合起来进行自注意力机制")
        # print(utterance_features.shape)  # (?, 3, 100)
        attention_features1.append(self_attention(utterance_features))
    # 将列表的注意力机制结合起来
    merged_attention1 = concatenate(attention_features1, axis=1)
    # reshape
    merged_attention1 = Lambda(lambda x: K.reshape(x, (-1, max_utt_len, 3 * dense_units)))(
        merged_attention1)  # TODO：这里改了

    # TODO 方案三 由于文本所蕴含信息更为重要，考虑文本和其他两种模态的跨模态注意力机制
    vt_att = bi_modal_attention(dense_video, dense_text)
    av_att = bi_modal_attention(dense_audio, dense_video)
    ta_att = bi_modal_attention(dense_text, dense_audio)

    # TODO:对比实验
    vt_at_concat1 = self_attention(av_att)

    # TODO 方案四 在方案三的基础之上进行自注意力机制  承接方案三
    vt_at_concat = concatenate([vt_att, ta_att])
    vt_at_concat_self = self_attention(vt_at_concat)
    vt_self_att = self_attention(vt_att)
    at_self_att = self_attention(ta_att)

    # TODO 方案五 三模态的跨模态注意力机制
    a_v_t_atten = tri_model_attention(dense_video, dense_text, dense_audio)

    # # TODO:方案二  获取每个语句对应的三种模态特征的重要程度
    # # 每个语句的自注意力机制   获取每个语句对应的三种模态特征的重要程度
    # attention_features1 = []
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
    #     attention_features1.append(self_attention(utterance_features))
    # # 将列表的注意力机制结合起来
    # merged_attention1 = concatenate(attention_features1, axis=1)
    # # reshape
    # merged_attention1 = Lambda(lambda x: K.reshape(x, (-1, max_utt_len, 3 * dense_units)))(merged_attention1)

    # TODO:方案三  考虑文本video组合的自注意力机制
    # 每个语句的自注意力机制   获取每个语句对应的三种模态特征的重要程度
    attention_features_text_to_audio = []
    attention_features_text_to_video = []
    for k in range(max_utt_len):
        # extract multi modal features for each utterance #
        m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_video)  # m1是每一条语句的video特征
        m2 = Lambda(lambda x: x[:, k:k + 1, :])(dense_audio)  # m2是每一条语句的audio特征
        m3 = Lambda(lambda x: x[:, k:k + 1, :])(dense_text)  # m3是每一条语句的text特征
        utterance_features_tv = concatenate([m1, m3], axis=1)
        attention_features_text_to_video.append(self_attention(utterance_features_tv))
    # 将列表的注意力机制结合起来
    merged_attention_two_combine = concatenate(attention_features_text_to_video, axis=1)
    # reshape
    merged_attention_two_combine = Lambda(lambda x: K.reshape(x, (-1, max_utt_len, 2 * dense_units)))(
        merged_attention_two_combine)

    # TODO:方案五  考虑文本audio组合的自注意力机制
    # 每个语句的自注意力机制   获取每个语句对应的三种模态特征的重要程度
    attention_features_text_to_audio = []
    for k in range(max_utt_len):
        # extract multi modal features for each utterance #
        m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_video)  # m1是每一条语句的video特征
        m2 = Lambda(lambda x: x[:, k:k + 1, :])(dense_audio)  # m2是每一条语句的audio特征
        m3 = Lambda(lambda x: x[:, k:k + 1, :])(dense_text)  # m3是每一条语句的text特征
        utterance_features_ta = concatenate([m2, m3], axis=1)
        attention_features_text_to_audio.append(self_attention(utterance_features_ta))
    # 将列表的注意力机制结合起来
    merged_attention_two_combine_ta = concatenate(attention_features_text_to_audio, axis=1)
    # reshape
    merged_attention_two_combine_ta = Lambda(lambda x: K.reshape(x, (-1, max_utt_len, 2 * dense_units)))(
        merged_attention_two_combine_ta)

    # TODO：方案6 每个模态内的自注意力机制
    vv_att = self_attention(dense_video)
    tt_att = self_attention(dense_text)
    aa_att = self_attention(dense_audio)

    # vv_att = self_attention(concatenate([dense_video, merged_attention]))
    # tt_att = self_attention(concatenate([dense_text, merged_attention]))
    # aa_att = self_attention(concatenate([dense_audio, merged_attention]))

    # # 每个模态间的自注意力机制
    # vvv_att = self_attention(concatenate([vv_att, merged_attention]))
    # ttt_att = self_attention(concatenate([tt_att, merged_attention]))
    # aaa_att = self_attention(concatenate([aa_att, merged_attention]))

    # TODO 方案8 三模态的跨模态注意力机制
    a_v_t_atten = tri_model_attention(dense_video, dense_text, dense_audio)

    # 结合在一起进行分类
    # 加了a_v_t_atten，分类准确率下降了？
    # 思想：文本模态中每句话先进行自注意力机制，获取每个语句内部单词的重要程度；
    # merged = concatenate([
    #      dense_text, dense_audio, dense_video,   # 经过dense层的文本、视觉、语音特征
    #      # merged_text_attention,                  # 经过dense层的文本特征中每句话的自注意力机制
    #      # merged_video_attention,                 # 经过dense层的图像特征中每句话的自注意力机制
    #      # merged_audio_attention,                 # 经过dense层的语音特征中每句话的自注意力机制
    #      # merged_attention1,                      # 跨模态语句级别自注意力机制
    #      # vt_att, ta_att,                         # 文本与其他模态的跨模态注意力机制
    #      # tt_att,                                 # 文本模态内的自注意力机制
    #      # vt_self_att, at_self_att,                # 先文本与其他模态的跨模态注意力机制，再进行自注意力机制
    #      # a_v_t_atten,                             # 三模态跨模态自注意力机制
    #      # vt_at_concat_self,                         # 先文本与其他模态的跨模态注意力机制，concat 再进行自注意力机制
    #      # merged_text_attention,  # 单一文本模态的注意力机制
    #      ])  # ,dense_audio, dense_text, dense_video
    merged = concatenate([
        dense_audio, dense_video, dense_text,
        merged_audio_attention, merged_video_attention, merged_text_attention,
        merged_attention1,
        # av_att,
        # vt_at_concat1,
    ])
    # merged_attention,
    # dense_audio, dense_video, dense_text, merged_attention1
    # ,  vv_att, tt_att, aa_att, a_v_t_atten
    # vt_att, av_att, ta_att

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
    # merged = TimeDistributed(Dense(64, activation='softmax'))(merged)   # 80..
    # merged = self_attention(merged)
    output = TimeDistributed(Dense(2, activation='softmax'))(merged)  # 2分类输出
    model = Model([in_text, in_audio, in_video], output)  # 输入是3个模态,输出是一个模态

    return model

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
    dense_text = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_text))
    dense_audio = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_audio))
    dense_video = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_video))

    # TODO:方案一  只考虑文本模态中每句话的自注意力机制
    # 文本每个语句的自注意力机制   获取每个语句内部单词的重要程度
    text_attention_features = []
    for k in range(max_utt_len):
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
        m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_video)  # m1是每一条语句的video特征  dense_video
        m2 = Lambda(lambda x: x[:, k:k + 1, :])(dense_audio)  # m2是每一条语句的audio特征  dense_audio
        m3 = Lambda(lambda x: x[:, k:k + 1, :])(dense_text)  # m3是每一条语句的text特征  dense_text
        utterance_features = concatenate([m1, m2, m3], axis=1)
        attention_features1.append(self_attention(utterance_features))
    merged_attention1 = concatenate(attention_features1, axis=1)
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

    return concatenate([Oatv, Ovta, Otav], axis=2)

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


# 通道注意力
def squeeze(inputs):    #   (?, 63, 100)
    # input_channels = int(inputs.shape[-1])

    input_channels = 1
    x = GlobalAveragePooling1D()(inputs)
    # print(x.shape)    #   (?, 100)
    x = Dense(input_channels)(inputs)
    # print(x.shape)    #  (?, 63, 1)
    x = Activation('sigmoid')(x)
    # print(x.shape)    # (?, 63, 1)
    # x = Reshape((1,  input_channels))(x)
    # print(x.shape)
    x = Multiply()([inputs, x])
    # print(x.shape)  # (?, 63, 100)
    return x

# 空间注意力机制
def space(inputs):
    x = Conv1D(filters=1, kernel_size=1, padding='same')(inputs)
    # print(x.shape)   # (?, 63, 1)
    x = Activation('sigmoid')(x)
    # print(x.shape)   # (?, 63, 1)
    x = Multiply()([inputs, x])
    # print(x.shape)  # (?, 63, 100)
    return x

# 复现小论文
def con_BIAM_contextual_attention_model2(mode):

    in_text = Input(shape=(train_text.shape[1], train_text.shape[2]))
    in_audio = Input(shape=(train_audio.shape[1], train_audio.shape[2]))
    in_video = Input(shape=(train_video.shape[1], train_video.shape[2]))

    print("----in_video的维度为-----")
    print(in_video.shape)
    # TODO:加入精简注意力机制
    squeeze_video = squeeze(in_video)
    squeeze_video = space(squeeze_video)


    masked_text = Masking(mask_value=0)(in_text)
    masked_audio = Masking(mask_value=0)(in_audio)
    masked_video = Masking(mask_value=0)(squeeze_video)

    drop_rnn = 0.5
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
    drop_dense = 0.5
    dense_units = 100


    dense_text = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_text))
    dense_audio = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_audio))
    dense_video = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_video))

    # TODO:加入文本模态的自注意力机制   根据两阶段算法设计的
    self_text = self_attention(rnn_text)
    rnn_text = concatenate([rnn_text, self_text])


    # TODO 方案三 由于文本所蕴含信息更为重要，考虑文本和其他两种模态的跨模态注意力机制
    vt_att = bi_modal_attention(dense_video, dense_text)
    ta_att = bi_modal_attention(dense_text, dense_audio)
    av_att = bi_modal_attention(dense_audio, dense_video)


    vt_self = self_attention(vt_att)
    ta_self = self_attention(ta_att)
    av_self = self_attention(av_att)


    a = concatenate([dense_text, vt_att], axis=2)
    b = concatenate([dense_video, ta_att], axis=2)
    c = concatenate([dense_audio, av_att], axis=2)

    a1 = self_attention(a)
    b1 = self_attention(b)
    c1 = self_attention(c)

    a2 = concatenate([a1, vt_att])
    b2 = concatenate([b1, ta_att])
    c2 = concatenate([c1, av_att])

    a3 = Dense(100, activation='tanh')(a2)
    b3 = Dense(100, activation='tanh')(b2)
    c3 = Dense(100, activation='tanh')(c2)

    # TODO：方案6 每个模态内的自注意力机制
    vv_att = self_attention(dense_video)
    tt_att = self_attention(dense_text)
    aa_att = self_attention(dense_audio)


    # TODO 方案四 三模态的跨模态注意力机制  (?, 126, 300)
    a_v_t_atten = tri_model_attention(dense_video, dense_text, dense_audio)

    a_v_t_atten_self = self_attention(a_v_t_atten)


    # TODO 在原文基础之上进行更改的
    # 文本每个语句的自注意力机制   获取每个语句内部单词的重要程度
    text_attention_features = []
    for k in range(max_utt_len):
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


    a_v_t_atten = tri_model_attention(dense_video, dense_text, dense_audio)

    # 映射到一个特征语义空间
    yinshe = concatenate([vt_att, ta_att, av_att])
    yinshe = Dense(100, activation='tanh')(yinshe)


    merged = concatenate([
        dense_audio, dense_video, dense_text,
        merged_text_attention, merged_video_attention, merged_audio_attention,
        # a3, b3, c3,
        vt_att, ta_att, av_att,
        # yinshe,
        # vv_att, aa_att, tt_att,
        a_v_t_atten_self,
        # a_v_t_atten,

        # vt_self, ta_self, av_self,
        # a_v_t_atten,
    ], axis=2)

    output = TimeDistributed(Dense(2, activation='softmax'))(merged)  # 2分类输出
    model = Model([in_text, in_audio, in_video], output)  # 输入是3个模态, 输出是一个模态

    return model


def final_contextual_attention_model2(mode):
    in_text = Input(shape=(train_text.shape[1], train_text.shape[2]))
    in_audio = Input(shape=(train_audio.shape[1], train_audio.shape[2]))
    in_video = Input(shape=(train_video.shape[1], train_video.shape[2]))

    # TODO:加入精简注意力机制
    squeeze_video = squeeze(in_video)
    squeeze_video = space(squeeze_video)
    squeeze_audio = squeeze(in_audio)
    squeeze_audio = space(squeeze_audio)
    suqeeze_text = squeeze(in_text)
    suqeeze_text = space(suqeeze_text)

    masked_text = Masking(mask_value=0)(suqeeze_text)
    masked_audio = Masking(mask_value=0)(squeeze_audio)
    masked_video = Masking(mask_value=0)(squeeze_video)


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
    dense_text = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_text))
    dense_audio = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_audio))
    dense_video = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_video))

    # TODO:方案一  只考虑文本模态中每句话的自注意力机制
    # 文本每个语句的自注意力机制   获取每个语句内部单词的重要程度
    text_attention_features = []
    for k in range(max_utt_len):
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
        m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_video)  # m1是每一条语句的video特征  dense_video
        m2 = Lambda(lambda x: x[:, k:k + 1, :])(dense_audio)  # m2是每一条语句的audio特征  dense_audio
        m3 = Lambda(lambda x: x[:, k:k + 1, :])(dense_text)  # m3是每一条语句的text特征  dense_text
        utterance_features = concatenate([m1, m2, m3], axis=1)
        attention_features1.append(self_attention(utterance_features))
    merged_attention1 = concatenate(attention_features1, axis=1)
    merged_attention1 = Lambda(lambda x: K.reshape(x, (-1, max_utt_len, 3 * dense_units)))(
        merged_attention1)

    # 由于图像和语音结合会有干扰，所以转换为尝试文本分别与其他两个模态的跨模态注意力机制
    attention_features2 = []
    for k in range(max_utt_len):
        m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_video)  # m1是每一条语句的video特征  dense_video
        m2 = Lambda(lambda x: x[:, k:k + 1, :])(dense_audio)  # m2是每一条语句的audio特征  dense_audio
        m3 = Lambda(lambda x: x[:, k:k + 1, :])(dense_text)  # m3是每一条语句的text特征  dense_text
        attention_features = concatenate([m1,  m3], axis=1)
        attention_features2.append(self_attention(attention_features))
    merged_attention2 = concatenate(attention_features2, axis=1)
    merged_attention2 = Lambda(lambda x: K.reshape(x, (-1, max_utt_len, 2 * dense_units)))(
        merged_attention2)

    attention_features3 = []
    for k in range(max_utt_len):
        m1 = Lambda(lambda x: x[:, k:k + 1, :])(dense_video)  # m1是每一条语句的video特征  dense_video
        m2 = Lambda(lambda x: x[:, k:k + 1, :])(dense_audio)  # m2是每一条语句的audio特征  dense_audio
        m3 = Lambda(lambda x: x[:, k:k + 1, :])(dense_text)  # m3是每一条语句的text特征  dense_text
        attention_features = concatenate([m2, m3], axis=1)
        attention_features3.append(self_attention(attention_features))
    merged_attention3 = concatenate(attention_features3, axis=1)
    merged_attention3 = Lambda(lambda x: K.reshape(x, (-1, max_utt_len, 2 * dense_units)))(
        merged_attention3)


    # TODO 方案三 由于文本所蕴含信息更为重要，考虑文本和其他两种模态的跨模态注意力机制
    vt_att = bi_modal_attention(dense_video, dense_text)
    av_att = bi_modal_attention(dense_audio, dense_video)
    ta_att = bi_modal_attention(dense_text, dense_audio)

    # TODO 方案四 三模态的跨模态注意力机制  (?, 126, 300)
    a_v_t_atten = tri_model_attention(dense_video, dense_text, dense_audio)



    # # # TODO:对比实验
    # # vt_at_concat1 = self_attention(av_att)
    # # # TODO 方案四 在方案三的基础之上进行自注意力机制  承接方案三
    # # vt_at_concat = concatenate([vt_att, ta_att])
    # # vt_at_concat_self = self_attention(vt_at_concat)
    # # vt_self_att = self_attention(vt_att)
    # # at_self_att = self_attention(ta_att)

    # TODO：方案6 每个模态内的自注意力机制
    vv_att = self_attention(dense_video)
    tt_att = self_attention(dense_text)
    aa_att = self_attention(dense_audio)


    # # 维度太高   可以考虑最大池化来降维
    #
    #
    # unit = 64
    # Bi = concatenate([vt_att, ta_att, av_att], axis=2)
    # Ci = Dense(unit, activation="tanh")(Bi)
    # Ci = Dense(unit)(Ci)
    # a = Activation("softmax")(Ci)
    # Bi = multiply([a, Ci])
    #
    # Di = concatenate([merged_text_attention, merged_video_attention, merged_audio_attention], axis=2)
    # Ti = Dense(unit, activation="tanh")(Di)
    # Ti = Dense(unit)(Ti)
    # b = Activation("softmax")(Ti)
    # Di = multiply([b, Ti])
    #
    # Ai = Dense(unit, activation="tanh")(a_v_t_atten)
    # Ei = Dense(unit)(Ai)
    # c = Activation("softmax")(Ei)
    # Ai = multiply([c, Ei])

    # dense_audio = multiply([dense_audio, merged_audio_attention])
    # dense_video = multiply([dense_video, merged_video_attention])
    # dense_text = multiply([dense_text, merged_text_attention])

    merged = concatenate([
        dense_audio, dense_video, dense_text,
        merged_text_attention, merged_video_attention, merged_audio_attention,
        # merged_attention1,    # 貌似直接这样结合效果不佳
        # merged_attention2, merged_attention3,   #  感觉跨模态的特征融合就很不合理
        # vt_att, ta_att,
        # a_v_t_atten,
        # Di,
        vt_att, ta_att,  av_att,
        # Bi ,
        # tt_att,
        # Ai,
        a_v_t_atten,
        # tt_att, aa_att, vv_att,        # 每个模态的自注意力机制
    ], axis=2)

    output = TimeDistributed(Dense(2, activation='softmax'))(merged)  # 2分类输出
    model = Model([in_text, in_audio, in_video], output)  # 输入是3个模态,输出是一个模态

    return model


def compare_atten(mode):
    in_text = Input(shape=(train_text.shape[1], train_text.shape[2]))
    in_audio = Input(shape=(train_audio.shape[1], train_audio.shape[2]))
    in_video = Input(shape=(train_video.shape[1], train_video.shape[2]))

    # TODO:加入精简注意力机制
    squeeze_video = squeeze(in_video)
    squeeze_video = space(squeeze_video)

    # TODO:加入文本模态的自注意力机制   根据两阶段算法设计的
    self_text = self_attention(in_text)
    rnn_text = concatenate([in_text, self_text])

    masked_text = Masking(mask_value=0)(rnn_text)
    masked_audio = Masking(mask_value=0)(in_audio)
    masked_video = Masking(mask_value=0)(squeeze_video)

    drop_rnn = 0.5
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
    drop_dense = 0.5
    dense_units = 100

    dense_text = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_text))
    dense_audio = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_audio))
    dense_video = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_video))

    # TODO 方案三 由于文本所蕴含信息更为重要，考虑文本和其他两种模态的跨模态注意力机制
    vt_att = bi_modal_attention(dense_video, dense_text)
    ta_att = bi_modal_attention(dense_text, dense_audio)
    av_att = bi_modal_attention(dense_audio, dense_video)

    vt_self = self_attention(vt_att)
    ta_self = self_attention(ta_att)
    av_self = self_attention(av_att)

    a = concatenate([dense_text, vt_att], axis=2)
    b = concatenate([dense_video, ta_att], axis=2)
    c = concatenate([dense_audio, av_att], axis=2)

    a1 = self_attention(a)
    b1 = self_attention(b)
    c1 = self_attention(c)

    a2 = concatenate([a1, vt_att])
    b2 = concatenate([b1, ta_att])
    c2 = concatenate([c1, av_att])

    a3 = Dense(100, activation='tanh')(a2)
    b3 = Dense(100, activation='tanh')(b2)
    c3 = Dense(100, activation='tanh')(c2)

    # TODO：方案6 每个模态内的自注意力机制
    vv_att = self_attention(dense_video)
    tt_att = self_attention(dense_text)
    aa_att = self_attention(dense_audio)

    # TODO 方案四 三模态的跨模态注意力机制  (?, 126, 300)
    a_v_t_atten = tri_model_attention(dense_video, dense_text, dense_audio)

    a_v_t_atten_self = self_attention(a_v_t_atten)
    a_v_t_atten_self1 = concatenate([a_v_t_atten_self, a_v_t_atten])

    # TODO 在原文基础之上进行更改的
    # 文本每个语句的自注意力机制   获取每个语句内部单词的重要程度
    text_attention_features = []
    for k in range(max_utt_len):
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

    a_v_t_atten = tri_model_attention(dense_video, dense_text, dense_audio)

    # 映射到一个特征语义空间
    yinshe = concatenate([vt_att, ta_att, av_att])
    yinshe = Dense(100, activation='tanh')(yinshe)

    # T+A
    # merged = concatenate([
    #     dense_text, dense_audio,
    #     merged_text_attention,  merged_audio_attention,
    #     ta_att,
    # ], axis=2)

    # V+A
    merged = concatenate([
        dense_video, dense_audio, dense_text,
        merged_video_attention, merged_audio_attention,
        av_att,
    ], axis=2)

    # # T+V
    # merged = concatenate([
    #     dense_text, dense_video,
    #     merged_text_attention, merged_video_attention,
    #     vt_att,
    # ], axis=2)

    # merged = concatenate([
    #     dense_text, dense_audio, dense_video,
    #     merged_text_attention, merged_video_attention, merged_audio_attention,
    #     vt_att, ta_att, av_att,
    #     # a_v_t_atten,
    #     a_v_t_atten_self,
    # ], axis=2)


    output = TimeDistributed(Dense(2, activation='softmax'))(merged)  # 2分类输出
    model = Model([in_text, in_audio, in_video], output)  # 输入是3个模态, 输出是一个模态

    return model

    return model


# 训练模型
def train(mode):
    runs = 5   # 求五次结果平均
    accuracy = []

    for j in range(runs):
        # model = contextual_attention_model(mode)
        # 第二次小论文修改模型
        # model = final_contextual_attention_model2(mode)
        # model = con_BIAM_contextual_attention_model2(mode)
        model = compare_atten(mode)
        # 打印出模型summary
        # model.summary()


        model.compile(optimizer='adam', loss='categorical_crossentropy', sample_weight_mode='temporal',
                      metrics=['accuracy'])

        # set callbacks #
        path = 'weights/Mosi_Trimodal_' + mode + '_Run_' + str(j) + '.hdf5'  # 报错，需要建立weights目录

        early_stop = EarlyStopping(monitor='val_acc', patience=20, verbose=0)  # 将val_loss 改成了 val_acc
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

    # for mode in ['MMMU_BA', 'MMUU_SA', 'MU_SA', 'None']:
    #     train(mode)
    mode = "sdq"
    train(mode)