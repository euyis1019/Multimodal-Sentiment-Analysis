import matplotlib.pyplot as plt

# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(14, 5))


# TODO:第三章

# 基于cifar-10的实验结果对比
# waters = ('NIN+Dropout', 'DSN', 'ResNet-CE', 'SqueezeNet', 'WideResNet', 'FractalNet', 'DSENet', 'MS-FNet', 'MFRA')
# buy_number = [10.41, 7.97, 9.59, 9.74, 4.97, 5.22, 5.70, 6.19, 3.86]
# a1 = ['NIN+Dropout']
# b1 = [10.41]
# a2 = ['DSN']
# b2 = [7.97]
# a3 = ['ResNet-CE']
# b3 = [9.59]
# a4 = ['SqueezeNet']
# b4 = [9.74]
# a5 = ['WideResNet']
# b5 = [4.97]
# a6 = ['FractalNet']
# b6 = [5.22]
# a7 = ['DSENet']
# b7 = [5.70]
# a8 = ['MS-FNet']
# b8 = [6.19]
# a9 = ['MFRA']
# b9 = [3.86]
#
# plt.bar(a1, b1, color="blue", width=0.4)
# plt.bar(a2, b2, color="yellow", width=0.4)
# plt.bar(a3, b3, color="pink", width=0.4)
# plt.bar(a4, b4, color="green", width=0.4)
# plt.bar(a5, b5, color="orange", width=0.4)
# plt.bar(a6, b6, color="purple", width=0.4)
# plt.bar(a7, b7, color="gray", width=0.4)
# plt.bar(a8, b8, color="gold", width=0.4)
# plt.bar(a9, b9, color="red", width=0.4)
#
# plt.title('不同模型的错误率对比')
# plt.ylabel("错误率")
# plt.xlabel("模型")
# plt.savefig('C:\\Users\\Hasee\\Desktop\\1-1-cifar-10-错误率.png', format='png')
# plt.show()

# 基于cifar-100的实验结果对比
# waters = ('NIN+Dropout', 'DSN', 'ResNet-CE', 'SqueezeNet', 'DSENet', 'MS-FNet', 'DNI', 'X-CNNS', 'MFRA')
# buy_number = [35.68, 34.57, 29.71, 39.80, 25.85, 29.36, 42.82, 29.36, 17.51]
# a1 = ['NIN+Dropout']
# b1 = [35.68]
# a2 = ['DSN']
# b2 = [34.57]
# a3 = ['ResNet-CE']
# b3 = [29.71]
# a4 = ['SqueezeNet']
# b4 = [39.80]
# a5 = ['DSENet']
# b5 = [25.85]
# a6 = ['MS-FNet']
# b6 = [29.36]
# a7 = ['DNI']
# b7 = [42.82]
# a8 = ['X-CNNS']
# b8 = [29.36]
# a9 = ['MFRA']
# b9 = [17.51]
#
# plt.bar(a1, b1, color="blue", width=0.4)
# plt.bar(a2, b2, color="yellow", width=0.4)
# plt.bar(a3, b3, color="pink", width=0.4)
# plt.bar(a4, b4, color="green", width=0.4)
# plt.bar(a5, b5, color="orange", width=0.4)
# plt.bar(a6, b6, color="purple", width=0.4)
# plt.bar(a7, b7, color="gray", width=0.4)
# plt.bar(a8, b8, color="gold", width=0.4)
# plt.bar(a9, b9, color="red", width=0.4)
#
# plt.title('不同模型的错误率对比')
# plt.ylabel("错误率")
# plt.xlabel("模型")
# plt.savefig('C:\\Users\\Hasee\\Desktop\\1-1-cifar-100-错误率.png', format='png')
# plt.show()

# Caltech-101数据集对比
# waters = ('文献81', '文献82', '文献83', '文献84', '文献85', 'MFRA')
# buy_number = [18.6, 13.5, 11.6, 6.6, 5.3, 5.36]
# a1 = ['文献81']
# b1 = [18.6]
# a2 = ['文献82']
# b2 = [13.5]
# a3 = ['文献83']
# b3 = [11.6]
# a4 = ['文献84']
# b4 = [6.6]
# a5 = ['文献85']
# b5 = [5.3]
# a6 = ['MFRA']
# b6 = [5.36]
#
#
# plt.bar(a1, b1, color="blue", width=0.4)
# plt.bar(a2, b2, color="yellow", width=0.4)
# plt.bar(a3, b3, color="pink", width=0.4)
# plt.bar(a4, b4, color="green", width=0.4)
# plt.bar(a5, b5, color="orange", width=0.4)
# plt.bar(a6, b6, color="red", width=0.4)
#
# plt.title('不同模型的错误率对比')
# plt.ylabel("错误率")
# plt.xlabel("模型")
# plt.savefig('C:\\Users\\Hasee\\Desktop\\1-1-Caltech-101-错误率.png', format='png')
# plt.show()

# TODO:第四章

# CMU-MOSI数据集
# waters = ('GME-LSTM', 'MARN', 'TFN', 'Dialogue-RNN', 'BC-LSTM', 'Multilogue-Net', 'Con-BIAM', 'AMF-BiGRU',  'MMMU-BA', 'MSAM')
# buy_number = [76.5, 77.1, 77.1, 79.8, 80.3, 81.19, 81.91, 82.05, 82.31, 82.45]
# plt.bar(waters, buy_number, color='pink', width=0.5)
# a1 = ['GME-LSTM']
# b1 = [76.5]
# a2 = ['MARN']
# b2 = [77.1]
# a3 = ['TFN']
# b3 = [77.1]
# a4 = ['Dialogue-RNN']
# b4 = [79.8]
# a5 = ['BC-LSTM']
# b5 = [80.3]
# a6 = ['Multilogue-Net']
# b6 = [81.19]
# a7 = ['Con-BIAM']
# b7 = [81.91]
# a8 = ['AMF-BiGRU']
# b8 = [82.05]
# a9 = ['MMMU-BA']
# b9 = [82.31]
# a10 = ['MSAM']
# b10 = [82.45]
#
# plt.bar(a1, b1, color="blue", width=0.4)
# plt.bar(a2, b2, color="yellow", width=0.4)
# plt.bar(a3, b3, color="pink", width=0.4)
# plt.bar(a4, b4, color="green", width=0.4)
# plt.bar(a5, b5, color="orange", width=0.4)
# plt.bar(a6, b6, color="purple", width=0.4)
# plt.bar(a7, b7, color="gray", width=0.4)
# plt.bar(a8, b8, color="gold", width=0.4)
# plt.bar(a9, b9, color="black", width=0.4)
# plt.bar(a10, b10, color="red", width=0.4)
#
# plt.title('不同模型的准确率对比')
# plt.ylabel("准确率")
# plt.xlabel("模型")
# plt.savefig('C:\\Users\\Hasee\\Desktop\\2-1-MOSI-准确率.png', format='png')
# plt.show()


# CMU-MOSEI数据集
# waters = ('MFN', 'Graph-MFN', 'CIM-Att', 'MMMU-BA', 'MAM', 'MSAM')
# buy_number = [77.70, 76.90, 80.50, 79.80, 81.00, 81.10]
# plt.bar(waters, buy_number, color='orange', width=0.5)
# a1 = ['MFN']
# b1 = [77.70]
# a2 = ['Graph-MFN']
# b2 = [76.90]
# a3 = ['CIM-Att']
# b3 = [80.50]
# a4 = ['MMMU-BA']
# b4 = [79.80]
# a5 = ['MAM']
# b5 = [81.00]
# a6 = ['MSAM']
# b6 = [81.10]
#
#
# plt.bar(a1, b1, color="blue", width=0.4)
# plt.bar(a2, b2, color="yellow", width=0.4)
# plt.bar(a3, b3, color="pink", width=0.4)
# plt.bar(a4, b4, color="green", width=0.4)
# plt.bar(a5, b5, color="orange", width=0.4)
# plt.bar(a6, b6, color="red", width=0.4)
#
# plt.title('不同模型的准确率对比')
# plt.ylabel("准确率")
# plt.xlabel("模型")
# plt.savefig('C:\\Users\\Hasee\\Desktop\\2-1-MOSEI-准确率.png', format='png')
# plt.show()

# # # TODO：消融实验
# # waters = ('MSAM_BiGRU', 'MSAM_SL-SAtten', 'MSAM_CS-Atten', 'MSAM_BC-Atten', 'MSAM_I-SAtten', 'MSAM')
# # buy_number = [80.85, 81.25, 81.52, 81.78, 81.91, 82.45]
# # 准确率
# a1 = ['MSAM_BiGRU']
# b1 = [80.85]
# a2 = ['MSAM_SL-SAtten']
# b2 = [81.25]
# a3 = ['MSAM_CS-Atten']
# b3 = [81.52]
# a4 = ['MSAM_BC-Atten']
# b4 = [81.78]
# a5 = ['MSAM_I-SAtten']
# b5 = [81.91]
# a6 = ['MSAM']
# b6 = [82.45]
# plt.bar(a1, b1, color="blue", width=0.4)
# plt.bar(a2, b2, color="yellow", width=0.4)
# plt.bar(a3, b3, color="pink", width=0.4)
# plt.bar(a4, b4, color="green", width=0.4)
# plt.bar(a5, b5, color="orange", width=0.4)
# plt.bar(a6, b6, color="red", width=0.4)
#
# plt.title('不同模型的准确率对比')
# plt.ylabel("准确率")
# plt.xlabel("模型")
# plt.savefig('C:\\Users\\Hasee\\Desktop\\2-1-MOSEI-消融实验-准确率.png', format='png')
# plt.show()


# TODO:第五章
# TODO：MOSI数据集
# CMU-MOSI数据集  准确率对比
# waters = ('GME-LSTM', 'MARN', 'TFN', 'MFRN', 'Multilogue-Net', 'DialogueRNN', 'AMF-BiGRU',  'DLAM')
# 准确率
# buy_number = [76.5, 77.1, 77.1, 78.1, 81.19, 79.8, 82.05, 82.31]
# a1 = ['GME-LSTM']
# b1 = [76.5]
# a2 = ['MARN']
# b2 = [77.1]
# a3 = ['TFN']
# b3 = [77.1]
# a4 = ['MFRN']
# b4 = [78.1]
# a5 = ['Multilogue-Net']
# b5 = [81.19]
# a6 = ['DialogueRNN']
# b6 = [79.8]
# a7 = ['AMF-BiGRU']
# b7 = [82.05]
# a8 = ['DLAM']
# b8 = [82.31]
#
# plt.bar(a1, b1, color="blue", width=0.4)
# plt.bar(a2, b2, color="yellow", width=0.4)
# plt.bar(a3, b3, color="pink", width=0.4)
# plt.bar(a4, b4, color="green", width=0.4)
# plt.bar(a5, b5, color="orange", width=0.4)
# plt.bar(a6, b6, color="gray", width=0.4)
# plt.bar(a7, b7, color="purple", width=0.4)
# plt.bar(a8, b8, color="red", width=0.4)
#
#
# plt.title('不同模型的准确率对比')
# plt.ylabel("准确率")
# plt.xlabel("模型")
# plt.savefig('C:\\Users\\Hasee\\Desktop\\3-1-MOSI-准确率.png', format='png')
# plt.show()


# # # F值
# # buy_number = [73.40, 77.0, 77.90, 77.90, 80.10, 79.48, 82.02, 82.20]
# # # 画不同颜色的柱形图   ---F值
# a1 = ['GME-LSTM']
# b1 = [73.40]
# a2 = ['MARN']
# b2 = [77.0]
# a3 = ['TFN']
# b3 = [77.90]
# a4 = ['MFRN']
# b4 = [77.90]
# a5 = ['Multilogue-Net']
# b5 = [80.10]
# a6 = ['DialogueRNN']
# b6 = [79.48]
# a7 = ['AMF-BiGRU']
# b7 = [82.02]
# a8 = ['DLAM']
# b8 = [82.20]
#
# plt.bar(a1, b1, color="blue", width=0.4)
# plt.bar(a2, b2, color="yellow", width=0.4)
# plt.bar(a3, b3, color="pink", width=0.4)
# plt.bar(a4, b4, color="green", width=0.4)
# plt.bar(a5, b5, color="orange", width=0.4)
# plt.bar(a6, b6, color="gray", width=0.4)
# plt.bar(a7, b7, color="purple", width=0.4)
# plt.bar(a8, b8, color="red", width=0.4)
#
# plt.title('不同模型的F1值对比')
# plt.ylabel("F1值")
# plt.xlabel("模型")
# plt.savefig('C:\\Users\\Hasee\\Desktop\\3-1-MOSI-F1值.png', format='png')
# plt.show()


# TODO：MOSEI数据集
# # 准确率
# a1 = ['MFRN']
# b1 = [77.9]
# a2 = ['Graph-MFN']
# b2 = [76.90]
# a3 = ['CIM-Att']
# b3 = [79.8]
# a4 = ['AMF-BiGRU']
# b4 = [78.48]
# a5 = ['MAM']
# b5 = [81.00]
# a6 = ['DLAM']
# b6 = [81.10]
# plt.bar(a1, b1, color="blue", width=0.4)
# plt.bar(a2, b2, color="yellow", width=0.4)
# plt.bar(a3, b3, color="pink", width=0.4)
# plt.bar(a4, b4, color="green", width=0.4)
# plt.bar(a5, b5, color="orange", width=0.4)
# plt.bar(a6, b6, color="red", width=0.4)
#
# plt.title('不同模型的准确率对比')
# plt.ylabel("准确率")
# plt.xlabel("模型")
# plt.savefig('C:\\Users\\Hasee\\Desktop\\3-1-MOSEI-准确率.png', format='png')
# plt.show()


# # F值
# a1 = ['MFRN']
# b1 = [77.4]
# a2 = ['Graph-MFN']
# b2 = [77.0]
# a3 = ['CIM-Att']
# b3 = [77.60]
# a4 = ['AMF-BiGRU']
# b4 = [78.18]
# a5 = ['MAM']
# b5 = [78.90]
# a6 = ['DLAM']
# b6 = [79.48]
# plt.bar(a1, b1, color="blue", width=0.4)
# plt.bar(a2, b2, color="yellow", width=0.4)
# plt.bar(a3, b3, color="pink", width=0.4)
# plt.bar(a4, b4, color="green", width=0.4)
# plt.bar(a5, b5, color="orange", width=0.4)
# plt.bar(a6, b6, color="red", width=0.4)
#
# plt.title('不同模型的F1值对比')
# plt.ylabel("F1值")
# plt.xlabel("模型")
# plt.savefig('C:\\Users\\Hasee\\Desktop\\3-1-MOSEI-F1值.png', format='png')
# plt.show()

# # TODO：消融实验   准确率
# a1 = ['DLAM_Concat']
# b1 = [70.74]
# a2 = ['DLAM_SAtten']
# b2 = [73.01]
# a3 = ['DLAM_RAtten']
# b3 = [73.27]
# a4 = ['DLAM_RAtten_SAtten']
# b4 = [75.00]
# a5 = ['DLAM_BiGRU']
# b5 = [80.32]
# a6 = ['DLAM_SL-SAtten']
# b6 = [80.72]
# a7 = ['DLAM_Bi-Atten']
# b7 = [81.12]
# a8 = ['DLAM_Tri-Atten']
# b8 = [81.38]
# a9 = ['DLAM_Self-Atten']
# b9 = [81.78]
# a10 = ['DLAM']
# b10 = [82.31]
#
#
# plt.bar(a1, b1, color="blue", width=0.4)
# plt.bar(a2, b2, color="yellow", width=0.4)
# plt.bar(a3, b3, color="pink", width=0.4)
# plt.bar(a4, b4, color="green", width=0.4)
# plt.bar(a5, b5, color="orange", width=0.4)
# plt.bar(a6, b6, color="purple", width=0.4)
# plt.bar(a7, b7, color="gray", width=0.4)
# plt.bar(a8, b8, color="gold", width=0.4)
# plt.bar(a9, b9, color="black", width=0.4)
# plt.bar(a10, b10, color="red", width=0.4)
#
# plt.title('不同模型的准确率对比')
# plt.ylabel("准确率")
# plt.xlabel("模型")
# plt.savefig('C:\\Users\\Hasee\\Desktop\\3-1-MOSI-消融实验-准确率.png', format='png')
# plt.show()

# # TODO：消融实验   F值
a1 = ['DLAM_Concat']
b1 = [71]
a2 = ['DLAM_SAtten']
b2 = [73]
a3 = ['DLAM_RAtten']
b3 = [73]
a4 = ['DLAM_RAtten_SAtten']
b4 = [74]
a5 = ['DLAM_BiGRU']
b5 = [80]
a6 = ['DLAM_SL-SAtten']
b6 = [81]
a7 = ['DLAM_Bi-Atten']
b7 = [81]
a8 = ['DLAM_Tri-Atten']
b8 = [81]
a9 = ['DLAM_Self-Atten']
b9 = [82]
a10 = ['DLAM']
b10 = [82]


plt.bar(a1, b1, color="blue", width=0.4)
plt.bar(a2, b2, color="yellow", width=0.4)
plt.bar(a3, b3, color="pink", width=0.4)
plt.bar(a4, b4, color="green", width=0.4)
plt.bar(a5, b5, color="orange", width=0.4)
plt.bar(a6, b6, color="purple", width=0.4)
plt.bar(a7, b7, color="gray", width=0.4)
plt.bar(a8, b8, color="gold", width=0.4)
plt.bar(a9, b9, color="black", width=0.4)
plt.bar(a10, b10, color="red", width=0.4)

plt.title('不同模型的F1值对比')
plt.ylabel("F1值")
plt.xlabel("模型")
plt.savefig('C:\\Users\\Hasee\\Desktop\\3-1-MOSI-消融实验-F1值.png', format='png')
plt.show()

# 画柱形图
# plt.bar(waters, buy_number, color='pink')

# 画曲线图
# plt.plot(waters, buy_number)
# plt.title('不同模型的错误率对比')
# plt.title('不同模型的F值对比')
# plt.title('不同模型的准确率对比')
# # plt.ylabel("错误率")
# # plt.ylabel("F值")
# plt.ylabel("准确率")
# plt.xlabel("模型")
# plt.savefig('C:\\Users\\Hasee\\Desktop\\1-1-cifar-10-准确率.png', format='png')
# plt.show()
