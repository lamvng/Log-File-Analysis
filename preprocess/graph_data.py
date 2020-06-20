import matplotlib.pyplot as plt
import math
import numpy as np
from preprocess.load_file import load_file


# Plot rectified function
def rectified(x):
    return max(0.0, x)
def plot_rectified():
    series_in = [x for x in range(-50, 100)]
    series_out = [rectified(x) for x in series_in]
    plt.rcParams["figure.figsize"] = (15, 8)
    plt.plot(series_in, series_out)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Hàm ReLu', fontsize=25)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.show()


# Plot sigmoid function
def sigmoid(x):
  return 1 / (1 + math.exp(-x))
def plot_sigmoid():
    series_in = [float(x) for x in np.arange(-10, 10, 0.01)]
    series_out = [sigmoid(x) for x in series_in]
    plt.rcParams["figure.figsize"] = (15,8)
    plt.plot(series_in, series_out)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Hàm Sigmoid', fontsize=25)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.show()


# Plot softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
def plot_softmax():
    series_in = np.arange(-10, 10, 0.1)
    series_out = softmax(series_in)
    plt.rcParams["figure.figsize"] = (15,8)
    plt.plot(series_in, series_out)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Hàm Softmax', fontsize=25)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.show()


# Plot dataset before encoding
def plot_train_before_encode(label_train):
    label_train.plot.bar(x="Số lượng", y="Nhãn", figsize=(18, 10))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Phân bố các nhãn gốc của bộ dữ liệu training', fontsize=25)
    plt.xlabel('Nhãn', fontsize=20)
    plt.ylabel('Số lượng', fontsize=20)

def plot_test_before_encode(label_test):
    label_test.plot.bar(x="Số lượng", y="Nhãn", figsize=(18, 10))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Phân bố các nhãn gốc của bộ dữ liệu testing', fontsize=25)
    plt.xlabel('Nhãn', fontsize=20)
    plt.ylabel('Số lượng', fontsize=20)



# Plot dataset after encoding
def plot_train_after_encode(attack_type_train):
    index = ['normal', 'dos', 'probe', 'u2r', 'r2l']
    # attack_type.plot.bar(x="Số lượng", y="Nhãn", figsize=(18,10))
    plt.bar(index, attack_type_train)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Phân bố các nhãn của bộ dữ liệu training', fontsize=25)
    plt.xlabel('Nhãn', fontsize=20)
    plt.ylabel('Số lượng', fontsize=20)
    # plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def plot_test_after_encode(attack_type_test):
    index = ['normal', 'dos', 'probe', 'u2r', 'r2l']
    # attack_type.plot.bar(x="Số lượng", y="Nhãn", figsize=(18,10))
    plt.bar(index, attack_type_test)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Phân bố các nhãn của bộ dữ liệu testing', fontsize=25)
    plt.xlabel('Nhãn', fontsize=20)
    plt.ylabel('Số lượng', fontsize=20)
    # plt.legend(['train', 'test'], loc='upper left')
    plt.show()




# Plotting a Bar Graph to compare the models
def important_features(top_columns, top_score):
    plt.bar(top_columns, top_score)
    plt.xlabel('Feature Labels')
    plt.ylabel('Feature Importances')
    plt.title('Comparison of the most important features')
    plt.show()


