from sklearn.naive_bayes import GaussianNB
import numpy
import os
from sklearn import metrics
import math


def naive_bayesian(x_train, y_train, x_test, y_test):
    model = GaussianNB()
    model.fit(x_train, y_train)
    expected = y_test
    predicted = model.predict(x_test)
    cm = metrics.confusion_matrix(expected, predicted)
    return cm


def CalculateD(row_i, row_j):
    size = row_i.shape[0]
    sum_distance = 0.0
    for i in range(size):
        sum_distance += pow((row_i[i] - row_j[i]), 2)
    distance = math.sqrt(sum_distance)
    return distance


def evaluate(cm, each_file):
    tp = cm[0][0]
    fn = cm[0][1]
    fp = cm[1][0]
    tn = cm[1][1]

    pd = tp * 1.0 / (tp + fn)
    pf = fp * 1.0 / (fp + tn)
    precision = tp * 1.0 / (tp + fp)
    recall = tp * 1.0 / (tp + fn)

    g_measure = (2.0 * pd * (1.0 - pf)) / (pd + (1.0 - pf))
    # auc = metrics.auc(pf, pd)
    # mcc = (tp * tn - fp * fn) * 1.0 / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    print("---------------")
    print(each_file)
    print('pd = ', pd)
    print('pf = ', pf)
    print('g_measure = ', g_measure)
    print("----------------")
    # f = open("result.txt", "a", encoding="utf-8")
    # f.write("___________________")
    # f.write("\n")
    # f.write("test data: " + str(each_file))
    # f.write("\n")
    # f.write("pd = " + str(pd))
    # f.write("\n")
    # f.write("pf = " + str(pf))
    # f.write("\n")
    # f.write("g_measure = " + str(g_measure))
    # f.write("\n")
    # f.write("precision = " + str(precision))
    # f.write("\n")
    # f.write("recall = " + str(recall))
    # f.write("\n")
    # f.write("___________________")
    # f.write("\n")
    # f.write("\n")
    # print('mcc = ', mcc)
    # print('precision = ', precision)
    # print('recall= ', recall)
    # print('auc = ', auc)
    # return pd, pf, g_measure,precision, recall, auc


def read_data(data_path):
    '''
    :param data_path: file path
    :return: numpy.darray of data
    '''
    xarray = []
    yarray = []
    num = 0
    with open(data_path, 'rU') as txt:
        for line in txt:
            xline = []
            content = line.strip().split(',')
            for x in content:
                if x == 'Y':
                    yarray.append(1.0)
                    num += 1
                elif x == 'N':
                    yarray.append(0.0)
                else:
                    xline.append(eval(x))
            xarray.append(xline)
    txt.close()
    xarray = numpy.array(xarray)
    yarray = numpy.array(yarray)
    return xarray, yarray


def get_filename(folder):
    '''
    get all name of file
    :param folder: folder path
    :return: all name of txt

    '''

    file_names = []
    for root, subfolders, filenames in os.walk(folder):
        for filename in filenames:
            file_names.append(folder + '/' + filename)
    return file_names


def merge_data(folder, test_name):
    # folder = 'se_data'
    # test_name = 'ant.txt'
    test_file_name = folder + '/' + test_name
    filenames = get_filename(folder)
    filenames = numpy.array(filenames)
    # print(filenames)
    flag = True
    X_train = numpy.array([[]])
    Y_train = numpy.array([])
    for item in filenames:
        if item != test_file_name:
            x, y = read_data(item)
            if flag is True:
                flag = False
                X_train = x[:, :]
                Y_train = y[:]
                continue
            X_train = numpy.vstack((X_train, x))
            Y_train = numpy.hstack((Y_train, y))
    return X_train, Y_train


def get_filtered_TDS(folder, test_name):
    X_train, Y_train = merge_data(folder, test_name)
    test_X_array, test_Y_array = read_data(folder + "/" + test_name)
    X_first_train, Y_first_train = merge_data(folder, test_name)
    # 至此我们得到了测试集和需要筛选的训练集，现在就是要遍历筛选出我们需要的和测试集相似的训练集
    all_group_pair = []  # all_group用来保存测试集和训练集的簇，以便于下一步的筛选
    for train_data_i in range(X_train.shape[0]):
        min_distance = float("inf")  # 将初始距离设置为正无穷
        save_test_x = []
        save_test_y = []
        each_group = []
        each_x_group = []
        each_y_group = []
        each_x_group.append(list(X_first_train[train_data_i]))
        each_y_group.append(Y_first_train[train_data_i])
        for test_data_j in range(test_X_array.shape[0]):
            # 计算训练集中每一个元素与每一个测试集之间的距离大小，选最小的，保存为改测试集的“粉丝”
            distance = CalculateD(X_first_train[train_data_i], test_X_array[test_data_j])
            if distance < min_distance:
                save_test_x = list(test_X_array[test_data_j])
                save_test_y = test_Y_array[test_data_j]
                min_distance = distance
        each_x_group.append(save_test_x)
        each_y_group.append(save_test_y)

        each_group.append(each_x_group)
        each_group.append(each_y_group)
        all_group_pair.append(each_group)
    # all_group_pair:[[[[1.0, 1.0], [2.0, 3.0]], [1.0, 0.0]]] 第一个是训练集，第二个是测试集，第三个是两个集对应的标签
    # 至此，得到每一个彩色球与他最近的白球的组合对，接下来，反着求白球与这些组队中最近的彩色球，这些彩色球将作为训练集
    second_train_X_data = []
    second_train_Y_data = []
    for i in range(test_X_array.shape[0]):
        min_distance = float("inf")  # 将初始距离设置为正无穷
        save_train_x = []
        save_train_y = []
        for j in range(len(all_group_pair)):
            if list(test_X_array[i]) in all_group_pair[j][0]:
                distance = CalculateD(numpy.array(all_group_pair[j][0][0]), numpy.array(all_group_pair[j][0][1]))
                if distance < min_distance:
                    save_train_x = all_group_pair[j][0][0]
                    save_train_y = all_group_pair[j][1][0]
                    min_distance = distance
        if len(save_train_x) > 0:
            second_train_X_data.append(save_train_x)
        if save_train_y == 1.0 or save_train_y == 0.0:
            second_train_Y_data.append(save_train_y)
    second_train_X_data = numpy.array(second_train_X_data)
    second_train_Y_data = numpy.array(second_train_Y_data)
    return second_train_X_data, second_train_Y_data


def NB(folder):
    filenames = os.listdir(folder)
    # print(filenames)
    for each_file in filenames:
        final_train_x, final_train_y = get_filtered_TDS(folder, each_file)
        test_x, test_y = read_data(folder + "/" + each_file)
        cm = naive_bayesian(final_train_x, final_train_y, test_x, test_y)
        evaluate(cm, each_file)
    pass

if __name__ == '__main__':
    NB("source_data")