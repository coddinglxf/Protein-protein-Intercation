"""
Created on Mon Apr 20 22:49:42 2015
generated used to calculated the f_scores
@author: aim
"""
def calculate_precious(y_pred, y):
    pre = y_pred.T
    output = y.T
    t_2_t = 0.0
    size = pre.size
    for i in range(size):
        a = int(pre[i])
        b = int(output[i])
        if a == b:
            t_2_t += 1
    return t_2_t


def calculate_f_measures_svm(y_pred, y):
    assert len(y) == len(y_pred)
    pre = y_pred
    output = y

    t_2_t = 0.0
    t_2_f = 0.0
    f_2_t = 0.0
    f_2_f = 0.0

    for i in xrange(len(y)):
        a = int(pre[i])
        b = int(output[i])
        if a == 1 and b == 1:
            t_2_t += 1
        if a == 0 and b == 1:
            t_2_f += 1
        if a == 1 and b == 0:
            f_2_t += 1
        if a == 0 and b == 0:
            f_2_f += 1
    temp = list()
    temp.append(t_2_t)
    temp.append(t_2_f)
    temp.append(f_2_t)
    temp.append(f_2_f)
    return temp


def calculate_f_measures(y_pred, y):
    assert len(y) == len(y_pred)
    pre = y_pred.T
    output = y.T

    t_2_t = 0.0
    t_2_f = 0.0
    f_2_t = 0.0
    f_2_f = 0.0
    size = pre.size
    for i in range(size):
        a = int(pre[i])
        b = int(output[i])
        if a == 1 and b == 1:
            t_2_t += 1
        if a == 0 and b == 1:
            t_2_f += 1
        if a == 1 and b == 0:
            f_2_t += 1
        if a == 0 and b == 0:
            f_2_f += 1
    temp = list()
    temp.append(t_2_t)
    temp.append(t_2_f)
    temp.append(f_2_t)
    temp.append(f_2_f)
    return temp


def f_measure(inputs):
    t_2_t = 0.0
    t_2_f = 0.0
    f_2_t = 0.0
    f_2_f = 0.0
    for i in xrange(len(inputs)):
        temp = inputs[i]
        t_2_t = t_2_t + temp[0]
        t_2_f = t_2_f + temp[1]
        f_2_t = f_2_t + temp[2]
        f_2_f = f_2_f + temp[3]
    p = (t_2_t) / (t_2_t + t_2_f + 0.001)
    r = (t_2_t) / (t_2_t + f_2_t + 0.001)
    f = 2 * p * r / (p + r + 0.001)
    precious = (t_2_t + f_2_f) / (t_2_t + t_2_f + f_2_t + f_2_f)
    print("the precision is %f, the recall is %f, the f_measure is %f, the pre is %f " % (p, r, f, precious))
    return [t_2_t, t_2_f, f_2_t, f_2_f, f, precious]