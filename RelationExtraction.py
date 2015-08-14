# coding=utf-8_
from numba.targets.imputils import _IternextResult

__author__ = 'aim'
import theano
import theano.tensor as T
import numpy
from Network import *
from Optimizers import *
from Sentence_Preprocessing import *
from Util_tools import *
from pprint import pprint
import pickle


def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX),
                             borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')


def Relation_Extraction(u,
                        dataset,
                        vector_length,
                        max_path_size,
                        cv,
                        tempfile,
                        windows=[3],
                        filer_size=50,
                        batch_size=2,
                        iterator_times=50,
                        num_class=2,
                        type=None,
):
    global all_result
    rng_need = numpy.random.RandomState(12345)

    input_x, output_y = dataset[0]
    test_x, test_y = dataset[1]

    print "*************the size of train_x***********"
    print input_x.shape[0], input_x.shape[1], output_y.shape[0]
    print "*************test_x************************"
    print test_x.shape[0], test_x.shape[1], test_y.shape[0]

    num_batches = input_x.shape[0] / batch_size
    num_test_batches = test_x.shape[0] / batch_size

    print('batch size is %d' % batch_size)
    print('num of train batch is %d ' % num_batches)
    print('num of test batch is %d ' % num_test_batches)

    print('***************share the train data***************')
    input_x, output_y = shared_dataset(dataset[0])
    print('***************share the test data***************')
    test_x, test_y = shared_dataset(dataset[1])

    words = theano.shared(value=u, name="words")
    # pprint(words.eval())

    index = T.iscalar('index')  # index to a [mini]batch
    x = T.matrix('input')
    y = T.ivector('output')
    layer_conv_input = words[T.cast(x.flatten(), dtype="int32")]. \
        reshape((x.shape[0], 1, x.shape[1], words.shape[1]))

    conv_result = list()
    conv_parameters = list()
    for each_window in windows:
        single_conv = LeNetConvPoolLayer(
            rng_need,
            input=layer_conv_input,
            filter_shape=(filer_size, 1, each_window, vector_length),
            image_shape=(batch_size, 1, max_path_size, vector_length),
            poolsize=(max_path_size - each_window + 1, 1)
        )
        conv_parameters = conv_parameters + single_conv.params
        conv_result.append(single_conv.output.flatten(2))
    # put all the conv wordnet together
    layer1_input = T.concatenate(conv_result, 1)

    layer1 = HiddenLayer(
        rng_need,
        input=layer1_input,
        n_in=filer_size * len(windows),
        n_out=500,
        activation=T.tanh
    )

    classifier = LogisticRegression(input=layer1.output, n_in=layer1.n_out, n_out=num_class)

    # cost = classifier.negative_log_likelihood(y) + 0.001 * (layer1.W ** 2).sum()
    cost = classifier.negative_log_likelihood(y)

    # define test model
    print('build the test model')
    test_model = theano.function(
        [index],
        [classifier.errors(y),
         classifier.p_y_given_x,
         classifier.y_pred,
         y,
         layer_conv_input.shape],
        givens={
            x: test_x[index * batch_size: (index + 1) * batch_size],
            y: test_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # add all the parameters together
    if type is None:
        parameter = layer1.params + conv_parameters + classifier.params
    else:
        parameter = layer1.params + conv_parameters + classifier.params + [words]

    updates = sgd_updates_adadelta(parameter, cost)

    print('build the train model')
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: input_x[index * batch_size: (index + 1) * batch_size],
            y: output_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    it_time = iterator_times
    f_measure_result = list()
    for it in xrange(1, it_time + 1):
        train_sum_error = 0
        for echoes in range(0, num_batches):
            train_error = train_model(echoes)
            train_sum_error = train_sum_error + train_error
            if echoes % 20 == 0:
                print(
                    'train on iterator %d, and on echoes is %d'
                    % (it, echoes)
                )
        print('in iterator time %d, and the error is %f' %
              (it, train_sum_error)
        )
        if it % 1 == 0:
            # test on test data
            error_of_test = 0.0
            test_all = list()
            for echoes in range(0, num_test_batches):
                test_error, p_y_given_x, p_pre, y, input_shape = test_model(echoes)
                error_of_test = error_of_test + test_error
                test_all.append(calculate_f_measures(p_pre, y))
                # print input_shape
            error_of_test /= num_test_batches
            print("test on iterator %d" % it)
            all_result = f_measure(test_all)
            print all_result
            f_measure_result.append(all_result[4])
            tempfile.write(str(it) + ' ' + str(f_measure(test_all)) + '\n')
            print(
                'cv is %d~~~~~~~~~~~~~~the iterator is %d and the errors is %f~~~~~~~~~~~~' %
                (cv, it, error_of_test)
            )
    # dump the words in file, and will try to use it again
    print "dump the trained word vectors"
    f_vector = open('resource//vector', 'w')
    pickle.dump(words.eval(), f_vector)
    f_vector.close()
    return all_result


if __name__ == '__main__':
    experiments_time = 10
    for times in xrange(experiments_time):
        CV = 10
        filename = "keda"
        vector_length = 50
        max_path_size = 30
        preprocessing = Preprocessing("data//" + filename, max_path_size=max_path_size, vector_length=vector_length)

        f_dictionary = open("resource//dictionary", 'w')
        pickle.dump(preprocessing.dictionary, f_dictionary)
        f_dictionary.close()

        f_avg = 0
        p_avg = 0
        r_avg = 0
        for current_cv in range(0, 2):
            dataset = preprocessing.get_data_with_cv(current_cv, CV=CV)
            file = open("wordnet//" + filename + "//" + filename + "_experiments_" + str(times) + "_cv_"
                        + str(current_cv) + ".txt", 'w')
            result = Relation_Extraction(preprocessing.u,
                                         dataset,
                                         vector_length,
                                         cv=current_cv,
                                         tempfile=file,
                                         windows=[2],
                                         filer_size=100,
                                         max_path_size=max_path_size,
                                         batch_size=50,
                                         iterator_times=50,
                                         type="rand")
            # calculate the current measures
            precious = result[0] / (result[0] + result[1] + 0.001)
            recall = result[0] / (result[0] + result[2] + 0.001)
            f_scores = 2 * precious * recall / (precious + recall + 0.001)

            file.write(str(result) + '\n')
            file.write('precious=' + str(precious) + '\t' +
                       'recall=' + str(recall) + '\t' + 'f_scores=' + str(f_scores) + '\n')
            # calculate the average measure
            p_avg += precious
            r_avg += recall
            f_avg += f_scores
            file.write('p_avg=' + str(p_avg / (current_cv + 1)) + '\t' +
                       'r_avg=' + str(r_avg / (current_cv + 1)) + '\t' +
                       'f_avg=' + str(f_avg / (current_cv + 1)) + '\n')
            file.close()