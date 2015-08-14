# coding=utf-8_
_author__ = 'aim'

from pprint import pprint
import numpy


class Preprocessing(object):
    def __init__(self,
                 filename,
                 batch_size=50,
                 max_path_size=10,
                 split_type="@",
                 variance=0.05049235,
                 vector_length=300,
                 word2vec="Vectors",
                 Mode='Single'):
        """

        :param filename: the input filename
        :param batch_size: the batch size of input
        :param max_path_size: define the maximum size of input path size
        :param split_type: define the split type of input, default as "@"
        :param variance: to initial the random word embedding by given variance
        :param vector_length:
        :param word2vec:
        :param Mode:
        """
        self.index_to_word = dict()
        self.dictionary = dict()
        self.batch_size = batch_size
        self.filename = filename
        self.max_path_size = max_path_size
        self.split_type = split_type
        self.variance = variance
        self.result = list()
        # the special symbol for all the sentence
        self.dictionary['</s>'] = 0
        # the word embedding length for each word in
        self.vector_length = vector_length
        self.word2vec = word2vec

        self.all_ones = 0
        self.test_ones = 0
        self.train_ones = 0

        self.index_all_word_in_file()
        self.index_all_path()

        # start to generate all the U for words, for random
        self.u = numpy.random.uniform(low=-self.variance, high=self.variance,
                                      size=(len(self.dictionary), vector_length))
        # start to generate the initial vectors from word2vec
        if Mode is not "Single":
            self.load_vec_from_word2vec()

    def index_all_word_in_file(self):
        print "start to index the word in the file"
        file_cool = open(self.filename, 'r')
        index = 1
        for each in file_cool:
            each = str(each)
            # label should be ignored
            # print each
            words = each.lstrip().rstrip().split(self.split_type)[1:]
            # print words
            for word in words:
                if word not in self.dictionary:
                    self.dictionary[word] = index
                    index += 1
        for each in self.dictionary:
            self.index_to_word[self.dictionary[each]] = each
        file_cool.close()

    def index_all_path(self):
        print "start to index each line in train file"
        file_cool = open(self.filename, 'r')
        flags = 1
        for line in file_cool:
            line = str(line).lstrip().rstrip()
            # label should be ignored
            words = line.lstrip().rstrip().split(self.split_type)[1:]
            if flags % 500 == 0:
                print ("prepreocessing %d in train file" % (flags))
                # print line, line.split(self.split_type)[0]
            flags += 1

            temp_res = list()
            for word in words:
                temp_res.append(self.dictionary[word])
            # if the path is too long, truncate the path as the max_path_size
            length = len(temp_res)
            if length >= self.max_path_size:
                remain = length - self.max_path_size
                left = remain / 2
                right = remain - left
                temp_res = temp_res[left:(length - right)]
            # that means the path is too short, thus the special "</s>" will need
            else:
                remain = self.max_path_size - length
                left = remain / 2
                right = remain - left
                temp_res = [self.dictionary.get("</s>")] * left + \
                           temp_res + \
                           [self.dictionary.get("</s>")] * right
            # assert that the length of preprocessing path should equal to the max_path_size
            assert len(temp_res) == self.max_path_size

            instance = dict()
            instance['label'] = 0 if line.split(self.split_type)[0] == "false" else 1
            instance['path'] = temp_res
            if instance['label'] == 1:
                self.all_ones += 1
            self.result.append(instance)
            # it is necessary to perm the dataset to make the distribution of date randomly
            self.perm = numpy.random.permutation(range(len(self.result)))

    def get_data_with_cv(self, cv, CV=4):
        """
        :param cv:
        :param CV:
        :return:
        """
        print "start to load the data with the given cv"

        self.test_ones = 0
        self.train_ones = 0
        piece = len(self.result) / CV
        start = piece * cv
        end = piece * (cv + 1)

        train_x = list()
        train_y = list()

        test_x = list()
        test_y = list()

        for i in xrange(len(self.result)):
            instance = self.result[self.perm[i]]
            if start <= i <= end:
                test_y.append(instance["label"])
                if instance['label'] == 1:
                    self.test_ones += 1
                test_x.append(instance["path"])
            else:
                train_y.append(instance["label"])
                if instance['label'] == 1:
                    self.train_ones += 1
                train_x.append(instance["path"])
        # assert the ones value in the path should be equal
        assert self.all_ones == self.test_ones + self.train_ones

        # solve the problem that the test data are not divided by batch_size
        # remain=self.batch_size-len(test_y)%self.batch_size
        # for i in xrange(remain):
        #     test_x.append(test_x[0])
        #     test_y.append(test_y[0])

        test_x = numpy.array(test_x)
        test_y = numpy.array(test_y)
        test_y = test_y.T




        test_tuple = tuple((test_x, test_y))


        train_x = numpy.array(train_x)
        train_y = numpy.array(train_y)
        train_y = train_y.T
        train_tuple = tuple((train_x, train_y))

        print ("train ones is %d while test ones is %d" % (self.train_ones, self.test_ones))
        return [train_tuple, test_tuple]

    def load_vec_from_word2vec(self):
        vectors = open("data//" + self.word2vec)
        word_vectors_hash = dict()
        for each in vectors:
            each = str(each)
            words = each.lstrip().lstrip().split(" ")
            vector = [float(words[i]) for i in range(2, len(words))]
            word_vectors_hash[words[0]] = vector
        self.WORDS = numpy.random.rand(self.dictionary.__len__(), self.vector_length)
        for each in self.index_to_word:
            self.WORDS[each] = numpy.array(word_vectors_hash[self.index_to_word[each]])


# pre = Preprocessing(filename="data//Test",
#                     max_path_size=30,
#                     Mode="Static")
# print pre.dictionary.__len__()
# print pre.dictionary
# print pre.index_to_word
# print pre.WORDS.shape
# print pre.WORDS[20]



