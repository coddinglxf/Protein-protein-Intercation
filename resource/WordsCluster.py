# codding=-utf-8_
__author__ = 'aim'
import pickle
from pprint import pprint
import numpy

f_vector = open('vector', 'r')
f_dictionary = open('dictionary', 'r')
vector = pickle.load(f_vector)
dictionary = pickle.load(f_dictionary)
f_vector.close()
f_dictionary.close()

index_to_word = dict()
for each in dictionary:
    index_to_word[dictionary[each]] = each

assert len(dictionary) == vector.shape[0]
pprint(vector)
pprint(dictionary)
pprint(index_to_word)


def Top_N(word, TopN=20, dictionary=dict(), vector=numpy.random.rand(2, 3)):
    """

    :param word:
    :param TopN:
    :param dictionary:
    :param vector:
    :return:
    """
    result = list()
    if word not in dictionary:
        print "word not included in the word dictionary, select another one!"
        return False
    else:
        index = dictionary[word]
        word_vector = vector[index]
        word_vector_sum = numpy.sqrt(numpy.sum(word_vector ** 2))
        for k in xrange(TopN):
            max = -10000000
            max_index = -1
            for i in xrange(vector.shape[0]):
                vector_sum = numpy.sqrt(numpy.sum(vector[i] ** 2))
                similarity = numpy.sum(word_vector * vector[i]) / (word_vector_sum * vector_sum)
                if similarity > max and i not in result:
                    max = similarity
                    max_index = i
            result.append(max_index)
    pprint(result)
    for each in result:
        pprint(index_to_word[each])
    return result


input_word = "test"
while input_word is not "exit":
    input_word = raw_input("please input words: \n")
    Top_N(word=input_word,
          dictionary=dictionary,
          vector=vector)