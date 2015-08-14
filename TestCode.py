__author__ = 'Administrator'
import theano
import theano.tensor as T
import numpy
from Network import *

u = numpy.random.rand(3, 4)
words = theano.shared(value=u, name='words')

print words.eval()

x = T.matrix('input')
y = T.ivector('output')
layer_input = words[T.cast(x.flatten(), dtype="int32")].reshape((x.shape[0], x.shape[1] * words.shape[1]))

classifier = LogisticRegression(input=layer_input, n_in=8, n_out=2)

loss = classifier.negative_log_likelihood(y)

params = classifier.params + [words]

grads = T.grad(loss, params)

updates = \
    [
        (param_i, param_i - 1 * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]
# grads = T.grad(y, words)
print "build the function"
train = theano.function(inputs=[x, y],
                        outputs=[loss, classifier.W, classifier.b],
                        updates=updates,
                        )

input_x = numpy.array([[0, 1], [1, 2]])
input_y = numpy.array([0, 1])

for i in xrange(5):
    out = train(input_x, input_y)
    print out
    print "..........................................."
print words.eval()