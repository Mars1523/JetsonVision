import tensorflow as tf

"""@author: Skeletor
this is a basic single-layer perceptron
It'll have constant bias input and step 
activation function to preform the AND operation
Since it's in TensorFlow,it'll be a bit
more simplistic then the pytorch example"""
"""#Here in tensorflow we build models out of empty tensors
#Known values are then plugged in and evaluated
#The training data is this truth table for AND,and has
the 4 possible operand pairs as inputs
and the respective results as outputs"""
T, F = 1, -1
bias = 1
trainingin = [[T, T, bias], [T, F, bias], [F, T, bias], [F, F, bias]]
trainingout = [[T], [F], [F], [F]]


"""Since all the above training data will be constant
the only special TensorFlow object that requires a little
care is this 3x1 tensor of weights. This is a variable
tensor that can be changed.
All values are initalized to pesudo-random RNG
I'm casting it to int32 because of a earlier glitch"""
w = tf.Variable(tf.random_normal([3, 1]))
w1 = tf.Variable(tf.cast(w, tf.int32))
"""The below is a step function. It essentially does this
step(x) =  1 if x > 0; -1 otherwise 
The step function is what we're using 
as a activation function for this neural network
However,step functions are only useful for single
layer networks like this,which is why you'll only see it here"""


def step(x):
    is_greater = tf.greater(x, 0)
    as_float = tf.to_float(is_greater)
    doubled = tf.multiply(as_float, 2)
    return tf.subtract(doubled, 1)


"""These are calculations of the output,error
 and mean squared error of the model.  This is the fowards pass"""
output = step(tf.matmul(trainingin, w1))
output1 = tf.cast(output, tf.int32)
error = tf.subtract(trainingout, output1)
meansquarederror = tf.reduce_mean(tf.square(error))
"""The eval of some tensor functions can update variables
such as the tensor of weights.  First,based off of error
the adjustment is calculated and added  This is the backwards pass"""
delta = tf.matmul(trainingin, error, transpose_a=True)
train = tf.assign(w1, tf.add(w1, delta))
"""Just initaliazing the tensorflow session"""
sess = tf.Session()
sess.run(tf.initializers.global_variables())
"""Now some of you may question why I'm using epochs
A step for a NN is one gradient upgrade
A epoch is one full cycle through the training data
A epoch consists of many steps,and it's dependent on the 
traing data size. For example,I have 30,000 images and a batch size of 100
the epoch should contain 30,000/100 = 300 steps. 
For the purpouses here,steps wouldn't be very useful."""

err, target = 1, 0
epoch, max_epochs = 0, 20

while err > target and epoch < max_epochs:
    epoch += 1
    err, _ = sess.run([meansquarederror, train])
    print("epoch: ", epoch, "mse: ", err)


