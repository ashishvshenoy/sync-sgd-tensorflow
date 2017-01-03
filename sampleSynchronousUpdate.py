import tensorflow as tf
import os


g = tf.Graph()

with g.as_default():

    # creating a model variable on task 0. This is a process running on node vm-48-1
    with tf.device("/job:worker/task:0"):
        w = tf.Variable(tf.ones([10, 1]), name="model")


    # creating 5 reader operators to be placed on different operators
    # here, they emit predefined tensors. however, they can be defined as reader
    # operators as done in "exampleReadCriteoData.py"
    gradients = []
    for i in range(0, 5):
        with tf.device("/job:worker/task:%d" % i):
            reader = tf.ones([10, 1], name="operator_%d" % i)
            # not the gradient compuation here is a random operation. You need
            # to use the right way (as described in assignment 3 desc).
            # we use this specific example to show that gradient computation
            # requires use of the model
            local_gradient = tf.mul(reader, tf.matmul(tf.transpose(w), reader))
            gradients.append(tf.mul(local_gradient, 0.1))


    # we create an operator to aggregate the local gradients
    with tf.device("/job:worker/task:0"):
        aggregator = tf.add_n(gradients)
        #
        assign_op = w.assign_add(aggregator)


    with tf.Session("grpc://vm-48-1:2222") as sess:
        sess.run(tf.initialize_all_variables())
        for i in range(0, 10):
            sess.run(assign_op)
            print w.eval()
        sess.close()
