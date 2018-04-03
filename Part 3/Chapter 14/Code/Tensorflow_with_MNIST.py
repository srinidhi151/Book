import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/tmp/data/",one_hot=True)
hidden_nodes_1=1000
hidden_nodes_2=1000
hidden_nodes_3=1000
number_classes=10
batch=100
X=tf.placeholder('float',[None,784])
Y=tf.placeholder('float')
def network_model(data):
    layer_1 = {'weight':tf.Variable(tf.random_normal([784, hidden_nodes_1])),
                      'bias':tf.Variable(tf.random_normal([hidden_nodes_1]))}

    layer_2 = {'weight':tf.Variable(tf.random_normal([hidden_nodes_1,hidden_nodes_2])),
                      'bias':tf.Variable(tf.random_normal([hidden_nodes_2]))}

    layer_3 = {'weight':tf.Variable(tf.random_normal([hidden_nodes_2,hidden_nodes_3])),
                      'bias':tf.Variable(tf.random_normal([hidden_nodes_3]))}

    output_layer = {'weight':tf.Variable(tf.random_normal([hidden_nodes_3, number_classes])),
                    'bias':tf.Variable(tf.random_normal([number_classes]))}
    
    l1 = tf.add(tf.matmul(data,layer_1['weight']), layer_1['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,layer_2['weight']), layer_2['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,layer_3['weight']), layer_3['bias'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weight']) + output_layer['bias']

    return output

def train(x):
    pred=network_model(x)
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
    optimizer=tf.train.AdamOptimizer().minimize(cost)
    n_epochs=50
    with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(n_epochs):
                loss=0
                for _ in range(int(mnist.train.num_examples/batch)):
                    epoch_x,epoch_y=mnist.train.next_batch(batch)
                    _,c=sess.run([optimizer,cost],feed_dict={X:epoch_x, Y:epoch_y})
                    loss+=c
                print('Epoch',epoch,'loss',loss)
                correct = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                print('Accuracy:',accuracy.eval({X:mnist.test.images, Y:mnist.test.labels}))
train(X)