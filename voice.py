# import libraries 
import tensorflow as tf
import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np
import random

# read data
voice = pd.read_csv("/Users/wnowak/Desktop/voice.csv")

# create "copies" of dataframe, but with a small amount of noise added to one variable.
# this creates a much larger training set
for i in range(5):
	copy = voice
	copy['meanfreq']=copy['meanfreq']+random.gauss(.0001,.001) # add noice to mean freq var
	voice=voice.append(copy,ignore_index=True) # make voice df 2x as big
	print("shape of df after {0}th intertion of this loop is {1}".format(i,voice.shape))


# pop off the label to make the training set
label = voice.pop("label")

# # converts from dataframe to np array. we need arrays for working with TF lib. 
voice=voice.values

# # convert train labels to one hots (so M/F gender converts to (1,0) vector)
train_labels = pd.get_dummies(label)
# # make np array
train_labels = train_labels.values

# split with sklearn package 
x_train,x_test,y_train,y_test = train_test_split(voice,train_labels,test_size=0.2)

# set dtypes
x_train,x_test,y_train,y_test = np.array(x_train,dtype='float32'), np.array(x_test,dtype='float32'),np.array(y_train,dtype='float32'),np.array(y_test,dtype='float32')

####
# begin using TF to set up NN graph
####

# # there are twenty features
# create place holder for inputs. feed in actual data later when i run session
x = tf.placeholder(tf.float32, [None, 20])

# # # take 20 features  to 1000 nodes in hidden layer. why 1000? i dont know.
w1 = tf.Variable(tf.random_normal([20, 1000],stddev=.5,name='w1'))
# # # add biases for each node
b1 = tf.Variable(tf.zeros([1000]))
# # calculate activations 
hidden_output = tf.nn.softmax(tf.matmul(x, w1) + b1)

# # bring from 1000 nodes to 2 for my output
# w2 = tf.Variable(tf.zeros([10,2]))
w2 = tf.Variable(tf.random_normal([1000, 2],stddev=.5,name='w2'))
b2 = tf.Variable(tf.zeros([2]))

# # placeholder for correct values 
y_ = tf.placeholder("float", [None,2])
# # #implement model. these are predicted ys
y = tf.nn.softmax(tf.matmul(hidden_output, w2) + b2)


# use with logits to add numerical stability (many answers near 0 can throw off CE loss)
loss = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(y, y_, name='xentropy')))

# optimize
opt = tf.train.AdamOptimizer(learning_rate=.005)
train_step = opt.minimize(loss, var_list=[w1,b1,w2,b2])

# for using TF board. i haven't fully investigated this feature of tensorflow
# # for value in [x, w1, loss]:
# #     tf.scalar_summary(value.op.name, value)
# # summaries = tf.merge_all_summaries()


# create mini batch of data to feed in. this speeds up training. 
#send in batch_size to feedforward. not all training data
def get_mini_batch(x,y):
	batch_size = 100
	rows=np.random.choice(x.shape[0], batch_size) 
	return x[rows], y[rows]

# # start session
sess = tf.Session()
# more for tensorboard. to do...
# summary_writer = tf.train.SummaryWriter('voices')
# # summary_writer = tf.train.SummaryWriter('voices', sess.graph)

# # init all vars
init = tf.initialize_all_variables()
sess.run(init)


ntrials = 10
for i in range(ntrials):
    # get mini batch
    a,b=get_mini_batch(x_train,y_train)
    # run train step, feeding arrays of 100 rows each time
    _, cost =sess.run([train_step,loss], feed_dict={x: a, y_: b})
    if i%100 ==0:
    	print("epoch is {0} and cost is {1}".format(i,cost))

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("test accuracy is {}".format(sess.run(accuracy, feed_dict={x: x_test, y_: y_test})))

print("train accuracy is {}".format(sess.run(accuracy, feed_dict={x: x_train, y_: y_train})))



ans = sess.run(y, feed_dict={x: x_test})
print(y_test[0:3])
print("Correct prediction\n",ans[0:3])