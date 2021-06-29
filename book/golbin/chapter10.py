import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

learning_rate = 0.001
total_epoch = 30
batch_size = 128
n_input = 28
n_step = 28
n_hidden = 128
n_class = 10

X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.float32, [None, n_class])

W = tf.placeholder(tf.random_normal([n_hidden, n_class])
b = tf.placeholder(tf.random_normal([n_class])

cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

outputs = tf.transpose(outputs, [1,0,2])
outputs = outputs[-1]

model = tf.matmul(outputs, W) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#학습
sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples / batch_size)

#실제학습
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(training_epoch):
	total_cost = 0
	
	for i in range(total_batch):
		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
		batch_xs = batch_xs.reshape((batch_size, n_step, n_input))
		
		_, cost_val = sess.run([optimizer, cost], feed_dict={X:babtch_xs, Y:batch_ys})
		total_cost += cost_val
	
	print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print('최적화 완료')

is_correct = tf.equal(tf.argmax(model,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

test_batch_size = len(mnist.test.images)
test_xs = mnist.test.images.reshape(test_batch_size, n_step, n_input)
test_ys = mnist.test.labels

print('정확도:', sess.run(accuracy, feed_dict={X:test_xs, Y:test_ys}))

#

import tensorflow as tf
import numpy as np

char_Arr = ['a','b','c','d','e','f','g',
			'h','i','j','k','l','m','n',
			'o','p','q','r','s','t','u',
			'v','w','x','y','z']

num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

seq_data = ['word','wood','deep','dive','cold','cool','load','love','kiss','kind']

def make_batch(deq_data):
	input_batch = []
	target_batch = []
	
	for seq in seq_data:
		input = [num_dic[n] for n in seq[:-1]]
		target = num_dic[seq[-1]]
		input_batch.append(np.eye(dic_len)[input])
		target_batch.append([target])
	
	return input_batch, target_batch

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

learning_rate = 0.01
n_hidden = 128
total_epoch = 30

n_step = 3
n_input = n_class = dic_len

X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.int32, [None])

W = tf.Variable(tf.random_normal([n_hidden, n_class])
b = tf.Variable(tf.random_normal([n_class])

cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)
cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)

outputs = tf.transpose(outputs, [1,0,2])
outputs = outputs[-1]
model = tf.matmul(outputs, W) + b

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#학습
sess = tf.Session()
sess.run(tf.global_variables_initializer())

input_batch, target_batch = make_batch(seq_data)

for epoch in range(total_epoch):
	_, loss = sess.run([optimizer, cost], feed_dict={X:input_batch, Y:target_batch})
	
	print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

print('최적화 완료')

prediction = tf.cast(tf.argmax(model,1), tf.int32)
prediction_check = tf.equal(prediction, Y)
accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

input_batch, target_batch = make_batch(seq_data)

predict, accuracy_val = sess.run([prediction, accuracy], feed_dict-{X:input_batch, Y:target_batch})

predict_words = []
for idx, val in enumerate(seq_data):
	last_char = char_arr[predict[idx]]
	predict_words.append(val[:3] + last_char)

print('입력값:', [w[:3] + ' ' for w in seq_data])
print('예측값:', predict_words)
print('정확도:', accuracy_val)

#

import tensorflow as tf
import numpy as np

char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz단어나무놀이소녀키스사랑']
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

seq_data = [['word','단어'],['wood','나무'],
			['game','게임'],['girl','소녀'],
			['kiss','키스'],['love','사랑']]

def make_batch(seq_data):
	input_batch = []
	output_batch = []
	target_batch = []
	
	for seq in seq_data:
		input = [num_dic[n] for n in seq[0]]
		output = [num_dic[n] for n in ('S' + seq[1])]
		target = [num_dic[n] for n in (seq[1] + 'E')]
		
		input_batch.append(np.eye(dic_len)[input])
		output_batch.append(np.eye(dic_len)[output])
		target_batch.append(target)
	
	return input_batch, output_batch, target_batch

learning_rate = 0.01
n_hidden = 128
total_epoch = 100

n_class = n_input = dic_len

#신경망
enc_input = tf.placeholder(tf.float32, [None, None, n_input])
dec_input = tf.placeholder(tf.float32, [None, None, n_input])
targets = tf.placeholder(tf.int64, [None, None])

with tf.variable_scope('encode'):
	enc_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
	enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell,output_keep_prob=0.5)
	
	outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input, dtype=tf.float32)

with tf.variable_scope('decode'):
	dec_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
	dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell,output_keep_prob=0.5)
	
	outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_input, initial_state=enc_states, dtype=tf.float32)

model  =tf.layers.dense(outputs, n_class, activation=None)

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=targets))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#학습
sess = tf.Session()
sess.run(tf.global_variables_initializer())

input_batch, output_batch, target_batch = make_batch(seq_data)

for epoch in range(total_epoch):
	_, loss = sess.run([optimizer,cost], feed_dict={enc_input:input_batch,
													dec_input:output_batch,
													targets:target_batch})
	print('Epoch:', '%04d' % (epoch+1), 'cost =', '{:.6f}'.format(loss))

print('최적화 완료!')

#확인
def translate(word):
	seq_data = [word,'P'*len(word)]
	
	input_batch, output_batch, target_batch = make_batch([seq_data])

	prediction = tf.argmax(model,2)

	result = sess.run(prediction, feed_dict={enc_input:input_batch,
											 dec_input:output_batch,
											 targets:target_batch})

	decoded = [char_arr[i] for i in result[0]]

	end = decoded.index('E')
	translated = ''.join(decoded[:end])

	return translated

print('\n=== 번역 테스트 ===')

print('word ->', translate('word'))
print('wodr ->', translate('wodr'))
print('love ->', translate('love'))
print('loev ->', translate('loev'))
print('abcd ->', translate('abcd'))