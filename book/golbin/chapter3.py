import tensorflow as tf

hello=tf.constant('Hello, TensorFlow!')
print(hello)

a=tf.constant(10)
b=tf.constant(32)
c=tf.add(a,b)
print(c)

sess=tf.Session()

print(sess.run(hello))
print(sess.run([a,b,c]))

sess.close()


X=tf.placeholder(tf.float32,[None,3])
print(X)

x_data = [[1,2,3],[4,5,6]]

W = tf.Variable(tf.random_normal([3,2]))
//W = tf.Variable([[0.1,0.1],[0,2,0.2],[0.3,0.3]]
b = tf.Variable(tf.random_normal([2,1]))

expr = tf.matmul(X,W)+b

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(x_data)
print(sess.run(W))
print(sess.run(b))
print(sess.run(expr,feed_dict={X:x_data}))

sess.close()