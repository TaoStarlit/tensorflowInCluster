import tensorflow as tf

sess=tf.Session()

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0, dtype=tf.float32)

addNode = node1 + node2

print(node1,node2,addNode)
print(sess.run(addNode))


a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b

print(a,b,adder_node)

print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))



