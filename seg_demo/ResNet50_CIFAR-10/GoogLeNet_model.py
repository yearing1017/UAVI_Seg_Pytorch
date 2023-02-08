import tensorflow as tf 

def variable_weight(name="weight", shape=None, stddev=0.02, wd=None):

	weight = tf.get_variable(name = name, shape = shape,
		initializer = tf.truncated_normal_initializer(stddev = stddev))

	if(wd != None):
		weight_decay = tf.multiply(tf.nn.l2_loss(weight), wd, name='weight_loss')
		tf.add_to_collection('loss', weight_decay)

	return weight

def conv_layer(name, x, filter_height, filter_width, stride, num_filters, padding = 'SAME'):

	input_channels = int(x.get_shape()[-1])

	with tf.variable_scope(name) as scope:
		W = tf.get_variable('weights', shape = [filter_height, filter_width, input_channels, num_filters], 
			initializer = tf.random_normal_initializer(mean = 0, stddev = 0.01))
		b = tf.get_variable('biases', shape = [num_filters], 
			initializer = tf.constant_initializer(0.0))
	
	conv = tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = padding)
	
	z = tf.nn.bias_add(conv, b)

	return tf.nn.relu(z, name = scope.name)

def dropout(name, x, keep_prob):

	return tf.nn.dropout(x, keep_prob = keep_prob, name = name)

def pool_layer(name, x, Kernel_shape,Stride_shape, padding='SAME', pooling_Mode='Max_Pool'):

	if pooling_Mode=='Max_Pool':
		return tf.nn.max_pool(x, Kernel_shape, Stride_shape, padding = padding, name = name)

	if pooling_Mode=='Avg_Pool':
		return tf.nn.avg_pool(x, Kernel_shape, Stride_shape, padding = padding, name = name)

def inception_layer(name, x, conv_11_size, conv_33_reduce_size, conv_33_size, conv_55_reduce_size, conv_55_size, pool_size):

	with tf.variable_scope(name) as scope:

		conv_11 = conv_layer((name+'conv_11'), x, 1, 1, 1, conv_11_size)

		conv_33_reduce = conv_layer((name+'conv_33_reduce'), x, 1, 1, 1, conv_33_reduce_size)
		conv_33 = conv_layer((name+'conv_33'),conv_33_reduce, 3, 3, 1, conv_33_size)

		conv_55_reduce = conv_layer((name+'conv_55_reduce'), x, 1, 1, 1, conv_55_reduce_size)
		conv_55 = conv_layer((name+'conv_55'),conv_55_reduce, 5, 5, 1, conv_55_size)

		pool = pool_layer((name+'pool'), x, [1,3,3,1], [1,1,1,1])

		conv_pool = conv_layer((name+'conv_pool'), pool, 1, 1, 1, pool_size)

		return tf.concat([conv_11,conv_33,conv_55,conv_pool], 3, name=scope.name)

class GoogLeNet(object):

	def __init__(self, x, keep_prob, num_classes):
		self.X = x
		self.KEEP_PROB = keep_prob
		self.NUM_CLASSES = num_classes
		self._build_model()

	def _build_model(self):

		conv_1 = conv_layer('conv_1', self.X, 7,7,2,64)
		max_pool_1 = pool_layer('max_pool_1',conv_1,[1,3,3,1],[1,2,2,1])

		conv_2 = conv_layer('conv_2',max_pool_1,3,3,1,192)
		conv_3 = conv_layer('conv_3',conv_2,3,3,1,192)
		max_pool_2 = pool_layer('max_pool_2',conv_3,[1,3,3,1],[1,2,2,1])

		inception_3a = inception_layer('inception_3a',max_pool_2,64,96,128,16,32,32)
		inception_3b = inception_layer('inception_3b',inception_3a,128,128,192,32,96,64)
		max_pool_3 = pool_layer('max_pool_3',inception_3b,[1,3,3,1],[1,2,2,1])

		inception_4a = inception_layer('inception_4a',max_pool_3,192,96,208,16,48,64)
		inception_4b = inception_layer('inception_4b',inception_4a,160,112,224,24,64,64)
		inception_4c = inception_layer('inception_4c',inception_4b,128,128,256,24,64,64)
		inception_4d = inception_layer('inception_4d',inception_4c,112,144,288,32,64,64)
		inception_4e = inception_layer('inception_4e',inception_4d,256,160,320,32,128,128)
		max_pool_4 = pool_layer('max_pool_4',inception_4e,[1,3,3,1],[1,2,2,1])

		inception_5a = inception_layer('inception_5a',max_pool_4,256,160,320,32,128,128)
		inception_5b = inception_layer('inception_5b',inception_5a,384,192,384,48,128,128)
		
		avg_pool_1 = pool_layer('avg_pool_1',inception_5b,[1,8,8,1],[1,1,1,1],pooling_Mode='Avg_Pool')

		# dropout_1=self.dropout('dropout_1',avg_pool_1, self.KEEP_PROB)
		# linear_1=self.conv_layer('linear-1',avg_pool_1,1,1,1,self.NUM_CLASSES)
		# linear_1=tf.reshape(linear_1,[-1,1*1*self.NUM_CLASSES])

		fc_input = tf.reshape(avg_pool_1, [-1,1024])
		dropout_1 = dropout('dropout_1',fc_input, self.KEEP_PROB)

		with tf.variable_scope('linear_1') as scope:
			input_channels = dropout_1.get_shape()[-1]
			W = variable_weight(shape = [input_channels,self.NUM_CLASSES], wd=0.004)
			b = tf.get_variable(name="b", shape = [self.NUM_CLASSES],
				initializer=tf.constant_initializer(0.0))
			# x_return=tf.nn.softmax(tf.matmul(drop_2,_w)+_b,name=scope.name)
			linear_1 = tf.matmul(dropout_1, W) + b
			# softmax_1=tf.nn.softmax(linear_1)

		self.output = linear_1