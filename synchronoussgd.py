import tensorflow as tf
import os
import time

tf.logging.set_verbosity(tf.logging.DEBUG)
num_features = 33762578
eta = -0.01
absolute_path = "/home/ubuntu/big_data/tfrecords"
filepaths = [["00","01","02","03","04"],["05","06","07","08","09"],["10","11","12","13","14"],["15","16","17","18","19"],["20","21"]]
gradients_device_1 = []
gradients_device_2 = []
gradients_device_3 = []
gradients_device_4 = []
gradients_device_5 = []
gradients = []

g = tf.Graph()
iterations = 20000000
test_iteration = 200

error_rate = 0


def read_and_decode_single() :
	#Trains on one sample from all 5 devices
	for i in range(0,5):
		with tf.device("/job:worker/task:%d" % i):
			filename_queue = tf.train.string_input_producer([absolute_path+path for path in filepaths[i]],num_epochs=None)
			reader = tf.TFRecordReader()
			_, serialized_data = reader.read(filename_queue)
			features = tf.parse_single_example(serialized_data,
													   features={
															'label': tf.FixedLenFeature([1], dtype=tf.int64),
															'index' : tf.VarLenFeature(dtype=tf.int64),
															'value' : tf.VarLenFeature(dtype=tf.float32),
													   }
													  )

			label = features['label']
			index = features['index']
			value = features['value']


			#dense_feature = tf.sparse_to_dense(tf.sparse_tensor_to_dense(index),[num_features,],tf.sparse_tensor_to_dense(value))

			local_gradient = train_model(features,label,i)
			if(i == 0) :
				gradients_device_1.append(local_gradient)
			if(i == 1):
				gradients_device_2.append(local_gradient)
			if(i == 2):
				gradients_device_3.append(local_gradient)
			if(i == 3):
				gradients_device_4.append(local_gradient)
			if(i == 4):
				gradients_device_5.append(local_gradient)
			#gradients.append(local_gradient)


def read_and_test_single() :
	with tf.device("/job:worker/task:%d" % 0):
		filename_queue = tf.train.string_input_producer([absolute_path+"22"],num_epochs=2)
		reader = tf.TFRecordReader()
		_, serialized_data = reader.read(filename_queue)
		features = tf.parse_single_example(serialized_data,
												   features={
														'label': tf.FixedLenFeature([1], dtype=tf.int64),
														'index' : tf.VarLenFeature(dtype=tf.int64),
														'value' : tf.VarLenFeature(dtype=tf.float32),
												   }
												  )

		label = features['label']
		index = features['index']
		value = features['value']

		dense_feature = tf.sparse_to_dense(tf.sparse_tensor_to_dense(index),[num_features,],tf.sparse_tensor_to_dense(value))

		predictions = tf.reduce_sum(tf.mul(w,dense_feature))
		signs = tf.sign(predictions)
		signs = tf.cast(signs,tf.int64)
		ones = tf.constant([1], dtype=tf.int64)
		temp =  tf.sub(ones,tf.mul(signs,label))
		temp = tf.cast(temp, tf.float32)
		this_is_error = tf.mul(0.5,temp)
	return this_is_error



def train_model(feature, label, device) :
	with tf.device("/job:worker/task:%d" % device):
		#local_gradient = tf.Variable(tf.zeros([num_features,1]))

		#filtered_feature = tf.gather(feature['value'].values,feature['index'].values)
		#filtered_local_gradient = tf.gather(local_gradient, feature['index'].values)
		filtered_w = tf.gather(w, feature['index'].values)
		p = tf.reduce_sum(tf.mul(filtered_w,feature['value'].values))
		label = tf.cast(label, tf.float32)
		q = tf.mul(label,p)
		r = tf.mul(tf.sigmoid(q)-1,feature['value'].values)

		filtered_local_gradient = tf.mul(label,r)
		filtered_local_gradient = tf.mul(filtered_local_gradient, eta)

		sparse_filtered_local_gradient = tf.SparseTensor(shape=[num_features], indices=[feature['index'].values], values=filtered_local_gradient)
		return sparse_filtered_local_gradient



def loadTestData(filename) :
	filename_queue = tf.train.string_input_producer([filename],num_epochs=None)
	reader = tf.TFRecordReader()
	_, serialized_data = reader.read(filename_queue)
	batch_serialized_examples = tf.train.shuffle_batch([serialized_data], batch_size=1,capacity=2,min_after_dequeue=1)
	test_features = tf.parse_single_example(serialized_data,
													   features={
															'label': tf.FixedLenFeature([1], dtype=tf.int64),
															'index' : tf.VarLenFeature(dtype=tf.int64),
															'value' : tf.VarLenFeature(dtype=tf.float32),
													   }
													  )
	dense = tf.sparse_to_dense(tf.sparse_tensor_to_dense(test_features['index']),[num_features,],tf.sparse_tensor_to_dense(test_features['value']))
	return dense,test_features['label']



with g.as_default():
	with tf.device("/job:worker/task:0"):
		w = tf.Variable(tf.ones([num_features]), name="model")

	trained_gradients = read_and_decode_single()
	tested_error = read_and_test_single()

	#gradients_op = tf.group(gradients_device_1,gradients_device_2,gradients_device_3,gradients_device_4,gradients_device_5)


	with tf.device("/job:worker/task:0"):
		gradients.append(gradients_device_1[0])
		gradients.append(gradients_device_2[0])
		gradients.append(gradients_device_3[0])
		gradients.append(gradients_device_4[0])
		gradients.append(gradients_device_5[0])

		dense_gradients = []
		aggregate_gradients = []
		temp = gradients[0]
		i=0
		while i in range(1, len(gradients)):
			temp = tf.sparse_add(temp,gradients[i])
		assign_op = tf.scatter_add(w,  tf.reshape(temp.indices, [-1]) ,  tf.reshape(temp.values, [-1]))
		#dense_gradient = tf.sparse_to_dense(tf.transpose(temp.indices),[num_features],temp.values )
		#dense_gradient = tf.mul(dense_gradient, eta)
		#assign_op = w.assign_add(dense_gradient)



	config = tf.ConfigProto(log_device_placement=True)
	with tf.Session("grpc://vm-4-1:2222", config=config) as sess:
		coordinator = tf.train.Coordinator()
		init_op = tf.initialize_all_variables()
		local_init_op = tf.initialize_local_variables()
		test_dense_features_batch, test_labels_batch = loadTestData(absolute_path+"22")
		#test_dense_features_batch, test_labels_batch = tf.train.shuffle_batch([test_dense_features_single, test_labels_single], batch_size=1,capacity=2,min_after_dequeue=1)
		sess.run(init_op)
		sess.run(local_init_op)
		tf.train.start_queue_runners(sess=sess, coord=coordinator)

		try:
			step = 0
			count = 0
			error_rates = []
			while not coordinator.should_stop():
				if(count!=0 and count%50 == 0 ):
					errors = []
					test = 0
					try:
						while test!=1000:
							result = sess.run(tested_error)
							errors.append(tested_error.eval()[0])
							test = test +1
							print "Test Sample : "+str(test)
					except tf.errors.OutOfRangeError :
						print "EOF in test file"
					error_count = 0
					for e in errors :
						if e==1 :
							error_count+=1
					error_rate = ((error_count*1.0)/50)*100
					print "Error Rate : "+str(error_rate)
					error_rates.append(error_rate)
					with open("error_file_synchronous", "a+") as error_file :
						error_file.write("\nError Rate after "+str(count)+" iteration "+str(error_rate))
				count+=1
				print "Samples Read : "+str(count*5)

				start_time = time.time()
				print "Getting dense features and labels"
				sess.run(gradients)
				duration = time.time() - start_time
				print "Duration of training : "+str(duration)
				start_time = time.time()
				print "Running assign_op"
				sess.run(assign_op)
				duration = time.time() - start_time
				print "Duration of assigning : "+str(duration)
				print "Evaluating w"
				print w.eval()
				
		except tf.errors.OutOfRangeError:
			print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
		finally:
			coordinator.request_stop()
		coordinator.join(threads)

		print error_rates

		tf.train.SummaryWriter("%s/sync_sgd" % (os.environ.get("TF_LOG_DIR")), sess.graph)
		sess.close()
