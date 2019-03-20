import os,sys

import tensorflow as tf 
import numpy as np

from base_nn import BaseNN 


class AutoencoderNN(BaseNN):
	def __init__(self,class_number=5):


		self.network_param = None


		self.network_info = None

	def getTrainingParameters(self):
		param = dict()
		
		return param

	def getValidationParameters(self):
		param = dict()
		
		return param

	def getEvaluationParameters(self):
		return self.getValidationParameters()


	def encoder(self,data_tuple=None, is_training=False, ps_device="/cpu:0", w_device="/gpu:0"):
		if data_tuple:
			inputs=data_tuple[0]

		else:
			raise NoDataError

		with tf.variable_scope('encoder'):

			input_shape=inputs.get_shape().as_list()
			
			inputs = tf.reshape(tensor=inputs,shape=(-1,np.prod(input_shape[1:])))
			# inputs = tf.layers.batch_normalization(inputs, training=is_training, name="batch_normalisation_1")
			# self.print_tensor_shape(inputs, "inputs")

			matmul_1 = self.matmul(inputs  , 2000, name='matmul_1_op', activation=tf.nn.sigmoid, ps_device=ps_device, w_device=w_device)
			drop_op_1  = tf.nn.dropout( matmul_1, 1)
			matmul_2 = self.matmul(drop_op_1, 1500, name='matmul_2_op', activation=tf.nn.sigmoid, ps_device=ps_device, w_device=w_device)
			drop_op_2  = tf.nn.dropout( matmul_2, 1)
			matmul_3 = self.matmul(drop_op_2, 1000, name='matmul_3_op', activation=tf.nn.sigmoid, ps_device=ps_device, w_device=w_device)
			drop_op_3  = tf.nn.dropout( matmul_3, 1)
			matmul_4 = self.matmul(drop_op_3, 500, name='matmul_4_op', activation=tf.nn.sigmoid, ps_device=ps_device, w_device=w_device)
			drop_op_4  = tf.nn.dropout( matmul_4, 1)
			matmul_5 = self.matmul(drop_op_4, 250, name='matmul_5_op', activation=tf.nn.sigmoid, ps_device=ps_device, w_device=w_device)
			drop_op_5  = tf.nn.dropout( matmul_5, 1)
			matmul_6 = self.matmul(drop_op_5, 100, name='matmul_6_op', activation=tf.nn.sigmoid, ps_device=ps_device, w_device=w_device)
			
			
		return matmul_6



	def decoder(self,data_tuple=None,output_shape=3006, is_training=False, ps_device="/cpu:0", w_device="/gpu:0"):
		if data_tuple:
			inputs=data_tuple[0]

		else:
			raise NoDataError


		with tf.variable_scope('decoder'):

			matmul_1 = self.matmul(inputs  , 250, name='matmul_1_op', activation=tf.nn.sigmoid, ps_device=ps_device, w_device=w_device)
			drop_op_1  = tf.nn.dropout( matmul_1, 1)
			matmul_2 = self.matmul(drop_op_1, 500, name='matmul_2_op', activation=tf.nn.sigmoid, ps_device=ps_device, w_device=w_device)
			drop_op_2  = tf.nn.dropout( matmul_2, 1)
			matmul_3 = self.matmul(drop_op_2, 1000, name='matmul_3_op', activation=tf.nn.sigmoid, ps_device=ps_device, w_device=w_device)
			drop_op_3  = tf.nn.dropout( matmul_3, 1)
			matmul_4 = self.matmul(drop_op_3, 1500, name='matmul_4_op', activation=tf.nn.sigmoid, ps_device=ps_device, w_device=w_device)
			drop_op_4  = tf.nn.dropout( matmul_4, 1)
			matmul_5 = self.matmul(drop_op_4, 2000, name='matmul_5_op', activation=tf.nn.sigmoid, ps_device=ps_device, w_device=w_device)
			drop_op_5  = tf.nn.dropout( matmul_5, 1)
			matmul_6 = self.matmul(drop_op_5, np.prod(output_shape[1:]), name='matmul_6_op', activation=tf.nn.sigmoid, ps_device=ps_device, w_device=w_device)
			
			final_layer = tf.reshape(tensor=matmul_6,shape=(-1,output_shape[1],output_shape[2]))
		return final_layer



	def loss(self, logits, data_tuple):
		original=data_tuple[0]

		input_shape=original.get_shape().as_list()
			
		#original = np.reshape(tensor=original,shape=(-1,np.prod(input_shape[1:])))

		with tf.variable_scope('loss'):
			loss=tf.reduce_mean(tf.square(logits-original))
			# loss=tf.nn.sigmoid_cross_entropy_with_logits(labels=original,logits=logits)
			# loss=tf.reduce_mean(loss)
			self.print_tensor_shape(loss, "loss")

		return loss

	def training(self, loss, learning_rate=1e-3, decay_steps=10000, decay_rate=0.96, staircase=False):
		with tf.variable_scope('optimisation'):
			global_step = tf.Variable(0, name='global_step', trainable=False)
			#optimizer = tf.train.AdamOptimizer(learning_rate)
			optimizer = tf.train.GradientDescentOptimizer(learning_rate)
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				train_op = optimizer.minimize(loss, global_step=global_step)

		return train_op

	def metrics(self, logits, data_tuple, metrics_collection_name='collection_metrics'):
		labels=data_tuple[-1]
		num_label=len(self.tfrecord_info['class_names'])
		
		with tf.variable_scope(metrics_collection_name):
			logits=tf.nn.softmax(logits)
			labels=tf.one_hot(tf.cast(labels,tf.int32),num_label)[:,0,:]
			self.print_tensor_shape(logits, "LOGITS")
			self.print_tensor_shape(labels, "LABELS")
			accuracy = tf.metrics.accuracy(predictions=tf.argmax(logits, axis=1), labels=tf.argmax(labels, axis=1), name='accuracy', metrics_collections=metrics_collection_name)
			auc_eval = tf.metrics.auc(predictions=logits, labels=labels, name='auc', metrics_collections=metrics_collection_name)
			fn_eval  = tf.metrics.false_negatives(predictions=logits, labels=labels, name='false_negatives', metrics_collections=metrics_collection_name)
			fp_eval  = tf.metrics.false_positives(predictions=logits, labels=labels, name='false_positives', metrics_collections=metrics_collection_name)
			tn_eval  = tf.metrics.true_negatives(predictions=logits, labels=labels, name='true_negatives', metrics_collections=metrics_collection_name)
			tp_eval  = tf.metrics.true_positives(predictions=logits, labels=labels, name='true_positives', metrics_collections=metrics_collection_name)



		metrics = dict()
		metrics['accuracy']=accuracy
		metrics['auc_eval']=auc_eval
		metrics['fn_eval']=fn_eval
		metrics['fp_eval']=fp_eval
		metrics['tn_eval']=tn_eval
		metrics['tp_eval']=tp_eval

		return metrics

	def getOps(self,data_tuple=None,is_training=True,learning_rate=1e-5,decay_steps=10000, decay_rate=0.96, staircase=False,ps_device="/cpu:0",w_device="/cpu:0"):
		ops=dict()

		#input
		ops['input']=data_tuple[0]

		#encode op
		ops['encode']=self.encoder(data_tuple=data_tuple,  is_training=is_training,  ps_device=ps_device, w_device=w_device)

		#decode ops
		input_shape=data_tuple[0].get_shape().as_list()
		ops['decode']=self.decoder(data_tuple=[ops['encode']],output_shape=input_shape , is_training=is_training,  ps_device=ps_device, w_device=w_device)
				

		#loss op
		ops['loss']=self.loss(ops['decode'],data_tuple)

		#train op
		ops['train']=self.training(ops['loss'],learning_rate=learning_rate, decay_steps=decay_steps, decay_rate=decay_rate, staircase=staircase)
	
		# #metrics ops
		# ops['metrics']=dict()
		# ops['metrics']['train']=self.metrics(ops['prediction'],data_tuple,metrics_collection_name='training_metrics')
		# ops['metrics']['validation']=self.metrics(ops['prediction'],data_tuple,metrics_collection_name='validation_metrics')
		# ops['metrics']['test']=self.metrics(ops['prediction'],data_tuple,metrics_collection_name='test_metrics')

		#summary ops
		ops['summary']=dict()

		#Training summary op
		smry=[]
		smry.append(tf.summary.scalar('train_loss', ops['loss']))
		ops['summary']['train']=tf.summary.merge(smry,collections=None,name='train_summary')

		#validation summary op
		smry=[]
		smry.append(tf.summary.scalar('validation_loss', ops['loss']))
		ops['summary']['validation']=tf.summary.merge(smry,collections=None,name='validation_summary')

		#Training summary op
		smry=[]
		smry.append(tf.summary.scalar('test_loss', ops['loss']))
		ops['summary']['test']=tf.summary.merge(smry,collections=None,name='test_summary')

		

		return ops
			
