import os,sys

import tensorflow as tf 
import numpy as np

from base_nn import BaseNN 


class ClassificationNN(BaseNN):
	def __init__(self,class_number=5):


		self.network_param = None


		self.network_info = None

	def getTrainingParameters(self):
		param = dict()
		param[self.keep_prob]=0.5
		return param

	def getValidationParameters(self):
		param = dict()
		param[self.keep_prob]=1
		return param

	def getEvaluationParameters(self):
		return self.getValidationParameters()

	def inference(self, data_tuple=None, is_training=False, ps_device="/cpu:0", w_device="/gpu:0"):
		if data_tuple:
			inputs=data_tuple[0]

		else:
			raise NoDataError

		with tf.variable_scope('network'):

			input_shape=inputs.get_shape().as_list()
			
			inputs = tf.reshape(tensor=inputs,shape=(-1,np.prod(input_shape[1:])))
			inputs = tf.layers.batch_normalization(inputs, training=is_training, name="batch_normalisation_1")
			self.print_tensor_shape(inputs, "inputs")

			matmul_1 = self.matmul(inputs  , 4096, name='matmul_1_op', activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
			matmul_2 = self.matmul(matmul_1, 2048, name='matmul_2_op', activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
			matmul_3 = self.matmul(matmul_2, 1024, name='matmul_3_op', activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
			matmul_4 = self.matmul(matmul_3, 512 , name='matmul_4_op', activation=tf.nn.relu, ps_device=ps_device, w_device=w_device)
			self.keep_prob=tf.placeholder(tf.float32)
			drop_op  = tf.nn.dropout( matmul_4, self.keep_prob)
			matmul_5 = self.matmul(drop_op, len(self.tfrecord_info['class_names']) , name='matmul_5_op', activation=None, ps_device=ps_device, w_device=w_device)

		return matmul_5

	def loss(self, logits, data_tuple, class_weights=None):
		labels=data_tuple[-1]
		self.print_tensor_shape(labels, "LABEL")
		num_label=len(self.tfrecord_info['class_names'])

		with tf.variable_scope('loss'):
			labels=tf.one_hot(tf.cast(labels,tf.int32),num_label)[:,0,:]

			loss=tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)
			self.print_tensor_shape(loss, "loss")

		return loss

	def training(self, loss, learning_rate=1e-3, decay_steps=10000, decay_rate=0.96, staircase=False):
		with tf.variable_scope('optimisation'):
			global_step = tf.Variable(0, name='global_step', trainable=False)
			optimizer = tf.train.AdamOptimizer(learning_rate)
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

		#prediction ops
		ops['prediction']=self.inference(data_tuple=data_tuple,  is_training=is_training,  ps_device=ps_device, w_device=w_device)
		with tf.variable_scope('softmax_prediction'):
			ops['softmax_prediction']=tf.nn.softmax(ops['prediction'])
		with tf.variable_scope('class_prediction'):
			ops['class_prediction']=tf.argmax(ops['softmax_prediction'],axis=1)

		if is_training:
			with tf.variable_scope('true_one_hot'):
				num_label=len(self.tfrecord_info['class_names'])
				ops['true_one_hot']=tf.one_hot(tf.cast(data_tuple[-1],tf.int32),num_label)[:,0,:]
			with tf.variable_scope('true_class'):
				ops['true_class']=data_tuple[-1][:,0]		

			#loss op
			ops['loss']=self.loss(ops['prediction'],data_tuple)

			#train op
		
			ops['train']=self.training(ops['loss'],learning_rate=learning_rate, decay_steps=decay_steps, decay_rate=decay_rate, staircase=staircase)
		
			#metrics ops
			ops['metrics']=dict()
			ops['metrics']['train']=self.metrics(ops['prediction'],data_tuple,metrics_collection_name='training_metrics')
			ops['metrics']['validation']=self.metrics(ops['prediction'],data_tuple,metrics_collection_name='validation_metrics')
			ops['metrics']['test']=self.metrics(ops['prediction'],data_tuple,metrics_collection_name='test_metrics')

			#summary ops
			ops['summary']=dict()

			#Training summary op
			smry=[]
			smry.append(tf.summary.scalar('train_loss', ops['loss']))
			for key,val in ops['metrics']['train'].items():
				if len(val)==1:
					smry.append(tf.summary.scalar('train_'+key, val[0]))
				else:
					for i in range(len(val)):
						smry.append(tf.summary.scalar('train_'+key+'_'+str(i),val[i]))

			ops['summary']['train']=tf.summary.merge(smry,collections=None,name='train_summary')

			#validation summary op
			smry=[]
			smry.append(tf.summary.scalar('validation_loss', ops['loss']))
			for key,val in ops['metrics']['validation'].items():
				if len(val)==1:
					smry.append(tf.summary.scalar('validation_'+key, val[0]))
				else:
					for i in range(len(val)):
						smry.append(tf.summary.scalar('validation_'+key+'_'+str(i),val[i]))

			ops['summary']['validation']=tf.summary.merge(smry,collections=None,name='validation_summary')

		return ops
			


