import os,sys

import tensorflow as tf 
import numpy as np

from collections import OrderedDict

from base_nn import BaseNN 

Adam = tf.keras.optimizers.Adam
_graph_replace = tf.contrib.graph_editor.graph_replace

class GanNN(BaseNN):
	def __init__(self,class_number=5):


		self.network_param = None


		self.network_info = None

	# def getTrainingParameters(self):
	# 	param = dict()
	# 	param[self.keep_prob]=0.5
	# 	return param

	# def getValidationParameters(self):
	# 	param = dict()
	# 	param[self.keep_prob]=1
	# 	return param

	# def getEvaluationParameters(self):
	# 	return self.getValidationParameters()


	def generator(self, data_tuple=None, is_training=False,regularization_constant=0.0, ps_device="/cpu:0", w_device="/gpu:0"):
		if data_tuple:
			data=data_tuple[0]

		else:
			raise NoDataError

		output_shape=data.get_shape().as_list()
		
		inputs = tf.random_normal([43, 64], mean=0, stddev=1)

		matmul_1 = self.matmul(inputs  , 256  , name='matmul_1_op', activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)
		drop_op_1  = tf.nn.dropout( matmul_1, 0.5)
		matmul_2 = self.matmul(drop_op_1, 512 , name='matmul_2_op', activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)
		drop_op_2  = tf.nn.dropout( matmul_2, 0.5)
		matmul_3 = self.matmul(drop_op_2, 1024 , name='matmul_3_op', activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)
		drop_op_3  = tf.nn.dropout( matmul_3, 0.5)
		matmul_4 = self.matmul(drop_op_3, 2048 , name='matmul_4_op', activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)
		drop_op_4  = tf.nn.dropout( matmul_4, 0.5)
		#matmul_5 = self.matmul(matmul_4, 4096 , name='matmul_5_op', activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)
		#matmul_6 = self.matmul(matmul_5, 8192 , name='matmul_6_op', activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)
		matmul_7 = self.matmul(drop_op_4, np.prod(output_shape[1:]) , name='matmul_7_op', activation=None, ps_device=ps_device, w_device=w_device)

		# with tf.name_scope('Generator_regularization'):
		# 	Reg_constant = tf.constant(regularization_constant)
		# 	reg_op = tf.nn.l2_loss(matmul_1) + tf.nn.l2_loss(matmul_2)  + tf.nn.l2_loss(matmul_3) + tf.nn.l2_loss(matmul_4) + tf.nn.l2_loss(matmul_7)
		# 	reg_op = reg_op*Reg_constant

		return matmul_7 , 0#reg_op

	def discriminator(self, data_tuple=None, input_reg_op=None,is_training=False,regularization_constant=0.0, ps_device="/cpu:0", w_device="/gpu:0"):
		if data_tuple != None:
			data=data_tuple[0]

		else:
			raise NoDataError

		data_shape=data.get_shape().as_list()
		print(data_shape)
		
		inputs = tf.reshape(tensor=data,shape=(-1,np.prod(data_shape[1:])))


		#matmul_1 = self.matmul(inputs  , 8192  , name='matmul_1_op', activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)
		#matmul_2 = self.matmul(inputs  , 4096 , name='matmul_2_op', activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)
		matmul_3 = self.matmul(inputs, 2048 , name='matmul_3_op', activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)
		drop_op_3  = tf.nn.dropout( matmul_3, 0.5)
		matmul_4 = self.matmul(drop_op_3, 1024 , name='matmul_4_op', activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)
		drop_op_4  = tf.nn.dropout( matmul_4, 0.5)
		matmul_5 = self.matmul(drop_op_4, 512 , name='matmul_5_op', activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)
		drop_op_5  = tf.nn.dropout( matmul_5, 0.5)
		matmul_6 = self.matmul(drop_op_5, 256 , name='matmul_6_op', activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)
		drop_op_6  = tf.nn.dropout( matmul_6, 0.5)
		matmul_7 = self.matmul(drop_op_6, 1 , name='matmul_7_op', activation=None, ps_device=ps_device, w_device=w_device)

		with tf.name_scope('Discriminator_regularization'):
			Reg_constant = tf.constant(regularization_constant)
			reg_op = tf.nn.l2_loss(matmul_3) + tf.nn.l2_loss(matmul_4)  + tf.nn.l2_loss(matmul_5) + tf.nn.l2_loss(matmul_6) + tf.nn.l2_loss(matmul_7)
			if input_reg_op is not None:
				reg_op=reg_op+input_reg_op
			reg_op = reg_op*Reg_constant

		return matmul_7#+reg_op

	def loss(self, d_real, d_false, class_weights=None):
		with tf.variable_scope('d_loss'):
			d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real))
			d_loss_real = tf.reduce_mean(d_loss_real)

			d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_false, labels=tf.zeros_like(d_false))
			d_loss_fake = tf.reduce_mean(d_loss_fake)

			d_loss=d_loss_fake+d_loss_real

		with tf.variable_scope('g_loss'):
			g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_false, labels=tf.ones_like(d_false)))
			#g_loss = -tf.reduce_mean(tf.log(d_false+1e-08))

		return g_loss,d_loss

	def training(self, g_loss,d_loss, learning_rate=1e-3, decay_steps=10000, decay_rate=0.96, staircase=False):
		vars_train = tf.trainable_variables()

		vars_gen = [var for var in vars_train if 'generator' in var.name]        
		vars_dis = [var for var in vars_train if 'discriminator' in var.name] 

		with tf.variable_scope('g_optimisation'):
			global_step = tf.Variable(0, name='global_step', trainable=False)
			optimizer = tf.train.AdamOptimizer(learning_rate)
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				g_train = optimizer.minimize(g_loss, global_step=global_step, var_list=vars_gen)

		with tf.variable_scope('d_optimisation'):
			global_step = tf.Variable(0, name='global_step', trainable=False)
			optimizer = tf.train.AdamOptimizer(learning_rate)
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				d_train = optimizer.minimize(d_loss, global_step=global_step, var_list=vars_dis)

		return g_train,d_train


	def metrics(self, logits, data_tuple, metrics_collection_name='collection_metrics'):
		# labels=data_tuple[-1]
		# num_label=len(self.tfrecord_info['class_names'])
		
		# with tf.variable_scope(metrics_collection_name):
		# 	logits=tf.nn.softmax(logits)
		# 	labels=tf.one_hot(tf.cast(labels,tf.int32),num_label)[:,0,:]
		# 	self.print_tensor_shape(logits, "LOGITS")
		# 	self.print_tensor_shape(labels, "LABELS")
		# 	accuracy = tf.metrics.accuracy(predictions=tf.argmax(logits, axis=1), labels=tf.argmax(labels, axis=1), name='accuracy', metrics_collections=metrics_collection_name)
		# 	auc_eval = tf.metrics.auc(predictions=logits, labels=labels, name='auc', metrics_collections=metrics_collection_name)
		# 	fn_eval  = tf.metrics.false_negatives(predictions=logits, labels=labels, name='false_negatives', metrics_collections=metrics_collection_name)
		# 	fp_eval  = tf.metrics.false_positives(predictions=logits, labels=labels, name='false_positives', metrics_collections=metrics_collection_name)
		# 	tn_eval  = tf.metrics.true_negatives(predictions=logits, labels=labels, name='true_negatives', metrics_collections=metrics_collection_name)
		# 	tp_eval  = tf.metrics.true_positives(predictions=logits, labels=labels, name='true_positives', metrics_collections=metrics_collection_name)



		# metrics = dict()
		# metrics['accuracy']=accuracy
		# metrics['auc_eval']=auc_eval
		# metrics['fn_eval']=fn_eval
		# metrics['fp_eval']=fp_eval
		# metrics['tn_eval']=tn_eval
		# metrics['tp_eval']=tp_eval

		return None

	def getOps(self,data_tuple=None,regularization_constant=0.0,is_training=True,learning_rate=1e-5,decay_steps=10000, decay_rate=0.96, staircase=False,ps_device="/cpu:0",w_device="/gpu:0"):
		ops=dict()

		#prediction ops
		with tf.variable_scope('generator'):
			ops['g'],gen_reg_op=self.generator( data_tuple=data_tuple, is_training=is_training,regularization_constant=regularization_constant, ps_device=ps_device, w_device=w_device)
		
		with tf.variable_scope('discriminator') as scope:
			ops['d_real']=self.discriminator( data_tuple=data_tuple, is_training=is_training,regularization_constant=regularization_constant, ps_device=ps_device, w_device=w_device)
			scope.reuse_variables()
			ops['d_false']=self.discriminator( data_tuple=[ops['g']],input_reg_op=gen_reg_op, is_training=is_training,regularization_constant=regularization_constant, ps_device=ps_device, w_device=w_device)

		ops['g_loss'],ops['d_loss']=self.loss(ops['d_real'],ops['d_false'])

		ops['g_train'],ops['d_train']=self.training(ops['g_loss'],ops['d_loss'],learning_rate=learning_rate, decay_steps=decay_steps, decay_rate=decay_rate, staircase=staircase)

		#unrolled training
		# ops['g_train'],ops['g_loss'],ops['d_train'],ops['d_loss']=self.unrolled_train(ops['d_real'],ops['d_false'],learning_rate=learning_rate)

		#summary ops
		ops['summary']=dict()

		#Training summary op
		smry=[]
		smry.append(tf.summary.scalar('discriminator_loss', ops['d_loss']))
		smry.append(tf.summary.scalar('generator_loss', ops['d_loss']))


		ops['summary']['train']=tf.summary.merge(smry,collections=None,name='train_summary')
		
		return ops
			



	#Unrolled network functions (from https://github.com/poolio/unrolled_gan)

	def unrolled_train(self,d_real,d_false,learning_rate=1e-5):
		d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real))
		d_loss_real = tf.reduce_mean(d_loss_real)

		d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_false, labels=tf.zeros_like(d_false))
		d_loss_fake = tf.reduce_mean(d_loss_fake)

		d_loss=d_loss_fake+d_loss_real

		g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")
		d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator")

		d_opt = Adam(lr=learning_rate)
		updates = d_opt.get_updates( d_loss,d_vars)
		d_train = tf.group(*updates, name="d_train_op")


		#UNROLLING

		update_dict = self.extract_update_dict(updates)
		cur_update_dict = update_dict
		for i in range(3):
		    cur_update_dict = self.graph_replace(update_dict,cur_update_dict)
		unrolled_loss = self.graph_replace(d_loss,cur_update_dict)

		g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
		g_train = g_opt.minimize(-unrolled_loss, var_list=g_vars)

		return g_train,unrolled_loss,d_train,d_loss

	def remove_original_op_attributes(self, graph):
	    for op in graph.get_operations():
	        op._original_op = None

	def graph_replace(self, *args, **kwargs):
	    self.remove_original_op_attributes(tf.get_default_graph())
	    return _graph_replace(*args, **kwargs)

	def extract_update_dict(self, update_ops):
		name_to_var = {v.name: v for v in tf.global_variables()}
		updates = OrderedDict()
		for update in update_ops:
		    var_name = update.op.inputs[0].name
		    var = name_to_var[var_name]
		    value = update.op.inputs[1]
		    if update.op.type == 'Assign':
		        updates[var.value()] = value
		    elif update.op.type == 'AssignAdd':
		        updates[var.value()] = var + value
		    else:
		        raise ValueError(
		            "Update op type (%s) must be of type Assign or AssignAdd" %
		            update_op.op.type)
		return updates