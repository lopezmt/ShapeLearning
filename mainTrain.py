import os, sys, json, csv, time, argparse ,vtk
import tensorflow as tf
import numpy as np

script_path = os.path.dirname(os.path.abspath( __file__ ))
source_path= os.path.join(script_path,'src')
sys.path.append(source_path)
sys.path.append(os.path.join(source_path,'networks'))
from utilities import Utilities

from test_nn import TestNN




class ShapeTrainer():
	def __init__(self):
		#dataset parameters
		self.dataset_info_path=None
		self.dataset_info=None

		self.tfrecord_info_path=None
		self.tfrecord_info=None

		self.train_set_description_path=None
		self.train_set_description=None

		self.validation_set_description_path=None
		self.validation_set_description=None

		self.test_set_description_path=None
		self.test_set_description=None

		self.dataset_type=None

		self.feature_to_use = 'point'

		self.keys_to_features = None #Used for read and decode functions


		#Network to use 
		self.model_info=None

		self.network_name=None
		self.network_type=None


		#training parameters
		self.epochs=1
		self.batch_size=32
		self.learning_rate=1e-3
		self.buffer_size=1000


		#output model and summary
		self.output_dir=None


		#utilities obj
		self.util=Utilities()


	def setDatasetInformation(self,path):
		self.dataset_info_path = path
		self.dataset_info = self.util.readJSONFile(path)

		self.tfrecord_info_path=self.dataset_info['tfrecord_info']
		self.tfrecord_info=self.util.readJSONFile(self.tfrecord_info_path)

		if 'train_description_path' in self.dataset_info.keys():
			self.train_set_description_path=self.dataset_info['train_description_path']
			self.train_set_description=self.util.readDictCSVFile(self.train_set_description_path)
		else:
			raise Exception('No training set present in this dataset\
							\nUnable to train')

		if 'validation_description_path' in self.dataset_info.keys():
			self.validation_set_description_path=self.dataset_info['validation_description_path']
			self.validation_set_description=self.util.readDictCSVFile(self.validation_set_description_path)
		else:
			print('WARNING:  No validation set found')

		if 'test_description_path' in self.dataset_info.keys():
			self.test_set_description_path=self.dataset_info['test_description_path']
			self.test_set_description=self.util.readDictCSVFile(self.test_set_description_path)
		else:
			print('WARNING:  No test set found')

		self.dataset_type=self.dataset_info['dataset_type']

	def setOutputModel(self,path):
		self.output_dir=path

	def setEpochs(self,epc):
		self.epochs=epc

	def setBatchSize(self,bs):
		self.batch_size=bs

	def setLearningRate(self,lr):
		self.learning_rate=lr

	def setBufferSize(self,bf):
		self.buffer_size=bf

	def usePointsFeature(self):
		self.feature_to_use='point'

	def useCellsFeature(self):
		self.feature_to_use='cell'


	#utility for gan training
	def saveBatchInVTKFiles(self,output_batch,step):
		print ('Saving output batch')
		gen_path=os.path.join(self.output_dir,'training_samples')
		try:
			os.mkdir(gen_path)
		except FileExistsError:
			pass
		gen_path=os.path.join(gen_path,'gan_gen_'+str(step))
		try:
			os.mkdir(gen_path)
		except FileExistsError:
			pass

		sampletemplate=self.train_set_description['VTK Files'][0]

		reader = vtk.vtkPolyDataReader()
		reader.SetFileName(sampletemplate)
		reader.Update()

		polydata = reader.GetOutput()

		polydata=self.util.deleteAllPolydataFeatures(polydata)

		for j in range(len(output_batch)):
			shape = output_batch[j]

			# if self.tfrecord_info["extraction_info"]["points_feature"] and self.tfrecord_info["extraction_info"]["cells_feature"]:
			# 	point_shape=shape[:self.tfrecord_info["extraction_info"]["points_feature"]['size']]
			# 	point_shape.reshape(self.tfrecord_info["extraction_info"]["points_feature"]['shape'])

			# 	cell_shape=shape[self.tfrecord_info["extraction_info"]["points_feature"]['size']:]
			# 	cell_shape.reshape(self.tfrecord_info["extraction_info"]["cells_feature"]['shape'])

			# elif self.tfrecord_info["extraction_info"]["points_feature"]:
			# 	point_shape=shape
			# 	point_shape.reshape(self.tfrecord_info["extraction_info"]["points_feature"]['shape'])

			# elif self.tfrecord_info["extraction_info"]["cells_feature"]:
			# 	cell_shape=shape
			# 	cell_shape.reshape(self.tfrecord_info["extraction_info"]["cells_feature"]['shape'])
			# else:
			# 	raise Exception('Unexpected error\
			# 				   \nNo Extraction information found')

			point_shape=shape
			point_shape=point_shape.reshape(self.tfrecord_info["extraction_info"]["points_feature"]['shape'])

			offset = 0

			for name in self.tfrecord_info["extraction_info"]["points_feature"]["feature_names"]:
				size =  self.tfrecord_info["extraction_info"]["points_feature"][name]["size"]

				array   = point_shape[:,offset:size]
				offset += size

				if name == 'Points':
					array = self.util.generateVTKPointsFromNumpy(shape)
					polydata.SetPoints(array)

				else:
					array = self.util.generateVTKFloatFromNumpy(array,name)
					polydata.GetPointData().AddArray(array)

			polydata.Modified()

			writer_poly = vtk.vtkPolyDataWriter()
			writer_poly.SetInputData(polydata)
			writer_poly.SetFileName(os.path.join(gen_path,'gan_gen_'+str(j)+'.vtk'))
			writer_poly.Update()

		print('Batch saved: '+ gen_path)
		print('')



	#train loops
	def classificationTraining(self):
		from classification_nn import ClassificationNN
		nn=ClassificationNN()
		nn.setTFRecordInfo(tfrecord_info=self.tfrecord_info)

		self.model_info=dict()
		self.model_info['model_type']='classification'
		self.model_info['feature_used']=self.feature_to_use
		self.model_info['dataset_info_path']=self.dataset_info_path
		self.model_structure=None
		self.model_parameters=None

		with tf.variable_scope("train_data"):
			train_dataset=nn.extractSet(self.train_set_description['TFRecords'],
										batch_size=self.batch_size, 
										num_epochs=self.epochs, 
										shuffle_buffer_size=self.buffer_size,
										variable_scope='train_set')
			train_ite = train_dataset.make_initializable_iterator()

		if self.validation_set_description_path:	
			with tf.variable_scope("validation_data"):
				val_dataset=nn.extractSet(self.validation_set_description['TFRecords'],
											batch_size=self.dataset_info['set_count']['num_validation'], 
											num_epochs=None, 
											shuffle_buffer_size=None,
											variable_scope='validation_set')
				val_ite = val_dataset.make_initializable_iterator()

		if self.test_set_description_path:
			with tf.variable_scope("test_data"):
				test_dataset=nn.extractSet(self.test_set_description['TFRecords'],
											batch_size=self.dataset_info['set_count']['num_test'], 
											num_epochs=None, 
											shuffle_buffer_size=None,
											variable_scope='test_set')
				test_ite = test_dataset.make_initializable_iterator()

		with tf.variable_scope("graph_input"):
			input_handle = tf.placeholder(tf.string, shape=[])
			ite = tf.data.Iterator.from_string_handle(input_handle, train_dataset.output_types, train_dataset.output_shapes)
			data_tuple=ite.get_next()



		ops=nn.getOps(data_tuple=data_tuple,
						learning_rate=self.learning_rate,
						is_training=True,
						# decay_steps=10000, 
						# decay_rate=0.96, 
						# staircase=False,
						ps_device="/cpu:0",
						w_device="/cpu:0")


		with tf.Session() as sess:
			#Global Variables Initialisation
			sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

			#Initialazing The Iterators
			train_handle,_ = sess.run([train_ite.string_handle(),train_ite.initializer])
			if self.validation_set_description_path:
				val_handle  ,_ = sess.run([val_ite.string_handle(),val_ite.initializer])
			if self.test_set_description_path:
				test_handle ,_ = sess.run([test_ite.string_handle(),test_ite.initializer])


			#Initializing The Summary Writer
			summary_writer = tf.summary.FileWriter(self.output_dir, sess.graph)

			#Initializing the model saver
			saver = tf.train.Saver()


			#training loop
					
			dataset_size=self.dataset_info['set_count']['num_train']
			epoch_step=int(dataset_size/self.batch_size)
			if dataset_size%batch_size != 0:
				epoch_step+=1

			initial_step = 0
			current_epoch = 1
			epochs_validation=10
			for step in range(epoch_step*self.epochs):

				feed_dict=nn.getTrainingParameters()
				feed_dict[input_handle]=train_handle
				_, loss, metrics, summary= sess.run([	ops['train'],
														ops['loss'],
														ops['metrics']['train'],
														ops['summary']['train']
														],
														feed_dict = feed_dict
														)
				summary_writer.add_summary(summary, step)
				summary_writer.flush()


				#SHOW TRAINING PROGRESS
				if (step+1)%epoch_step == 0 :
					print('Epoch %d/%d'     %(current_epoch,self.epochs))
					print('loss:          %.3f'    %(loss))
					print('accuracy:      %.3f%%'  %(metrics['accuracy'][1]*100))
					print('')
					
					#VALIDATION
					if self.validation_set_description_path and current_epoch%epochs_validation == 0:
						feed_dict=nn.getValidationParameters()
						feed_dict[input_handle]=val_handle
						val_loss, val_metrics, val_summary  = sess.run([ops['loss'],
																		ops['metrics']['validation'],
																		ops['summary']['validation'],
																		],
																		feed_dict = feed_dict
																		)

						summary_writer.add_summary(val_summary, step)
						summary_writer.flush()

						print('VALIDATION:')
						print('loss:            %.3f'    %(val_loss))
						print('accuracy:      %.3f%%'  %(val_metrics['accuracy'][1]*100))
						print('')

					current_epoch+=1

			#TEST
			if self.test_set_description_path:
				feed_dict=nn.getEvaluationParameters()
				feed_dict[input_handle]=test_handle
				test_loss , test_metrics = sess.run([ops['loss'],ops['metrics']['test']],feed_dict=feed_dict)
				print('TEST:')
				print('loss:            %.3f'    %(test_loss))
				print('accuracy:      %.3f%%'  %(test_metrics['accuracy'][1]*100))
				print('')


			#saving end results
			self.model_info['end_results']=dict()
			self.model_info['end_results']['train_set']=dict()
			self.model_info['end_results']['train_set']['loss']=float(loss)
			self.model_info['end_results']['train_set']['accuracy']=float(metrics['accuracy'][1])
			if self.validation_set_description_path:
				self.model_info['end_results']['validation_set']=dict()
				self.model_info['end_results']['validation_set']['loss']=float(val_loss)
				self.model_info['end_results']['validation_set']['accuracy']=float(val_metrics['accuracy'][1])
			if self.test_set_description_path:
				self.model_info['end_results']['test_set']=dict()
				self.model_info['end_results']['test_set']['loss']=float(test_loss)
				self.model_info['end_results']['test_set']['accuracy']=float(test_metrics['accuracy'][1])

			#save model and create model info
			print('Saving Model')
			save_path=saver.save(sess,os.path.join(self.output_dir,'final_model.ckpt'))

			self.model_info['model_path']=save_path

			model_info_path=os.path.join(self.output_dir,'model_info.json')
			self.util.writeJSONFile(self.model_info,model_info_path)

			print('Model informations saved: %s' %(model_info_path))

	def ganTraining(self):
		self.model_info=dict()
		self.model_info['model_type']='gan'
		self.model_info['feature_used']=self.feature_to_use
		self.model_info['dataset_info_path']=self.dataset_info_path
		self.model_structure=None
		self.model_parameters=None

		##GAN training
		from gan_nn import GanNN
		nn=GanNN()
		nn.setTFRecordInfo(tfrecord_info=self.tfrecord_info)
		with tf.variable_scope("true_data"):
			true_data=nn.extractSet(self.train_set_description['TFRecords'],
										batch_size=self.batch_size, 
										num_epochs=None, 
										shuffle_buffer_size=self.buffer_size,
										variable_scope='true_data')
			true_data_ite = true_data.make_initializable_iterator()

			data_tuple=true_data_ite.get_next()

		ops=nn.getOps(data_tuple=data_tuple,
					learning_rate=self.learning_rate,
					regularization_constant=0.0,
					is_training=True,
					# decay_steps=10000, 
					# decay_rate=0.96, 
					# staircase=False,
					ps_device="/cpu:0",
					w_device="/gpu:0")

		with tf.Session() as sess:
			#Global Variables Initialisation
			sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

			#Initialazing The Iterators
			sess.run([true_data_ite.initializer])

			#Initializing The Summary Writer
			summary_writer = tf.summary.FileWriter(self.output_dir, sess.graph)

			#Initializing the model saver
			saver = tf.train.Saver()

			# discriminator_pre_train_step=0
			# for j in range(discriminator_pre_train_step):
			# 	_, d_loss = sess.run([ops['d_train'], ops['d_loss']])
			# 	print('pre training discriminator\
			# 		 \nstep: %d    loss: %.3f'%(j,d_loss))

			# i = 0
			# while True:
			# 	i=i+1

			dataset_size=self.dataset_info['shape_number']
			epoch_step=int(dataset_size/self.batch_size)
			if dataset_size%batch_size != 0:
				epoch_step+=1

			initial_step = 0
			current_epoch = 1
			epochs_validation=10
			for step in range(epoch_step*self.epochs):
				_, _, d_loss, g_loss, g, d_real, d_false, summary = sess.run([ops['d_train'], 
																				ops['g_train'],
																				ops['d_loss'],
																				ops['g_loss'],
																				ops['g'],
																				ops['d_real'],
																				ops['d_false'],
																				ops['summary']['train']])


				if np.isnan(d_loss) or np.isnan(g_loss) or np.isinf(d_loss) or np.isinf(g_loss) :
					print('Error during training, invalid losses values:\
						 \nd_loss = '+str(d_loss)+'\
						 \ng_loss'+str(g_loss)+'\
						 \nd_real:',d_real,'\
						 \nd_false:',d_false)
					raise Exception('TrainingFailure')


				summary_writer.add_summary(summary, step)
				summary_writer.flush()

				if (step+1)%epoch_step == 0 :
					print('Epoch %d/%d\
						 \nGenerator loss = %.4f\
						 \nDiscriminator loss = %.4f\n'%(current_epoch,self.epochs,g_loss,d_loss))
					current_epoch+=1

				save_batch_step =500

				if step % save_batch_step == 0:
					self.saveBatchInVTKFiles(g,step)

			#saving end results
			self.model_info['end_results']=dict()
			self.model_info['end_results']['train_set']=dict()
			self.model_info['end_results']['train_set']['discriminator_loss']=float(d_loss)
			self.model_info['end_results']['train_set']['generator_loss']=float(g_loss)

			#save model and create model info
			print('Saving Model')
			save_path=saver.save(sess,os.path.join(self.output_dir,'final_model.ckpt'))

			self.model_info['model_path']=save_path

			model_info_path=os.path.join(self.output_dir,'model_info.json')
			self.util.writeJSONFile(self.model_info,model_info_path)

			print('Model informations saved: %s' %(model_info_path))

	def autoencoderTraining(self):
		from autoencoder_nn import AutoencoderNN
		nn=AutoencoderNN()
		nn.setTFRecordInfo(tfrecord_info=self.tfrecord_info)

		self.model_info=dict()
		self.model_info['model_type']='autoencoder'
		self.model_info['feature_used']=self.feature_to_use
		self.model_info['dataset_info_path']=self.dataset_info_path
		self.model_structure=None
		self.model_parameters=None

		with tf.variable_scope("train_data"):
			train_dataset=nn.extractSet(self.train_set_description['TFRecords'],
										batch_size=self.batch_size, 
										num_epochs=self.epochs, 
										shuffle_buffer_size=self.buffer_size,
										variable_scope='train_set')
			train_ite = train_dataset.make_initializable_iterator()

		if self.validation_set_description_path:	
			with tf.variable_scope("validation_data"):
				val_dataset=nn.extractSet(self.validation_set_description['TFRecords'],
											batch_size=self.dataset_info['set_count']['num_validation'], 
											num_epochs=None, 
											shuffle_buffer_size=None,
											variable_scope='validation_set')
				val_ite = val_dataset.make_initializable_iterator()

		if self.test_set_description_path:
			with tf.variable_scope("test_data"):
				test_dataset=nn.extractSet(self.test_set_description['TFRecords'],
											batch_size=self.dataset_info['set_count']['num_test'], 
											num_epochs=None, 
											shuffle_buffer_size=None,
											variable_scope='test_set')
				test_ite = test_dataset.make_initializable_iterator()

		with tf.variable_scope("graph_input"):
			input_handle = tf.placeholder(tf.string, shape=[])
			ite = tf.data.Iterator.from_string_handle(input_handle, train_dataset.output_types, train_dataset.output_shapes)
			data_tuple=ite.get_next()


		ops=nn.getOps(data_tuple=data_tuple,
						learning_rate=self.learning_rate,
						is_training=True,
						# decay_steps=10000, 
						# decay_rate=0.96, 
						# staircase=False,
						ps_device="/cpu:0",
						w_device="/gpu:0")

		with tf.Session() as sess:
			#Global Variables Initialisation
			sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

			#Initialazing The Iterators
			train_handle,_ = sess.run([train_ite.string_handle(),train_ite.initializer])
			if self.validation_set_description_path:
				val_handle  ,_ = sess.run([val_ite.string_handle(),val_ite.initializer])
			if self.test_set_description_path:
				test_handle ,_ = sess.run([test_ite.string_handle(),test_ite.initializer])


			#Initializing The Summary Writer
			summary_writer = tf.summary.FileWriter(self.output_dir, sess.graph)

			#Initializing the model saver
			saver = tf.train.Saver()

			#training loop
			dataset_size=self.dataset_info['set_count']['num_train']
			epoch_step=int(dataset_size/self.batch_size)
			if dataset_size%batch_size != 0:
				epoch_step+=1

			initial_step = 0
			current_epoch = 1
			epochs_validation=10
			for step in range(epoch_step*self.epochs):

				feed_dict=nn.getTrainingParameters()
				feed_dict[input_handle]=train_handle

				_,loss,summary,inp,enc,dec= sess.run([	ops['train'],
														ops['loss'],
														ops['summary']['train'],
														ops['input'],
														ops['encode'],
														ops['decode']
														],
														feed_dict = feed_dict
														)
				summary_writer.add_summary(summary, step)
				summary_writer.flush()

				assert not np.any(np.isnan(inp))

				if np.isnan(loss) or np.isinf(loss) :
					print(inp)
					print(dec)
					print('###########################################')
					
					print(dec-inp)
					print('###########################################')


					print('\nError during training, invalid losse value:\
						 \nloss = '+str(loss)+'\
						 \nencoding:',enc,'\
						 \ndecoding:',dec)
					raise Exception('TrainingFailure')

				

				#SHOW TRAINING PROGRESS
				sys.stdout.write('\r')
				sys.stdout.write("Epoch %d/%d [%-30s] %d%%   loss = %.5f" % (current_epoch,self.epochs,'='*(30*(step%epoch_step+1)//epoch_step),(100*(step%epoch_step+1)//epoch_step),float(loss)))
				sys.stdout.flush()
				if (step+1)%epoch_step == 0 :
					print('')
					print('')
					
					#VALIDATION
					if self.validation_set_description_path and current_epoch%epochs_validation == 0:
						feed_dict=nn.getValidationParameters()
						feed_dict[input_handle]=val_handle
						val_loss, val_summary  = sess.run([ ops['loss'],
															ops['summary']['validation'],
															],
															feed_dict = feed_dict
															)

						summary_writer.add_summary(val_summary, step)
						summary_writer.flush()

						print('VALIDATION')
						print('loss:            %.5f'    %(val_loss))
						print('')

					current_epoch+=1

			#TEST
			if self.test_set_description_path:
				feed_dict=nn.getEvaluationParameters()
				feed_dict[input_handle]=test_handle
				test_loss , reconstruction = sess.run([ops['loss'],ops['decode']],feed_dict=feed_dict)
				print('TEST:')
				print('loss:            %.5f'    %(test_loss))
				print('')

			#reconstruction
			reconstruction_path=os.path.join(self.output_dir,'reconstruction')
			try:
				os.mkdir(reconstruction_path)
			except FileExistsError:
				pass

			reconstruction_description=self.test_set_description
			reconstruction_description['Reconstruction']=list()

			sampletemplate=self.train_set_description['Output VTK Files'][0]
			reader = vtk.vtkPolyDataReader()
			reader.SetFileName(sampletemplate)
			reader.Update()
			polydata = reader.GetOutput()
			polydata=self.util.deleteAllPolydataFeatures(polydata)

			for i in range(reconstruction.shape[0]):
				shape = reconstruction[i]
				shape = np.array(shape)

				point_shape=shape
				point_shape=point_shape.reshape(self.tfrecord_info["output_extraction_info"]["points_feature"]['shape'])

				offset = 0

				for name in self.tfrecord_info["output_extraction_info"]["points_feature"]["feature_names"]:
					size =  self.tfrecord_info["output_extraction_info"]["points_feature"][name]["size"]

					array   = point_shape[:,offset:size]
					offset += size

					if name == 'Points':
						array = self.util.generateVTKPointsFromNumpy(array)
						polydata.SetPoints(array)

					else:
						array = self.util.generateVTKFloatFromNumpy(array,name)
						polydata.GetPointData().AddArray(array)

				polydata.Modified()

				shape_path=os.path.join(reconstruction_path,'reconstruction'+str(i)+'.vtk')

				writer_poly = vtk.vtkPolyDataWriter()
				writer_poly.SetInputData(polydata)
				writer_poly.SetFileName(shape_path)
				writer_poly.Update()

				reconstruction_description['Reconstruction'].append(shape_path)

			self.util.writeDictCSVFile(reconstruction_description,os.path.join(reconstruction_path,'reconstruction_description.csv'))





	#main function
	def trainModel(self):
		print('Starting training')

		graph = tf.Graph()

		with graph.as_default():
			if self.dataset_type == 'classification':
				self.classificationTraining()

			if self.dataset_type == 'gan':
				self.ganTraining()

			if self.dataset_type == 'autoencoder':
				self.autoencoderTraining()

						


	







parser = argparse.ArgumentParser(description='Neural network trainer for shapes', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', action='store', dest='dataset_info_path', help='JSON file containing the dataset information' ,type=str,required = True)
parser.add_argument('--feature_to_use', action='store', dest='feature_to_use', help='define which features should be used (point or cell)' ,type=str,default='point')
parser.add_argument('--out', dest="output_dir", help='Output directory, the model and the summaries will be saved here', default="./", type=str)
parser.add_argument('--epochs', dest="epochs", help='Number of epochs to train the model', default=10, type=int)
parser.add_argument('--batch_size', dest="batch_size", help='Batch size for training', default=32, type=int)
parser.add_argument('--learning_rate', dest="learning_rate", help='Learning rate for training', default=1e-3, type=float)
parser.add_argument('--buffer_size', dest="buffer_size", help='Buffer size for training', default=200, type=int)


if __name__=='__main__':
	args = parser.parse_args()


	dataset_info_path=args.dataset_info_path
	feature_to_use=args.feature_to_use
	output_dir=args.output_dir
	epochs=args.epochs
	batch_size=args.batch_size
	learning_rate=args.learning_rate
	buffer_size=args.buffer_size

	trainer=ShapeTrainer()
	trainer.setDatasetInformation(dataset_info_path)
	trainer.setOutputModel(output_dir)
	trainer.setEpochs(epochs)
	trainer.setBatchSize(batch_size)
	trainer.setLearningRate(learning_rate)
	trainer.setBufferSize(buffer_size)

	if feature_to_use=='point':
		trainer.usePointsFeature()
	elif feature_to_use=='cell':
		trainer.useCellsFeature()
	else:
		raise Exception("Unknown type of feature '%s', valid choices are 'point' or 'cell'")


	trainer.trainModel()