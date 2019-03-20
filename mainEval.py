import os,sys,argparse

import tensorflow as tf

script_path = os.path.dirname(os.path.abspath( __file__ ))
source_path = os.path.join(script_path,'src')
sys.path.append(source_path)
sys.path.append(os.path.join(source_path,'networks'))
from utilities import Utilities
from shapeDataExtractor import ShapeDataExtractor


class ShapeEvaluator():
	def __init__(self):

		self.input_description_path=None
		self.input_description=None


		self.model_info_path=None
		self.model_info=None

		self.dataset_info_path=None
		self.dataset_info=None

		self.tfrecord_info_path=None
		self.tfrecord_info=None


		self.output_dir=None

		self.util=Utilities()

	def setInputDescription(self,path):
		self.input_description_path=path
		self.input_description=self.util.readDictCSVFile(path)


	def setModelInformation(self,path):
		self.model_info_path=path
		self.model_info=self.util.readJSONFile(path)

		self.dataset_info_path=self.model_info['dataset_info_path']
		self.dataset_info=self.util.readJSONFile(self.dataset_info_path)

		self.tfrecord_info_path=self.dataset_info['tfrecord_info']
		self.tfrecord_info=self.util.readJSONFile(self.tfrecord_info_path)

	def setOutputDirectory(self,path):
		self.output_dir=path

	def evaluate(self):
		print('Starting evaluation')

		graph = tf.Graph()

		with graph.as_default():

			if self.model_info['model_type']=='classification':

				from classification_nn import ClassificationNN
				nn=ClassificationNN()
				nn.setTFRecordInfo(tfrecord_info=self.tfrecord_info)


				with tf.variable_scope("evaluation_data"):
					if 'VTK Files' in self.input_description:
						data_extractor=ShapeDataExtractor()
						data_extractor.setCSVDescription(self.input_description_path)
						if self.tfrecord_info['extraction_info']['points_feature']:
							data_extractor.setPointFeature(self.tfrecord_info['extraction_info']['points_feature']['feature_names'])
						if self.tfrecord_info['extraction_info']['cells_feature']:
							data_extractor.setCellFeature(self.tfrecord_info['extraction_info']['cells_feature']['feature_names'])

						data_extractor.setOutputDirectory(os.path.join(self.output_dir,'tfrecords'))

						tfrecord_info_path=data_extractor.extractAndSave()
						nn.setTFRecordInfo(tfrecord_info_path=tfrecord_info_path)

						

					dataset=nn.extractSet(self.input_description['TFRecords'],
											batch_size=len(self.input_description['TFRecords']), 
											num_epochs=None, 
											shuffle_buffer_size=None,
											variable_scope='evaluation_set')
					ite = dataset.make_initializable_iterator()
					data_tuple=ite.get_next()


				nn.setTFRecordInfo(tfrecord_info=self.tfrecord_info)
				ops=nn.getOps(data_tuple=data_tuple,
							# images=None, 
							is_training=False,
							#learning_rate=self.learning_rate,
							# decay_steps=10000, 
							# decay_rate=0.96, 
							# staircase=False,
							ps_device="/cpu:0",
							w_device="/cpu:0")

				with tf.Session() as sess:
					#Global Variables Initialisation
					sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

					#Initialazing The Iterators
					sess.run([ite.initializer])

					#Initializing the model saver
					saver = tf.train.Saver()
					saver.restore(sess,self.model_info['model_path'])

					#eval
					feed_dict=nn.getEvaluationParameters()
					predictions = sess.run(ops['class_prediction'],feed_dict=feed_dict)


		#convert digit into original class name

		for i in range(len(predictions)):
			predictions[i]=self.tfrecord_info['class_corres_digit_to_name'][str(predictions[i])]

		new_description=self.input_description
		new_description['Predictions']=predictions

		prediction_path=os.path.join(self.output_dir,'prediction_description.csv')
		self.util.writeDictCSVFile(new_description,prediction_path)

		print('Prediction description saved: %s'%(prediction_path))

		return prediction_path





parser = argparse.ArgumentParser(description='Neural network evaluator for shapes', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', action='store', dest='model_info_path', help='JSON file containing the model information' ,type=str,required = True)
parser.add_argument('--input', action='store', dest='input_description_path', help='CSV file containing the input data location' ,type=str)
parser.add_argument('--out', dest="output_dir", help='Output directory', default="./", type=str)



if __name__=='__main__':
	args = parser.parse_args()


	model_info_path=args.model_info_path
	input_description_path=args.input_description_path
	output_dir=args.output_dir


	evaluator=ShapeEvaluator()
	evaluator.setInputDescription(input_description_path)
	evaluator.setModelInformation(model_info_path)
	evaluator.setOutputDirectory(output_dir)

	evaluator.evaluate()


