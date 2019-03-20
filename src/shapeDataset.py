import os,sys

import numpy as np
from imblearn.over_sampling import SMOTE

from shapeDataExtractor import ShapeDataExtractor
from utilities import Utilities



class ShapeDataset():
	def __init__(self):
		#CSV File if data need to be extracted
		self.data_description=None
		#point features to extract
		self.target_points_feature = None
		#cell features to extract
		self.target_cells_feature = None

		#JSON File # if data is already extracted
		self.tfrecord_info_path=None
		self.tfrecord_info=None			#contain dict from self.tfrecord_info_path (JSON File)
		self.tfrecord_description=None	#containe dict from self.tfrecord_info['csv_description'] (CSV File)

		#Output dir
		self.output_dir=None

		#dataset description
		self.dataset_description=None

		self.train_set=None
		self.train_dir=None
		
			

		self.val_set=None
		self.val_dir=None
		self.val_size=0.1

		self.test_set=None
		self.test_dir=None
		self.test_size=0.1

		#parameters for 'classification' dataset type
		self.min_num_shapes_per_class=8
		self.use_SMOTE=True


		self.util=Utilities()

	#between 0 and 1
	def setTestSize(self, i):
		self.test_size=i

	#between 0 and 1
	def setValidationSize(self, i):
		self.val_size=i

	def setDataDescription(self,path):
		self.data_description=path

	def setPointFeature(self,arr):
		self.target_points_feature=arr

	def setCellFeature(self,arr):
		self.target_cells_feature=arr

	def setOutputPointFeature(self,arr):
		self.output_target_points_feature=arr

	def setOutputCellFeature(self,arr):
		self.output_target_cells_feature=arr

	def setTFRecordInfo(self,path):
		self.tfrecord_info_path=path

	def setOutputDirectory(self,path):
		self.output_dir=path

	def setMinNumberOfShapesPerClass(self,n):
		self.min_num_shapes_per_class=n

	def setSMOTE(self,bool):
		self.use_SMOTE=bool





	#take a dict containing for each key a array of array
	#return one array of array with an additional column representing the group
	#example:
	#input:
	#dict = {group1:[[path1,path2],
	#				 [path3,path4]
	#				],
	#
	#		 group2:[[path5,path6],
	#				 [path7,path8]
	#				],				
	#		} 
	#
	#
	#return:
	#array=[[path1,path2,group1],
	#		[path3,path4,group1],
	#		[path5,path6,group2],
	#		[path7,path8,group2],
	#	   ]

	def mergeDictGroup(self,dict):
		result=None
		for group , array in dict.items():
			a=array
			for row in a:
				row.append(group)
			if not result:
				result=a
			else:
				result.extend(a)

		return result

	#Data generation with smote only the generated data is returned
	def generateWithSMOTE(self,dataset,labels):
		sm=SMOTE(kind='regular')
		dataset_res, labels_res = sm.fit_sample(dataset,labels)

		dataset_res=dataset_res[dataset.shape[0]:,:]
		labels_res=labels_res[labels.shape[0]:]

		return dataset_res,labels_res

	def generateTFRecordsWithSMOTE(self):
		data_keys = self.tfrecord_info['data_keys']

		input_data=[]
		label_data=[]

		print('Reading data from TFRecords')
		for row in self.train_set:
			tfr_path=row[0]
			record_data = self.util.extractFloatArraysFromTFRecord(tfr_path,self.tfrecord_info)

			if 'points_feature' in data_keys  and 'cells_feature' in data_keys:
				points_and_cells=record_data['points_feature']
				points_and_cells.extend(record_data['cells_feature'])
				input_data.append(points_and_cells)
			elif 'cells_feature'in data_keys :
				input_data.append(record_data['cells_feature'])
			elif 'points_feature'in data_keys :
				input_data.append(record_data['points_feature'])
			else:
				raise Exception('Unexpected error:\
								\nNo data found in the tfrecords')

			label_data.append(record_data['output'])

		input_data=np.array(input_data)
		label_data=np.array(label_data)

		print('Generating new data')
		gen_input,gen_label=self.generateWithSMOTE(input_data,label_data.reshape(-1))
		gen_label=gen_label.reshape((len(gen_label),-1))

		smote_tfrecords_path=os.path.join(self.output_dir,'SMOTE_tfrecords')
		try:
			os.mkdir(smote_tfrecords_path)
		except FileExistsError:
			pass

		print('Saving new data')
		smote_set=[['TFRecords','VTK Files','Group']]
		for i in range(gen_input.shape[0]):
			features_to_write=dict()
			
			if 'points_feature' in data_keys  and 'cells_feature' in data_keys:
				length_points=len(record_data['points_feature'])
				length_cells=len(record_data['cells_feature'])
				features_to_write['points_feature']=gen_input[i,:length_points].tolist()
				features_to_write['cells_feature']=gen_input[i,length_points:length_cells].tolist()
			elif 'cells_feature'in data_keys :

				features_to_write['cells_feature']=gen_input[i,:].tolist()
			elif 'points_feature'in data_keys :
				features_to_write['points_feature']=gen_input[i,:].tolist()

			features_to_write['output']=gen_label[i,:].tolist()


			smote_tfr_path=os.path.join(smote_tfrecords_path,'SMOTE_TFRecord_'+str(i)+'.tfrecord')
			self.util.writeTFRecord(features_to_write,smote_tfr_path)

			row = [smote_tfr_path,'SMOTE GENERATED',self.tfrecord_info['class_corres_digit_to_name'][str(gen_label[i][0])]]
			
			smote_set.append(row)

		#count the number of generated records for each group
		
		groups,num_smote = np.unique(np.array(smote_set[1:])[:,2],return_counts=True)
		for i in range(len(groups)):
			self.dataset_description['dataset_class_count'][str(groups[i])]['num_smote']=int(num_smote[i])
		
		smote_tfrecord_description_path=os.path.join(smote_tfrecords_path,'smote_tfrecord_description.csv')
		self.util.writeCSVFile(smote_set,smote_tfrecord_description_path)

		self.train_set.extend(smote_set[1:])

		return smote_tfrecord_description_path



	def createDataset(self,autocomplete_feature_name=True):
		try:
			os.mkdir(self.output_dir)
		except FileExistsError:
			pass

		if not self.tfrecord_info_path and self.data_description: #we need to extract the data

			data_extractor=ShapeDataExtractor()
			data_extractor.setCSVDescription(self.data_description)
			data_extractor.setOutputDirectory(os.path.join(self.output_dir,'tfrecords'))
			data_extractor.setPointFeature(self.target_points_feature)
			data_extractor.setCellFeature(self.target_cells_feature)
			data_extractor.setOutPointFeature(self.output_target_points_feature)
			data_extractor.setOutCellFeature(self.output_target_cells_feature)


			self.tfrecord_info_path=data_extractor.saveTFRecords(autocomplete_feature_name=autocomplete_feature_name)

		if self.tfrecord_info_path:
			print('Starting dataset generation')
			print('Using feature extraction info located at :\
				   \n'+self.tfrecord_info_path)

			self.tfrecord_info=self.util.readJSONFile(self.tfrecord_info_path)
			self.tfrecord_description=self.util.readDictCSVFile(self.tfrecord_info['csv_description'])

			if self.tfrecord_info['dataset_type']=='classification':

				dataset_description_path=self.createClassificationDataset()
				print('Dataset informations: %s' %(dataset_description_path))

				print('Dataset generated')

				return dataset_description_path

			if self.tfrecord_info['dataset_type']=='gan':

				dataset_description_path=self.createGANDataset()
				print('Dataset informations: %s' %(dataset_description_path))

				print('Dataset generated')

				return dataset_description_path

			if self.tfrecord_info['dataset_type']=='autoencoder':

				dataset_description_path=self.createAutoencoderDataset()
				print('Dataset informations: %s' %(dataset_description_path))

				print('Dataset generated')

				return dataset_description_path

		else:
			raise Exception('No data information found\
							\nplease provide a CSV file describing the data or a JSON file containing the tfrecords informations.')


	def createGANDataset(self):

		print('')
		print('A generative adversarial network dataset will be generated')
		self.dataset_description=dict()
		self.dataset_description['dataset_type']='gan'
		self.dataset_description['tfrecord_info']=self.tfrecord_info_path
		self.dataset_description['train_description_path']=self.tfrecord_info['csv_description']
		self.dataset_description['shape_number']=len(self.tfrecord_description['TFRecords'])

		dataset_description_path=os.path.join(self.output_dir,'dataset_info.json')
		self.util.writeJSONFile(self.dataset_description,dataset_description_path)

		return dataset_description_path

	def createClassificationDataset(self):

		self.dataset_description=dict()
		self.dataset_description['dataset_type']='classification'
		self.dataset_description['SMOTE']=self.use_SMOTE
		self.dataset_description['tfrecord_info']=self.tfrecord_info_path
		print('')
		print('A classification dataset will be generated')
		print('The minimum number of shapes per class is set to %d' %(self.min_num_shapes_per_class))
		if self.use_SMOTE:
			print('Some training data will be generated using SMOTE')

		for group , value in self.tfrecord_info['class_count'].items():
			if value<self.min_num_shapes_per_class:
				raise Exception("The group '%s' have only %d shapes, a minimum of %d is required" %(group,value,self.min_num_shapes_per_class))


		#split the set according to their group
		group_description=dict()
		for i in range(len(self.tfrecord_description['Group'])):
			if self.tfrecord_description['Group'][i] not in group_description.keys():
				group_description[self.tfrecord_description['Group'][i]]=[]

			row=[self.tfrecord_description['TFRecords'][i],self.tfrecord_description['VTK Files'][i]]
			group_description[self.tfrecord_description['Group'][i]].append(row)


		#randomize the files and split each group in a train set, a validation set and a test set
		self.dataset_description['dataset_class_count']=dict()
		train_set_dict=dict()
		val_set_dict=dict()
		test_set_dict=dict()

		num_valid = int(self.val_size*self.tfrecord_info['example_number']/len(self.tfrecord_info['class_names']))
		valid_set_exist=True
		if(num_valid == 0):
			print('WARNING: No samples in the validation set, try increasing the validation ratio')
			valid_set_exist=False

		num_test  = int(self.test_size*self.tfrecord_info['example_number']/len(self.tfrecord_info['class_names']))
		test_set_exist=True
		if(num_test == 0):
			print('WARNING: No samples in the test set, try increasing the test ratio')
			test_set_exist=False

		train_set_exist=False
		for group ,array in group_description.items():
			self.dataset_description['dataset_class_count'][group]=dict()

			#randomize
			array=np.array(array)
			permutation=np.random.permutation(len(array))
			array=array[permutation]

			#split
			

			num_train = len(array)-num_valid-num_test


			if (num_train < 6) and self.use_SMOTE:
				raise Exception("Not enougth samples in the training set to generate data with SMOTE \
								\nTo generate data with SMOTE, 6 samples per group are needed but only %d are available in the group '%s'\
								\nTry to decrease the test and/or validation ratio\
								\n\
								\nSamples per group in the validation set: %d\
								\nValidation ratio: %.3f\
								\n\
								\nSamples per group in the test set: %d\
								\nTest ratio: %.3f\
								" %(num_train,group,num_valid,self.val_size,num_test,self.test_size))

			if (num_train != 0):
				train_set_exist = True


			train_set_dict[group]=array[:num_train].tolist()
			val_set_dict[group]=array[num_train:num_train+num_valid].tolist()
			test_set_dict[group]=array[num_train+num_valid:].tolist()

			num_test=len(test_set_dict[group])

			self.dataset_description['dataset_class_count'][group]['num_train']=num_train
			self.dataset_description['dataset_class_count'][group]['num_validation']=num_valid
			self.dataset_description['dataset_class_count'][group]['num_test']=num_test

		


		#remerge train val and test dict in 1 array of array ([[tfr_path1,vtk_path1,group1],
		#add a column to each row describing the group 					    ...
		#													   [tfr_pathN,vtk_pathN,groupN]])
		#
		self.train_set=self.mergeDictGroup(train_set_dict)
		self.val_set=self.mergeDictGroup(val_set_dict)
		self.test_set=self.mergeDictGroup(test_set_dict)


		if train_set_exist:
			#If setted use SMOTE to generate training data
			if self.use_SMOTE:
				self.dataset_description['smote_tfrecord_description_path']=self.generateTFRecordsWithSMOTE()
				
			#create soft links for each set and save the csv description
			print('Generating soft links')
			#TRAIN
			self.train_dir=os.path.join(self.output_dir,'train')
			try:
			    os.mkdir(self.train_dir)
			except FileExistsError:
			    pass

			for i in range(len(self.train_set)):
				new_record_path=os.path.join(self.train_dir,'train_'+str(i)+'.tfrecord')
				self.util.force_symlink(self.train_set[i][0],new_record_path)
				self.train_set[i][0]=new_record_path

			train_csv=[['TFRecords','VTK Files','Group']]
			train_csv.extend(self.train_set)
			train_csv_path=os.path.join(self.train_dir,'train_description.csv')
			self.util.writeCSVFile(train_csv,train_csv_path)

			self.dataset_description['train_description_path']=train_csv_path

		#VALIDATION
		if valid_set_exist:
			self.val_dir=os.path.join(self.output_dir,'validation')
			try:
			    os.mkdir(self.val_dir)
			except FileExistsError:
			    pass

			for i in range(len(self.val_set)):
				new_record_path=os.path.join(self.val_dir,'validation_'+str(i)+'.tfrecord')
				self.util.force_symlink(self.val_set[i][0],new_record_path)
				self.val_set[i][0]=new_record_path

			val_csv=[['TFRecords','VTK Files','Group']]
			val_csv.extend(self.val_set)
			val_csv_path=os.path.join(self.val_dir,'validation_description.csv')
			self.util.writeCSVFile(val_csv,val_csv_path)

			self.dataset_description['validation_description_path']=val_csv_path

		#TEST
		if test_set_exist:
			self.test_dir=os.path.join(self.output_dir,'test')
			try:
			    os.mkdir(self.test_dir)
			except FileExistsError:
			    pass

			for i in range(len(self.test_set)):
				new_record_path=os.path.join(self.test_dir,'test_'+str(i)+'.tfrecord')
				self.util.force_symlink(self.test_set[i][0],new_record_path)
				self.test_set[i][0]=new_record_path

			test_csv=[['TFRecords','VTK Files','Group']]
			test_csv.extend(self.test_set)
			test_csv_path=os.path.join(self.test_dir,'test_description.csv')
			self.util.writeCSVFile(test_csv,test_csv_path)

			self.dataset_description['test_description_path']=test_csv_path

		self.dataset_description['set_count']=dict()
		self.dataset_description['set_count']['num_train']=0
		self.dataset_description['set_count']['num_validation']=0
		self.dataset_description['set_count']['num_test']=0

		for key,value in self.dataset_description['dataset_class_count'].items():
			try:
				self.dataset_description['set_count']['num_train']+=value['num_train']
			except:
				pass
			try:
				self.dataset_description['set_count']['num_train']+=value['num_smote']
			except:
				pass
			try:
				self.dataset_description['set_count']['num_validation']+=value['num_validation']
			except:
				pass
			try:
				self.dataset_description['set_count']['num_test']+=value['num_test']
			except:
				pass
			
	

		dataset_description_path=os.path.join(self.output_dir,'dataset_info.json')
		self.util.writeJSONFile(self.dataset_description,dataset_description_path)




		return dataset_description_path


	def createAutoencoderDataset(self):
		self.dataset_description=dict()
		self.dataset_description['dataset_type']='autoencoder'
		self.dataset_description['tfrecord_info']=self.tfrecord_info_path
		print('')
		print('A autoencoder dataset will be generated')

		#split the set according to their group
		description_array=list()
		for i in range(len(self.tfrecord_description['TFRecords'])):
			row=[self.tfrecord_description['TFRecords'][i],self.tfrecord_description['Input VTK Files'][i],self.tfrecord_description['Output VTK Files'][i]]
			description_array.append(row)


		#randomize the files and split into a train set, a validation set and a test set
		num_valid = int(self.val_size*len(description_array))
		valid_set_exist=True
		if(num_valid == 0):
			print('WARNING: No samples in the validation set, try increasing the validation ratio')
			valid_set_exist=False

		num_test  = int(self.test_size*len(description_array))
		test_set_exist=True
		if(num_test == 0):
			print('WARNING: No samples in the test set, try increasing the test ratio')
			test_set_exist=False

		train_set_exist=False
		num_train = len(description_array)-num_valid-num_test
		if (num_train != 0):
			train_set_exist = True
		


		#randomize
		array=np.array(description_array)
		permutation=np.random.permutation(len(array))
		array=array[permutation]

		#split
		self.train_set=array[:num_train].tolist()
		self.val_set=array[num_train:num_train+num_valid].tolist()
		self.test_set=array[num_train+num_valid:].tolist()


		if train_set_exist:
			#create soft links for each set and save the csv description
			print('Generating soft links')
			#TRAIN
			self.train_dir=os.path.join(self.output_dir,'train')
			try:
			    os.mkdir(self.train_dir)
			except FileExistsError:
			    pass

			for i in range(len(self.train_set)):
				new_record_path=os.path.join(self.train_dir,'train_'+str(i)+'.tfrecord')
				self.util.force_symlink(self.train_set[i][0],new_record_path)
				self.train_set[i][0]=new_record_path

			train_csv=[['TFRecords','Input VTK Files','Output VTK Files']]
			train_csv.extend(self.train_set)
			train_csv_path=os.path.join(self.train_dir,'train_description.csv')
			self.util.writeCSVFile(train_csv,train_csv_path)

			self.dataset_description['train_description_path']=train_csv_path

		#VALIDATION
		if valid_set_exist:
			self.val_dir=os.path.join(self.output_dir,'validation')
			try:
			    os.mkdir(self.val_dir)
			except FileExistsError:
			    pass

			for i in range(len(self.val_set)):
				new_record_path=os.path.join(self.val_dir,'validation_'+str(i)+'.tfrecord')
				self.util.force_symlink(self.val_set[i][0],new_record_path)
				self.val_set[i][0]=new_record_path

			val_csv=[['TFRecords','Input VTK Files','Output VTK Files']]
			val_csv.extend(self.val_set)
			val_csv_path=os.path.join(self.val_dir,'validation_description.csv')
			self.util.writeCSVFile(val_csv,val_csv_path)

			self.dataset_description['validation_description_path']=val_csv_path

		#TEST
		if test_set_exist:
			self.test_dir=os.path.join(self.output_dir,'test')
			try:
			    os.mkdir(self.test_dir)
			except FileExistsError:
			    pass

			for i in range(len(self.test_set)):
				new_record_path=os.path.join(self.test_dir,'test_'+str(i)+'.tfrecord')
				self.util.force_symlink(self.test_set[i][0],new_record_path)
				self.test_set[i][0]=new_record_path

			test_csv=[['TFRecords','Input VTK Files','Output VTK Files']]
			test_csv.extend(self.test_set)
			test_csv_path=os.path.join(self.test_dir,'test_description.csv')
			self.util.writeCSVFile(test_csv,test_csv_path)

			self.dataset_description['test_description_path']=test_csv_path

		self.dataset_description['set_count']=dict()
		self.dataset_description['set_count']['num_train']=len(self.train_set)
		self.dataset_description['set_count']['num_validation']=len(self.val_set)
		self.dataset_description['set_count']['num_test']=len(self.test_set)

		dataset_description_path=os.path.join(self.output_dir,'dataset_info.json')
		self.util.writeJSONFile(self.dataset_description,dataset_description_path)




		return dataset_description_path