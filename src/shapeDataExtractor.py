import os,sys,vtk
import utilities
import numpy as np 
import tensorflow as tf 


class ShapeDataExtractor:
	def __init__(self):
		#dataset type
		self.dataset_type = None
		#input csv file 
		self.csv_description_path = None
		self.csv_description = None
		#output directory
		self.output_dir = None
		#point features to extract
		self.target_points_feature = None
		#cell features to extract
		self.target_cells_feature = None

		#Extraction info
		self.extraction_info=None

		#tfrecord info
		self.tfrecord_info = None

		#for classification datasets
		self.min_num_shapes_per_class=5



		#utility functions
		self.util=utilities.Utilities()

	#parametrisation functions
	def setCSVDescription(self,path):
		self.csv_description_path=path
		self.csv_description=self.util.readDictCSVFile(path)

	def setOutputDirectory(self,path):
		self.output_dir=path

	def setPointFeature(self,arr):
		self.target_points_feature=arr

	def setCellFeature(self,arr):
		self.target_cells_feature=arr


	def setOutPointFeature(self,arr):
		self.out_target_points_feature=arr

	def setOutCellFeature(self,arr):
		self.out_target_cells_feature=arr



	#auto detect the type of dataset thanks to the columns name of the csv file
	def detectDatasetType(self):
              
		if 'VTK Files' in self.csv_description.keys() and 'Group' in self.csv_description.keys():
		    self.dataset_type='classification'

		elif 'VTK Files' in self.csv_description.keys():
			self.dataset_type='gan'

		elif 'Input VTK Files' in self.csv_description.keys() and 'Output VTK Files' in self.csv_description.keys():
		    self.dataset_type='autoencoder'

		else:
			raise Exception("Impossible to determine the dataset type!\
							 \nFor classification type, the csv file should contain columns named 'VTK Files' and 'Group'.\
							 \nFor generation type, the csv file should contain columns named 'input VTK File' and 'output VTK File'.")
		self.tfrecord_info['dataset_type']=self.dataset_type
		return


	#polydata feature extraction functions

	def extractFeaturesForAll(self,vtk_files,autocomplete_feature_name=True):
		if not self.target_points_feature and not self.target_cells_feature:
			raise Exception('No features to extract')

		self.extraction_info=dict()
		self.extraction_info['dataset_type']=self.dataset_type
		self.extraction_info['points_feature']=None
		self.extraction_info['cells_feature']=None

		if self.target_points_feature:
			self.extraction_info['points_feature']=dict()
			self.extraction_info['points_feature']['feature_names']=None

		if self.target_cells_feature:
			self.extraction_info['cells_feature']=dict()
			self.extraction_info['cells_feature']['feature_names']=None


		all_data=dict()
		for file in vtk_files:
			data = self.extractFeatures(file,autocomplete_feature_name=autocomplete_feature_name)
			for key, val in data.items():
				if key not in all_data.keys():
					all_data[key]=[]
				all_data[key].append(val)


		for key, value in all_data.items():
			all_data[key]=np.array(all_data[key])

		return all_data

	def extractFeatures(self,vtk_file,autocomplete_feature_name=True):
		#check existance
		if not os.path.isfile(vtk_file):
			raise Exception('File not found: '+vtk_file)

		#read file 
		#print('Reading: '+vtk_file)
		reader = vtk.vtkPolyDataReader()
		reader.SetFileName(vtk_file)
		reader.Update()

		polydata = reader.GetOutput()

		data=dict()

		if self.target_points_feature:
			point_data = self.extractFeaturePoints(polydata,autocomplete_feature_name=autocomplete_feature_name)
			point_data = point_data.reshape(-1)

			data['points_feature']=point_data

		if self.target_cells_feature:
			cell_data = self.extractFeatureCells(polydata,autocomplete_feature_name=autocomplete_feature_name)
			cell_data = cell_data.reshape(-1)

			data['cells_feature']=cell_data


		return data 

	def extractFeaturePoints(self,polydata,autocomplete_feature_name=True):

		if self.extraction_info['points_feature']['feature_names'] is None:

			extractable_features_points_names=self.getExtractablePointsFeatureNames(polydata,autocomplete_feature_name=autocomplete_feature_name)
			
			if len(extractable_features_points_names)==0:
				raise Exception('Unable to find any of the point features \
								\nif no point features are needed, please set feature_cells to None')
			self.extraction_info['points_feature']['feature_names']=extractable_features_points_names

		#generate a data array containing an array for each point of the polydata
		data = []
		for i in range(polydata.GetNumberOfPoints()):
			data.append([])
		#look all the desired point feature and try to extract them
		for feature in self.extraction_info['points_feature']['feature_names']:

			if feature == "Points": # Extract the points coordinates
				points_buffer=list()
				for i in range(polydata.GetNumberOfPoints()):
					ptn = polydata.GetPoint(i)
					data[i].extend(ptn)
					if feature not in self.extraction_info['points_feature'].keys():
						self.extraction_info['points_feature'][feature]=dict()
						self.extraction_info['points_feature'][feature]['size']=len(ptn)
			else:
				array=polydata.GetPointData().GetScalars(feature)
				if array and array.GetName() == feature:
					feature_buffer=list()
					for i in range(0, array.GetNumberOfTuples()):
						ft = array.GetTuple(i)
						data[i].extend(ft)
						if feature not in self.extraction_info['points_feature'].keys():
							self.extraction_info['points_feature'][feature]=dict()
							self.extraction_info['points_feature'][feature]['size']=len(ft)
				else:
					raise Exception("'Impossible to extract the '"+feature+"' point feature")

		data = np.array(data)

		if 'shape' not in self.extraction_info['points_feature'].keys():
			self.extraction_info['points_feature']['shape']=data.shape
			self.extraction_info['points_feature']['size']=data.size
		elif data.shape != self.extraction_info['points_feature']['shape']:
			raise Exception('Feature point data shapes are not matching\
							 \n got '+data.shape+' but '+self.extraction_info['points_feature']['shape']+' was expected')


		return data

	def extractFeatureCells(self,polydata,autocomplete_feature_name=True):

		if self.extraction_info['cells_feature']['feature_names'] is None:

			extractable_features_cells_names=self.getExtractableCellsFeatureNames(polydata,autocomplete_feature_name=autocomplete_feature_name)

			if len(extractable_features_cells_names)==0:
				raise Exception('Unable to find any of the cell features \
								 \nif no cell features are needed, please set feature_cells to None')

			self.extraction_info['cells_feature']['feature_names']=extractable_features_cells_names

		#generate a data array containing an array for each point of the polydata
		data = []
		for i in range(polydata.GetNumberOfCells()):
			data.append([])
		#look all the desired point feature and try to extract them
		for feature in self.extraction_info['cells_feature']['feature_names']:


			array=polydata.GetCellData().GetScalars(feature)
			if array and array.GetName() == feature:
				feature_buffer=list()
				for i in range(0, array.GetNumberOfTuples()):
					ft = array.GetTuple(i)
					data[i].extend(ft)
					if feature not in self.extraction_info['cells_feature'].keys():
						self.extraction_info['cells_feature'][feature]=dict()
						self.extraction_info['cells_feature'][feature]['size']=len(ft)
			else:
				raise Exception("'Impossible to extract the '"+feature+"' cell feature")

		data = np.array(data)

		if 'shape' not in self.extraction_info['cells_feature'].keys():
			self.extraction_info['cells_feature']['shape']=data.shape
			self.extraction_info['cells_feature']['size']=data.size
		elif data.shape != self.extraction_info['cells_feature']['shape']:
			raise Exception('Feature cell data shapes are not matching\
							 \n got '+data.shape+' but '+self.extraction_info['cells_feature']['shape']+' was expected')


		return data

	def getExtractablePointsFeatureNames(self,polydata,autocomplete_feature_name=True):
		extracted_features_points_names=list() # contain the name of all the features that have been extracted
		available_features_points = self.getPointsArrayNames(polydata) # contain the available features in the shape

		for feature in self.target_points_feature:
			if feature == "Points": 
				extracted_features_points_names.append(feature)
			elif autocomplete_feature_name:
				real_features_names = [name for name in available_features_points if feature in name]
				if len(real_features_names)==0:
					raise Exception("Unable to find the point feature '"+feature+"' with autocompletion = "+str(autocomplete_feature_name))
				extracted_features_points_names.extend(real_features_names)
			elif feature in available_features_points:
				extracted_features_points_names.append(feature)
			else:
				raise Exception("Unable to find the point feature '"+feature+"' with autocompletion = "+str(autocomplete_feature_name))

		return extracted_features_points_names

	def getExtractableCellsFeatureNames(self,polydata,autocomplete_feature_name=True):
		extracted_features_cells_names=list() # contain the name of all the features that have been extracted
		available_features_cells = self.getCellsArrayNames(polydata) # contain the available features in the shape

		for feature in self.target_cells_feature:
			if autocomplete_feature_name:
				real_features_names = [name for name in available_features_cells if feature in name]
				if len(real_features_names)==0:
					raise Exception("Unable to find the cell feature '"+feature+"' with autocompletion = "+str(autocomplete_feature_name))
				extracted_features_cells_names.extend(real_features_names)
			elif feature in available_features_cells:
				extracted_features_cells_names.append(feature)
			else:
				raise Exception("Unable to find the cell feature '"+feature+"' with autocompletion = "+str(autocomplete_feature_name))

		return extracted_features_cells_names

	def getPointsArrayNames(self,geometry):
		arraynames = []
		pointdata = geometry.GetPointData()
		for i in range(pointdata.GetNumberOfArrays()):
		    arraynames.append(pointdata.GetArrayName(i))
		return arraynames

	def getCellsArrayNames(self,geometry):
		arraynames = []
		celldata = geometry.GetCellData()
		for i in range(celldata.GetNumberOfArrays()):
		    arraynames.append(celldata.GetArrayName(i))
		return arraynames
	

	#main functions
	def saveTFRecords(self,autocomplete_feature_name=True):
		if self.csv_description_path == None:
			raise Exception('No input CSV file have been defined')
		if self.output_dir == None:
			raise Exception('No output dir have been defined')

		self.tfrecord_info=dict()
		self.detectDatasetType()
		

		if self.dataset_type == 'classification':
			return self.saveClassificationTFRecords(autocomplete_feature_name=autocomplete_feature_name)
		elif self.dataset_type == 'gan':
			return self.saveGANTFRecords(autocomplete_feature_name=autocomplete_feature_name)
		elif self.dataset_type == 'autoencoder':
			return self.saveAutoencoderTFRecords(autocomplete_feature_name=autocomplete_feature_name)

		else :
			raise Exception("Unexpected error, unknown dataset type '"+self.dataset_type+"'")

	def saveClassificationTFRecords(self,autocomplete_feature_name=True):
		print('Starting data extraction for a classification dataset')
		print('')
		
		#get the class names and bind them to a number

		class_names = []
		for name in self.csv_description['Group']:
			if name not in class_names:
				class_names.append(name)

		print('%d group detected\n'%(len(class_names)))

		class_corres_name_to_digit=dict()
		class_corres_digit_to_name=dict()
		for label, name in enumerate(class_names):
			label=label
			class_corres_name_to_digit[name]=label
			class_corres_digit_to_name[label]=name

		self.tfrecord_info['class_names']=class_names
		self.tfrecord_info['class_corres_name_to_digit']=class_corres_name_to_digit
		self.tfrecord_info['class_corres_digit_to_name']=class_corres_digit_to_name
		self.tfrecord_info['class_number']=len(class_names)

		class_count=dict()
		total_example=0
		for name in class_names:
			class_count[name]=self.csv_description['Group'].count(name)
			total_example += class_count[name]

		self.tfrecord_info['class_count']=class_count
		self.tfrecord_info['example_number']=total_example

		#convert labels from the csv file in digits
		label_data = []
		for label in self.csv_description['Group']:
			label = class_corres_name_to_digit[label]
			label_data.append([label])
		label_data=np.array(label_data)

		

		if self.target_points_feature:
			print('Points feature wanted:\
				  \n\t-%s \n' %('\n\t-'.join(self.target_points_feature)))
		else:
			print('No points feature to extract\n')

		if self.target_cells_feature:
			print('Cells feature wanted:\
				  \n\t-%s \n' %('\n\t-'.join(self.target_cells_feature)))
		else:
			print('No cells feature to extract\n')

		if autocomplete_feature_name:
			print('Feature names autocompletion is on\n')
		else:
			print('Feature names autocompletion is off\n')

		print('Reading files and extracting features ...')

		#get the features


		feature_data=self.extractFeaturesForAll(self.csv_description['VTK Files'],autocomplete_feature_name=autocomplete_feature_name)
		self.tfrecord_info['extraction_info']=self.extraction_info

		print('Extraction succed!')
		if self.tfrecord_info['extraction_info']['points_feature']:
			print('Points feature extracted: \
				   \n\t-%s \n' %('\n\t-'.join(self.tfrecord_info['extraction_info']['points_feature']['feature_names'])))
		else:
			print('No points feature extracted\n')
		if self.tfrecord_info['extraction_info']['cells_feature']:
			print('Cells feature extracted: \
				   \n\t-%s \n' %('\n\t-'.join(self.tfrecord_info['extraction_info']['cells_feature']['feature_names'])))
		else:
			print('No cells feature extracted\n')


		
		#write the records
		print('Saving TFRecords')
		try:
			os.mkdir(self.output_dir)
		except FileExistsError:
			pass


		tfrecords=[]
		for i in range(label_data.shape[0]):
			features_to_write=dict()
			
			if 'points_feature' in feature_data.keys():
				features_to_write['points_feature']=feature_data['points_feature'][i,:].tolist()
			if 'cells_feature' in feature_data.keys():
				features_to_write['cells_feature']=feature_data['cells_feature'][i,:].tolist()
			features_to_write['output']=label_data[i,:].tolist()

			tfrecord_path=os.path.join(self.output_dir,'TFRecord_'+str(i)+'.tfrecord')
			self.util.writeTFRecord(features_to_write,tfrecord_path)
			tfrecords.append(tfrecord_path)

		data_keys=[]
		if 'points_feature' in feature_data.keys():
			data_keys.append('points_feature')
		if 'cells_feature' in feature_data.keys():
			data_keys.append('cells_feature')
		data_keys.append('output')


		self.tfrecord_info['data_keys']=data_keys
		
		for key,value in features_to_write.items():
			self.tfrecord_info[key]=dict()
			self.tfrecord_info[key]['size']=np.array(value).size
			self.tfrecord_info[key]['type']=str(np.array(value).dtype)
			if key != 'output':
				self.tfrecord_info[key]['shape']=self.tfrecord_info['extraction_info'][key]['shape']
			elif key =='output':
				self.tfrecord_info[key]['shape']=[1]
			else:
				raise Exception('Unexpected error: unknown feature %s'%(key))

		self.tfrecord_info['original_csv_description']=self.csv_description_path

		new_csv_description=self.csv_description
		new_csv_description['TFRecords']=tfrecords

		new_csv_description_path=os.path.join(self.output_dir,'tfrecord_description.csv')
		self.util.writeDictCSVFile(new_csv_description,new_csv_description_path)
		


		self.tfrecord_info['csv_description']=new_csv_description_path
		tfrecord_info_path=os.path.join(self.output_dir,'tfrecord_info.json')
		self.util.writeJSONFile(self.tfrecord_info,tfrecord_info_path)

		print('Saving done\n')
		#report
		
		print('TFRecord description: '+new_csv_description_path)
		print('TFRecord informations: '+tfrecord_info_path)
		print('')
		return tfrecord_info_path

	def saveGANTFRecords(self,autocomplete_feature_name=True):
		print('Starting data extraction for a generative dataset')
		print('')
		try:
			os.mkdir(self.output_dir)
		except FileExistsError:
			pass

		if self.target_points_feature:
			print('Points feature wanted:\
				  \n\t-%s \n' %('\n\t-'.join(self.target_points_feature)))
		else:
			print('No points feature to extract\n')

		if self.target_cells_feature:
			print('Cells feature wanted:\
				  \n\t-%s \n' %('\n\t-'.join(self.target_cells_feature)))
		else:
			print('No cells feature to extract\n')

		if autocomplete_feature_name:
			print('Feature names autocompletion is on\n')
		else:
			print('Feature names autocompletion is off\n')

		print('Reading files and extracting features ...')

		feature_data=self.extractFeaturesForAll(self.csv_description['VTK Files'],autocomplete_feature_name=autocomplete_feature_name)
		self.tfrecord_info['extraction_info']=self.extraction_info

		print('Extraction succed!')
		if self.tfrecord_info['extraction_info']['points_feature']:
			print('Points feature extracted: \
				   \n\t-%s \n' %('\n\t-'.join(self.tfrecord_info['extraction_info']['points_feature']['feature_names'])))
		else:
			print('No points feature extracted\n')
		if self.tfrecord_info['extraction_info']['cells_feature']:
			print('Cells feature extracted: \
				   \n\t-%s \n' %('\n\t-'.join(self.tfrecord_info['extraction_info']['cells_feature']['feature_names'])))
		else:
			print('No cells feature extracted\n')


		
		#write the records
		print('Saving TFRecords')
		tfrecords=[]
		for i in range(len(self.csv_description['VTK Files'])):
			features_to_write=dict()
			
			if 'points_feature' in feature_data.keys():
				features_to_write['points_feature']=feature_data['points_feature'][i,:].tolist()
			if 'cells_feature' in feature_data.keys():
				features_to_write['cells_feature']=feature_data['cells_feature'][i,:].tolist()


			tfrecord_path=os.path.join(self.output_dir,'TFRecord_'+str(i)+'.tfrecord')
			self.util.writeTFRecord(features_to_write,tfrecord_path)
			tfrecords.append(tfrecord_path)

		new_csv_description=self.csv_description
		new_csv_description['TFRecords']=tfrecords

		new_csv_description_path=os.path.join(self.output_dir,'tfrecord_description.csv')
		self.util.writeDictCSVFile(new_csv_description,new_csv_description_path)


		

		data_keys=[]
		if 'points_feature' in feature_data.keys():
			data_keys.append('points_feature')
		if 'cells_feature' in feature_data.keys():
			data_keys.append('cells_feature')

		self.tfrecord_info['data_keys']=data_keys

		for key,value in features_to_write.items():
			self.tfrecord_info[key]=dict()
			self.tfrecord_info[key]['size']=np.array(value).size
			self.tfrecord_info[key]['type']=str(np.array(value).dtype)
			if key != 'output':
				self.tfrecord_info[key]['shape']=self.tfrecord_info['extraction_info'][key]['shape']
			else:
				self.tfrecord_info[key]['shape']=[1]


		self.tfrecord_info['original_csv_description']=self.csv_description_path
		self.tfrecord_info['csv_description']=new_csv_description_path
		tfrecord_info_path=os.path.join(self.output_dir,'tfrecord_info.json')
		self.util.writeJSONFile(self.tfrecord_info,tfrecord_info_path)

		print('Saving done\n')
		#report
		
		print('TFRecord description: '+new_csv_description_path)
		print('TFRecord informations: '+tfrecord_info_path)
		print('')

		return tfrecord_info_path

	def saveAutoencoderTFRecords(self,autocomplete_feature_name=True):
		print('Starting data extraction for a autoencoder dataset')
		print('')
		try:
			os.mkdir(self.output_dir)
		except FileExistsError:
			pass




		print('Extracting input files features ')

		if self.target_points_feature:
			print('Points feature wanted:\
				  \n\t-%s \n' %('\n\t-'.join(self.target_points_feature)))
		else:
			print('No points feature to extract\n')

		if self.target_cells_feature:
			print('Cells feature wanted:\
				  \n\t-%s \n' %('\n\t-'.join(self.target_cells_feature)))
		else:
			print('No cells feature to extract\n')

		if autocomplete_feature_name:
			print('Feature names autocompletion is on\n')
		else:
			print('Feature names autocompletion is off\n')

		print('Reading files and extracting features ...')

		input_feature_data=self.extractFeaturesForAll(self.csv_description['Input VTK Files'],autocomplete_feature_name=autocomplete_feature_name)
		self.tfrecord_info['input_extraction_info']=self.extraction_info

		print('Extraction succed!')
		if self.tfrecord_info['input_extraction_info']['points_feature']:
			print('Points feature extracted: \
				   \n\t-%s \n' %('\n\t-'.join(self.tfrecord_info['input_extraction_info']['points_feature']['feature_names'])))
		else:
			print('No points feature extracted\n')
		if self.tfrecord_info['input_extraction_info']['cells_feature']:
			print('Cells feature extracted: \
				   \n\t-%s \n' %('\n\t-'.join(self.tfrecord_info['input_extraction_info']['cells_feature']['feature_names'])))
		else:
			print('No cells feature extracted\n')



		print('\n\n###############################################\n\n')
		print('Extracting output files features ')

		self.setPointFeature(self.out_target_points_feature)
		self.setCellFeature(self.out_target_cells_feature)

		if self.target_points_feature:
			print('Points feature wanted:\
				  \n\t-%s \n' %('\n\t-'.join(self.target_points_feature)))
		else:
			print('No points feature to extract\n')

		if self.target_cells_feature:
			print('Cells feature wanted:\
				  \n\t-%s \n' %('\n\t-'.join(self.target_cells_feature)))
		else:
			print('No cells feature to extract\n')

		if autocomplete_feature_name:
			print('Feature names autocompletion is on\n')
		else:
			print('Feature names autocompletion is off\n')

		print('Reading files and extracting features ...')

		output_feature_data=self.extractFeaturesForAll(self.csv_description['Output VTK Files'],autocomplete_feature_name=autocomplete_feature_name)
		self.tfrecord_info['output_extraction_info']=self.extraction_info

		print('Extraction succed!')
		if self.tfrecord_info['output_extraction_info']['points_feature']:
			print('Points feature extracted: \
				   \n\t-%s \n' %('\n\t-'.join(self.tfrecord_info['output_extraction_info']['points_feature']['feature_names'])))
		else:
			print('No points feature extracted\n')
		if self.tfrecord_info['output_extraction_info']['cells_feature']:
			print('Cells feature extracted: \
				   \n\t-%s \n' %('\n\t-'.join(self.tfrecord_info['output_extraction_info']['cells_feature']['feature_names'])))
		else:
			print('No cells feature extracted\n')

		
		#write the records
		print('Saving TFRecords')
		input_feature_data
		output_feature_data
		tfrecords=[]
		for i in range(len(self.csv_description['Input VTK Files'])):
			features_to_write=dict()
			
			if 'points_feature' in input_feature_data.keys():
				features_to_write['input_points_feature']=input_feature_data['points_feature'][i,:].tolist()
			if 'cells_feature' in input_feature_data.keys():
				features_to_write['input_cells_feature']=input_feature_data['cells_feature'][i,:].tolist()


			if 'points_feature' in output_feature_data.keys():
				features_to_write['output_points_feature']=output_feature_data['points_feature'][i,:].tolist()
			if 'cells_feature' in output_feature_data.keys():
				features_to_write['output_cells_feature']=output_feature_data['cells_feature'][i,:].tolist()


			tfrecord_path=os.path.join(self.output_dir,'TFRecord_'+str(i)+'.tfrecord')
			self.util.writeTFRecord(features_to_write,tfrecord_path)
			tfrecords.append(tfrecord_path)

		new_csv_description=self.csv_description
		new_csv_description['TFRecords']=tfrecords

		new_csv_description_path=os.path.join(self.output_dir,'tfrecord_description.csv')
		self.util.writeDictCSVFile(new_csv_description,new_csv_description_path)


		

		data_keys=[]

		for key,value in features_to_write.items():
			data_keys.append(key)
			self.tfrecord_info[key]=dict()
			self.tfrecord_info[key]['size']=np.array(value).size
			self.tfrecord_info[key]['type']=str(np.array(value).dtype)
			if 'input_' in key:
				self.tfrecord_info[key]['shape']=self.tfrecord_info['input_extraction_info'][key.split('input_')[1]]['shape']
			elif 'output_' in key:
				self.tfrecord_info[key]['shape']=self.tfrecord_info['output_extraction_info'][key.split('output_')[1]]['shape']
			elif key == 'output':
				self.tfrecord_info[key]['shape']=[1]

		self.tfrecord_info['data_keys']=data_keys


		self.tfrecord_info['original_csv_description']=self.csv_description_path
		self.tfrecord_info['csv_description']=new_csv_description_path
		tfrecord_info_path=os.path.join(self.output_dir,'tfrecord_info.json')
		self.util.writeJSONFile(self.tfrecord_info,tfrecord_info_path)

		print('Saving done\n')
		#report
		
		print('TFRecord description: '+new_csv_description_path)
		print('TFRecord informations: '+tfrecord_info_path)
		print('')

		return tfrecord_info_path


	def extractAndSave(self):
		try:
			os.mkdir(self.output_dir)
		except FileExistsError:
			pass

		feature_data=self.extractFeaturesForAll(self.csv_description['VTK Files'],autocomplete_feature_name=False)
		self.tfrecord_info['extraction_info']=self.extraction_info
		tfrecords=[]
		for i in range(len(self.csv_description['VTK Files'])):
			features_to_write=dict()
			
			if 'points_feature' in feature_data.keys():
				features_to_write['points_feature']=feature_data['points_feature'][i,:].tolist()
			if 'cells_feature' in feature_data.keys():
				features_to_write['cells_feature']=feature_data['cells_feature'][i,:].tolist()


			tfrecord_path=os.path.join(self.output_dir,'TFRecord_'+str(i)+'.tfrecord')
			self.util.writeTFRecord(features_to_write,tfrecord_path)
			tfrecords.append(tfrecord_path)

		new_csv_description=self.csv_description
		new_csv_description['TFRecords']=tfrecords

		new_csv_description_path=os.path.join(self.output_dir,'tfrecord_description.csv')
		self.util.writeDictCSVFile(new_csv_description,new_csv_description_path)


		

		data_keys=[]
		if 'points_feature' in feature_data.keys():
			data_keys.append('points_feature')
		if 'cells_feature' in feature_data.keys():
			data_keys.append('cells_feature')

		self.tfrecord_info['data_keys']=data_keys

		for key,value in features_to_write.items():
			self.tfrecord_info[key]=dict()
			self.tfrecord_info[key]['size']=np.array(value).size
			self.tfrecord_info[key]['type']=str(np.array(value).dtype)
			if key != 'output':
				self.tfrecord_info[key]['shape']=self.tfrecord_info['extraction_info'][key]['shape']
			else:
				self.tfrecord_info[key]['shape']=[1]


		self.tfrecord_info['original_csv_description']=self.csv_description_path
		self.tfrecord_info['csv_description']=new_csv_description_path
		tfrecord_info_path=os.path.join(self.output_dir,'tfrecord_info.json')
		self.util.writeJSONFile(self.tfrecord_info,tfrecord_info_path)

		return tfrecord_info_path


