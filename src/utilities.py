import os,sys,csv,json
import tensorflow as tf
import numpy as np
import vtk




class Utilities():
	def __init__(self):
		pass


	#CSV Files management

	#Read a csv file,
	#return a dictionary where the keys are the first line of the file 
	#and the values are lists containing the columns
	def readDictCSVFile(self,csv_path):
		try:
			with open(csv_path, mode='r') as csv_file:
				csv_reader = csv.DictReader(csv_file)

				csv_dict=dict()
				for row in csv_reader:

					for key,value in row.items():
						if key not in csv_dict:
							csv_dict[key]=[value]
						else:
							csv_dict[key].append(value)

		except Exception as e:
			raise Exception('Unable to read CSV file from'+ csv_path+ ':\n'+ str(e))
		

		return csv_dict

	#write a csv file from a dictionary,
	def writeDictCSVFile(self,dict,output_csv):
		rows = []

		keys = []
		for key in dict.keys():
			keys.append(key)
		rows.append(keys)


		for i in range(len(dict[keys[0]])):
			row=[]
			for key in rows[0]:
				row.append(dict[key][i])
			rows.append(row)

		return self.writeCSVFile(rows,output_csv)

	#write a csv file from a list of rows
	def writeCSVFile(self,rows,path):
		try:
			os.mkdir(os.path.dirname(path))
		except FileExistsError:
			pass

		try:
			with open(path, mode='w') as csvoutput:
			    writer = csv.writer(csvoutput, lineterminator='\n')
			    writer.writerows(rows)
		except Exception as e:
			raise Exception('Unable to save CSV file to'+ path+ ':\n'+ str(e))

		return path



	#JSON Files managment
	def readJSONFile(self,json_path):
		try:
			with open(json_path, "r") as f:
				dict = json.load(f)
		except Exception as e:
			raise Exception('Unable to read JSON file '+ json_path+ ':'+ str(e))

		return dict

	def writeJSONFile(self,dict,json_path,indent = 4):
		try:
			os.mkdir(os.path.dirname(json_path))
		except FileExistsError:
			pass

		try:
			with open(json_path, 'w') as f:
			    json.dump(dict, f,indent = 4)
		except Exception as e:
			raise Exception('Unable to save JSON file to'+ json_path+ ':'+ str(e))

		return json_path



	#TFRecord writting functions
	#input: a dict containing numpy arrays
	def writeTFRecord(self,feature_data,path):
		#convert the dict's values into tensorflow feature
		converted_features=dict()
		for feature_name , value in feature_data.items():
			if np.array(value).dtype == 'float64':
				converted_features[feature_name]=self._float_feature(value)
			elif np.array(value).dtype == 'int64':
				converted_features[feature_name]=self._int64_feature(value)
			else :
				converted_features[feature_name]=self._bytes_feature(value)

		#Create a tensorflow global feature containing all the features
		features_data = tf.train.Features(feature=converted_features)

		#create the tensorflow example
		features_example=tf.train.Example(features=features_data)
		
		#write the example in a tfRecord
		with tf.python_io.TFRecordWriter(path) as writer:
			writer.write(features_example.SerializeToString())
		#print('Saved record: '+path)
		return path

	def _int64_feature(self,value):
		if not isinstance(value, list):
			value = [value]
		return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

	def _float_feature(self,value):
		if not isinstance(value, list):
			value = [value]
		return tf.train.Feature(float_list=tf.train.FloatList(value=value))

	def _bytes_feature(self,value):
		if not isinstance(value, list):
			value = [value]
		return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

	#TFRecord reading function (outside any tensorflow graph) 
	#return a dict where each data_key is associated with his data
	def extractFloatArraysFromTFRecord(self,tfr_path,tfrecord_info):
		record_iterator = tf.python_io.tf_record_iterator(path=tfr_path)

		data=dict()

		for string_record in record_iterator:

			example = tf.train.Example()
			example.ParseFromString(string_record)

			for key in tfrecord_info['data_keys']:
				if 'float' in tfrecord_info[key]['type']:
					arr=example.features.feature[key].float_list.value
					for i in range(len(arr)):
						arr[i]=float(arr[i])
					data[key]=arr
				elif 'int' in tfrecord_info[key]['type']:
					arr=example.features.feature[key].int64_list.value
					for i in range(len(arr)):
						arr[i]=int(arr[i])
					data[key]=arr
				else:
					arr=example.features.feature[key].bytes_list.value
					for i in range(len(arr)):
						arr[i]=str(arr[i])
					data[key]=arr

			return data


	#symbolic link function
	def force_symlink(self,file1, file2):
	    try:
	        os.symlink(file1, file2)
	    except FileExistsError:
	        os.remove(file2)
	        os.symlink(file1, file2)


	#VTK Polydata Utilities
	def deleteAllPolydataFeatures(self,geometry):
		#point data
		arraynames = []
		pointdata = geometry.GetPointData()
		for i in range(pointdata.GetNumberOfArrays()):
		    arraynames.append(pointdata.GetArrayName(i))
		for name in arraynames:
			pointdata.RemoveArray(name) 	

		#cell data
		arraynames = []
		celldata = geometry.GetCellData()
		for i in range(celldata.GetNumberOfArrays()):
		    arraynames.append(celldata.GetArrayName(i))
		for name in arraynames:
			celldata.RemoveArray(name) 	

		geometry.Modified()
		return geometry

	def generateVTKFloatFromNumpy(self,npArray,name):
		array_shape=npArray.shape
		nb_points=array_shape[0]
		nb_components=array_shape[1]


		array=vtk.vtkFloatArray()
		array.SetName(name)
		array.SetNumberOfComponents(nb_components)
		for i in range(nb_points):
			array.InsertNextTuple(npArray[i,:])

		return array
					
	def generateVTKPointsFromNumpy(self,npArray):
		npArray=npArray.flatten()
		num_points = int(npArray.shape[0]/3)

		vtk_points = vtk.vtkPoints()
		for i in range(num_points):
		    vtk_points.InsertNextPoint(npArray[3*i],npArray[3*i+1],npArray[3*i+2])
		return vtk_points