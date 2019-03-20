import os,sys,vtk
import numpy as np

script_path = os.path.dirname(os.path.abspath( __file__ ))
sys.path.append(os.path.join(script_path,'../src'))

from utilities import Utilities



rescale_path ='/work/lpzmateo/data/DL_shapes/shapes/no_features_no_groups/rescaled'
csv_path = '/work/lpzmateo/data/DL_shapes/shapes/no_features_no_groups/dataset_description.csv'



util=Utilities()



csv_dict = util.readDictCSVFile(csv_path)
print('getting params')
min_coord = None
max_coord = None
for path in csv_dict['VTK Files']:
	reader = vtk.vtkPolyDataReader()
	reader.SetFileName(path)
	reader.Update()
	polydata = reader.GetOutput()

	for i in range(polydata.GetNumberOfPoints()):
		ptn = polydata.GetPoint(i)
		for n in ptn:
			if (min_coord == None):
				min_coord=n			
			elif ( min_coord > n):
				min_coord = n


			if (max_coord == None):
				max_coord=n
			elif ( max_coord < n):
				max_coord = n

print(min_coord,'  ',max_coord)
min_coord=min_coord - 0.1*min_coord
max_coord=max_coord + 0.1*max_coord
print(min_coord,'  ',max_coord)


print('rescaling and writing')
new_csv =dict()
new_csv['VTK Files']=[]

for path in csv_dict['VTK Files']:
	print('Reading ',path)
	reader = vtk.vtkPolyDataReader()
	reader.SetFileName(path)
	reader.Update()
	polydata = reader.GetOutput()


	#get data put it into a numpy array
	ptn=[]
	for i in range(polydata.GetNumberOfPoints()):
		ptn.extend(polydata.GetPoint(i))
	ptn=np.array(ptn)
	#rescale
	ptn = (ptn - min_coord) / (max_coord - min_coord)
	#convert to VTKPoints array
	ptn = util.generateVTKPointsFromNumpy(ptn)
	#set points
	polydata.SetPoints(ptn)
	polydata.Modified()

	base_name=os.path.basename(path)
	shape_path=os.path.join(rescale_path,base_name)

	print('writing ',shape_path)
	writer_poly = vtk.vtkPolyDataWriter()
	writer_poly.SetInputData(polydata)
	writer_poly.SetFileName(shape_path)
	writer_poly.Update()


	new_csv['VTK Files'].append(shape_path)

output_csv = os.path.join(rescale_path,'dataset_description.csv')
util.writeDictCSVFile(new_csv,output_csv)
print(output_csv)




csv_dict = util.readDictCSVFile(output_csv)
print('getting params')
min_coord = None
max_coord = None
for path in csv_dict['VTK Files']:
	reader = vtk.vtkPolyDataReader()
	reader.SetFileName(path)
	reader.Update()
	polydata = reader.GetOutput()

	for i in range(polydata.GetNumberOfPoints()):
		ptn = polydata.GetPoint(i)
		for n in ptn:
			if (min_coord == None):
				min_coord=n			
			elif ( min_coord > n):
				min_coord = n


			if (max_coord == None):
				max_coord=n
			elif ( max_coord < n):
				max_coord = n

print(min_coord,'  ',max_coord)


