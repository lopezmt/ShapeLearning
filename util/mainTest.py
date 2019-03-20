import os,sys,vtk
import numpy as np

script_path = os.path.dirname(os.path.abspath( __file__ ))
sys.path.append(os.path.join(script_path,'src'))

from utilities import Utilities

				
def generateVTKPointsFromNumpy(npArray):
	num_points = int(npArray.shape[0]/3)

	vtk_points = vtk.vtkPoints()
	for i in range(num_points):
	    vtk_points.InsertNextPoint(npArray[3*i],npArray[3*i+1],npArray[3*i+2])
	return vtk_points


if __name__ =='__main__':

	util = Utilities()



	descript = "/work/lpzmateo/data/DL_shapes/datasets/GAN_condyles/tfrecord_description.csv"

	info = "/work/lpzmateo/data/DL_shapes/datasets/GAN_condyles/tfrecord_info.json"

	csv_dict=util.readDictCSVFile(descript)

	json_dict=util.readJSONFile(info)

	for i in range(len(csv_dict['TFRecords'])):


		data= util.extractFloatArraysFromTFRecord(csv_dict['TFRecords'][i],json_dict)

		template='/work/lpzmateo/data/DL_shapes/shapes/no_features/G00_Control_65condyles/34551376_Left_aligned.vtk'

		reader = vtk.vtkPolyDataReader()
		reader.SetFileName(template)
		reader.Update()

		polydata = reader.GetOutput()

		g=np.array(data['points_feature'])
		vtk_points=generateVTKPointsFromNumpy(g)
		polydata.SetPoints(vtk_points)

		polydata.Modified()

		writer_poly = vtk.vtkPolyDataWriter()
		writer_poly.SetInputData(polydata)
		writer_poly.SetFileName(os.path.join('/work/lpzmateo/data/DL_shapes/tests/gan_test_'+str(i)+'.vtk'))
		writer_poly.Update()
		print(i)





				

