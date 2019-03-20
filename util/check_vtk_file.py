import os,sys,vtk

script_path = os.path.dirname(os.path.abspath( __file__ ))
sys.path.append(os.path.join(script_path,'../src'))

from utilities import Utilities



csv_path = '/work/lpzmateo/data/DL_shapes/shapes/SubCortical_shapes/data_description.csv'

util=Utilities()


csv_dict = util.readDictCSVFile(csv_path)
print('start')
for path in csv_dict['VTK Files']:

	reader = vtk.vtkPolyDataReader()
	reader.SetFileName(path)
	reader.Update()
	polydata = reader.GetOutput()

	for i in range(polydata.GetNumberOfPoints()):
		ptn = polydata.GetPoint(i)
		for n in ptn:
			if (n>10000000):
				print(path)

