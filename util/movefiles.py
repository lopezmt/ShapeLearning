import os,sys
import glob

script_path = os.path.dirname(os.path.abspath( __file__ ))
sys.path.append(os.path.join(script_path,'src'))

from utilities import Utilities







#ls /ASD/Autism2/IBIS1/IBIS/Proc_Data/*/V*/mri/registered_v3.13/sMRI/AutoSeg_SubCort/SALT/stx_noscale*t1w_RAI_Bias_label_editLV_HippoR_pp_surfSPHARM_procalign.vtk


#input
shape_names = ['ThalR','ThalL','PutR','Putl','HippoR','HippoL','GPR','GPL','CaudR','CaudL','AmyR','AmyL']
path_part_1 = '/ASD/Autism2/IBIS1/IBIS/Proc_Data/*/V*/mri/registered_v3.13/sMRI/AutoSeg_SubCort/SALT/stx_noscale*t1w_RAI_Bias_label_editLV_'
path_part_2 = '_pp_surfSPHARM_procalign.vtk'

#output
out_path='/work/lpzmateo/data/DL_shapes/shapes/SubCortical_shapes'


util=Utilities()



final_files=dict()
final_files['VTK Files']=[]

for i in range(len(shape_names)):

	shape_name=shape_names[i]

	shapes_path = os.path.join(out_path,shape_name)
	try:
		os.mkdir(shapes_path)
	except FileExistError:
		pass


	path_proto=path_part_1+shape_name+path_part_2
	dirlist=glob.glob(path_proto)
	print('\n -------------------------------------------------- \n')
	print(len(dirlist))
	for n in range(len(dirlist)):
		path = dirlist[n]

		new_path = os.path.join(shapes_path,str(n)+'.vtk')
		final_files['VTK Files'].append(new_path)
		util.force_symlink(path,new_path)
		print(path)

description_path = os.path.join(out_path,'data_description.csv')

util.writeDictCSVFile(final_files,description_path)


print(description_path)