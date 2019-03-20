import os,sys

script_path = os.path.dirname(os.path.abspath( __file__ ))
sys.path.append(os.path.join(script_path,'../src'))

from utilities import Utilities


csv_path = '/work/lpzmateo/data/DL_shapes/shapes/no_features_no_groups/rescaled/dataset_description.csv'

util=Utilities()


csv_dict = util.readDictCSVFile(csv_path)

new_dict=dict()
new_dict['Input VTK Files']  = csv_dict['VTK Files']
new_dict['Output VTK Files'] = csv_dict['VTK Files']

new_path = '/work/lpzmateo/data/DL_shapes/shapes/no_features_no_groups/rescaled/autoencoder_dataset_description.csv'

util.writeDictCSVFile(new_dict,new_path)