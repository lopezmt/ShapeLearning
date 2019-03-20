import os,sys,argparse

script_path = os.path.dirname(os.path.abspath( __file__ ))
sys.path.append(os.path.join(script_path,'src'))
from shapeDataset import ShapeDataset



parser = argparse.ArgumentParser(description='Shape Variation Analyzer', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataDescription', action='store', dest='csvPath', default=None,help='csv file describing each shape' )
parser.add_argument('--tfrecordInfo', action='store', dest='tfrecord_info_path', default=None,help='JSON file describing the TFRecords')
parser.add_argument('--test_size', help='test ratio', type=float, default=0.1)
parser.add_argument('--validation_size', help='validation ratio', default=0.1, type=float)
parser.add_argument('--feature_points', help='Extract the following features from the polydatas GetPointData', nargs='+', default=None, type=str)
parser.add_argument('--feature_cells', help='Extract the following features from the polydatas GetCellData', nargs='+', default=None, type=str)
parser.add_argument('--out_feature_points', help='Extract the following features from the output polydatas Points', nargs='+', default=None, type=str)
parser.add_argument('--out_feature_cells', help='Extract the following features from the output polydatas Cells', nargs='+', default=None, type=str)
parser.add_argument('--auto_complete', action='store_true', dest='autocomplete_feature_name',help='Set autocompletion of features name during the data extraction ')
parser.add_argument('--smote', action='store_true', dest='smote',help='Define if SMOTE should be used to generate data, (ONLY for clasification datasets).')
parser.add_argument('--out', dest="dataset_path", help='TFRecords directory output', default="./", type=str)


if __name__=='__main__':
	args = parser.parse_args()

	csvPath = args.csvPath
	tfrecord_info_path=args.tfrecord_info_path

	feature_points = args.feature_points
	feature_cells = args.feature_cells

	out_feature_points = args.out_feature_points
	out_feature_cells = args.out_feature_cells

	dataset_path = args.dataset_path

	test_size=args.test_size
	val_size=args.validation_size
	use_smote=args.smote


	dataset=ShapeDataset()

	dataset.setDataDescription(csvPath)
	dataset.setPointFeature(feature_points)
	dataset.setCellFeature(feature_cells)

	dataset.setOutputPointFeature(out_feature_points)
	dataset.setOutputCellFeature(out_feature_cells)

	dataset.setTFRecordInfo(tfrecord_info_path)
	dataset.setOutputDirectory(dataset_path)
	dataset.setTestSize(test_size)
	dataset.setValidationSize(val_size)

	dataset.setMinNumberOfShapesPerClass(6)
	dataset.setSMOTE(use_smote)

	description_path = dataset.createDataset(autocomplete_feature_name=args.autocomplete_feature_name)

	print('output: %s' %(description_path))

	#
	#Feature extractor usage
	#
	# data_extractor = shapeDataExtractor()
	# data_extractor.setCSVDescription(csvPath)
	# data_extractor.setOutputDirectory(records_path)
	# data_extractor.setPointFeature(feature_points)
	# data_extractor.setCellFeature(feature_cells)

	# data_extractor.saveTFRecords(autocomplete_feature_name=True)

	










