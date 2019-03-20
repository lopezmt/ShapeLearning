import os,sys,argparse

script_path = os.path.dirname(os.path.abspath( __file__ ))
sys.path.append(os.path.join(script_path,'src'))
from shapeDataExtractor import ShapeDataExtractor



parser = argparse.ArgumentParser(description='Shape data extractor', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataDescription', action='store', dest='csvPath', default=None,help='csv file describing each shape' )
parser.add_argument('--feature_points', help='Extract the following features from the polydatas Points', nargs='+', default=None, type=str)
parser.add_argument('--feature_cells', help='Extract the following features from the polydatas Cells', nargs='+', default=None, type=str)
parser.add_argument('--out_feature_points', help='Extract the following features from the output polydatas Points', nargs='+', default=None, type=str)
parser.add_argument('--out_feature_cells', help='Extract the following features from the output polydatas Cells', nargs='+', default=None, type=str)
parser.add_argument('--auto_complete', action='store_true', dest='autocomplete_feature_name',help='Set autocompletion of features name during the data extraction ')
parser.add_argument('--out', dest="tfrecord_path", help='TFRecords directory output', default="./", type=str)


if __name__=='__main__':
	args = parser.parse_args()

	csvPath = args.csvPath

	feature_points = args.feature_points
	feature_cells = args.feature_cells

	out_feature_points = args.out_feature_points
	out_feature_cells = args.out_feature_cells

	tfrecord_path = args.tfrecord_path
	auto_complete=args.autocomplete_feature_name

	data_extractor=ShapeDataExtractor()

	data_extractor.setCSVDescription(csvPath)
	data_extractor.setPointFeature(feature_points)
	data_extractor.setCellFeature(feature_cells)

	data_extractor.setOutPointFeature(feature_points)
	data_extractor.setOutCellFeature(feature_cells)

	data_extractor.setOutputDirectory(tfrecord_path)

	tfrecord_info_path=data_extractor.saveTFRecords(autocomplete_feature_name=auto_complete)
	print(tfrecord_info_path)