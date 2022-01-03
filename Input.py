import json, os

''' User can write down the setting '''

PROJECT = "New_Version"


START = '2017-01-01'
FINISH = '2022-12-21'


# indicator
## 기간이 같거나, TP, Converting, Factory_info 파일만 바꾼 경우 -> True
USE_PRIOR_PREPROCESS = False
''' if the use_prior_input is False '''
# user puts the input file name down
# 1. If 'do_proproc' is True
'''
if you want preprocessing with all series in the data, write "all", 
if not, fill the number what you want to simulate for in the list 
'''

# If reuse the prior file on preprocess --> 전처리 안 할 때는 굳이 안 건드려도 됨
PATH_PRIOR_PROCESS = "Layout_data_20191211_20201118.json"

# If you want to run the Preprocess.py
PATH_ACTIVITY = "Layout_Activity.xlsx"
PATH_BOM = "Layout_BOM.xlsx"

# mandatory path of data for simulation
PATH_CONVERTING = "Converting.xlsx"  # mapping department to factory
PATH_DOCK_AND_SERIES = "호선도크.xlsx"  # mapping series to dock
PATH_DISTANCE = "from_to_distance.xlsx"  # information about distance factory to factory
PATH_ROAD = "from_to_objectid.csv"
PATH_PROCESS_INFO = "Factory_info.xlsx"  # Capacity and in&out point

# information for resource : Transporter
PATH_TP = 'transporter.xlsx'

def create_path(input_path, output_path, project_name):
    # Create Project Path
    if not os.path.exists("./{0}".format(project_name)):
        os.makedirs("./{0}".format(project_name))
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)


if __name__ == "__main__":
    input_data = dict()
    INPUT_PATH = "./{0}/Input/".format(PROJECT)
    OUTPUT_PATH = "./{0}/Result/".format(PROJECT)

    # path of input and output data
    input_data['default_input'] = INPUT_PATH
    input_data['default_result'] = OUTPUT_PATH
    input_data['project_name'] = PROJECT


    # create folder with input, output path when the folder is not in the path
    create_path(INPUT_PATH, OUTPUT_PATH, PROJECT)

    # set whether need to pre-processing
    input_data['use_prior_process'] = USE_PRIOR_PREPROCESS

    if USE_PRIOR_PREPROCESS:  # 기존 전처리 파일 사용
        input_data['path_preprocess'] = input_data['default_input'] + PATH_PRIOR_PROCESS

    else:  # 새로 전처리 하는 경우
        input_data['path_activity_data'] = INPUT_PATH + PATH_ACTIVITY
        input_data['path_bom_data'] = INPUT_PATH+ PATH_BOM

        input_data['start_date'] = START
        input_data['finish_date'] = FINISH
        start_temp = START.split("-")
        start_date = start_temp[0] + start_temp[1] + start_temp[2]
        finish_temp = FINISH.split("-")
        finish_date = finish_temp[0] + finish_temp[1] + finish_temp[2]

    # the other datas path to simulate
    input_data['path_converting_data'] = INPUT_PATH + PATH_CONVERTING
    input_data['path_dock_series_data'] = INPUT_PATH + PATH_DOCK_AND_SERIES
    input_data['path_road'] = INPUT_PATH + PATH_ROAD  ## object_id
    input_data['path_distance'] = INPUT_PATH + PATH_DISTANCE  ## distance
    input_data['path_process_info'] = INPUT_PATH + PATH_PROCESS_INFO
    input_data['path_transporter'] = INPUT_PATH + PATH_TP


    input_data['parameter_stock_lag_time'] = 5
    input_data['parameter_road_warning'] = 1

    # the assumptions to simulate
    input_data['process_area'] = float("inf")
    input_data['machine_num'] = 10000

    # Save data
    with open('./{0}/input_data.json'.format(PROJECT), 'w') as f:
        json.dump(input_data, f)


