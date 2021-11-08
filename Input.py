import json, os

''' User can write down the setting '''
# set default path
INPUT_PATH = "./data/"
PROJECT = 'Series20'

# indicator
USE_PRIOR_PREPROCESS = False

''' if the use_prior_input is False '''
# user puts the input file name down
# 1. If 'do_proproc' is True
'''
if you want preprocessing with all series in the data, write "all", 
if not, fill the number what you want to simulate for in the list 
'''

# If reuse the prior file on preprocess
PATH_PRIOR_PROCESS = 'Layout_data.json'

# If you want to run the Preprocess.py
PATH_ACTIVITY = 'Layout_Activity.xlsx'
PATH_BOM = 'Layout_BOM.xlsx'

# SERIES = [i+1 for i in range(10)]
SERIES = [i+1 for i in range(10)]

# mandatory path of data for simulation
PATH_CONVERTING = 'converting_new.json'  # mapping department to factory
PATH_DOCK_AND_SERIES = '호선도크.xlsx'  # mapping series to dock
PATH_DISTANCE = 'network_distance.json'  # information about distance factory to factory
PATH_ROAD = 'network_edge.json'
PATH_PROCESS_INFO = 'Factory_info.xlsx'  # Capacity and in&out point
PATH_STOCK_AREA = 'Stockyard_area.xlsx'

# mandatory variables for simulation
PROCESS_AREA = float("inf")  # default factory area
MACHINE_NUM = 1000  # default

# information for resource : Transporter
PATH_TP = 'transporter.xlsx'


def create_path(input_path, project_name):
    if not os.path.exists(input_path[:-1]):
        os.makedirs(input_path[:-1])

    result_path = './result/' + project_name
    # if it doesn't have result folder
    if not os.path.exists('./result'):
        os.makedirs('./result')

    if not os.path.exists(result_path):
        os.makedirs(result_path)


if __name__ == "__main__":
    input_data = dict()

    # path of input and output data
    input_data['default_input'] = INPUT_PATH
    input_data['default_result'] = './result/' + PROJECT + '/'
    input_data['project_name'] = PROJECT


    # create folder with input, output path when the folder is not in the path
    create_path(INPUT_PATH, PROJECT)

    # set whether need to pre-processing
    input_data['use_prior_process'] = USE_PRIOR_PREPROCESS

    if USE_PRIOR_PREPROCESS:  # 기존 전처리 파일 사용
        input_data['path_preprocess'] = INPUT_PATH + PATH_PRIOR_PROCESS

    else:  # 새로 전처리 하는 경우
        input_data['path_activity_data'] = INPUT_PATH + PATH_ACTIVITY
        input_data['path_bom_data'] = INPUT_PATH + PATH_BOM

        input_data['series_to_preproc'] = SERIES

    # the other datas path to simulate
    input_data['path_converting_data'] = INPUT_PATH + PATH_CONVERTING
    input_data['path_dock_series_data'] = INPUT_PATH + PATH_DOCK_AND_SERIES
    input_data['path_road'] = INPUT_PATH + PATH_ROAD
    input_data['path_distance'] = INPUT_PATH + PATH_DISTANCE
    input_data['path_process_info'] = INPUT_PATH + PATH_PROCESS_INFO
    input_data['path_transporter'] = INPUT_PATH + PATH_TP

    # the assumptions to simulate
    input_data['process_area'] = PROCESS_AREA
    input_data['machine_num'] = MACHINE_NUM

    # Save data
    with open(input_data['default_result'] + 'input_data.json', 'w') as f:
        json.dump(input_data, f)

    # with open(input_data['default_result'] + 'main.bat', 'w') as f:
    #     f.write("call conda env list \n")
    #     f.write("call conda activate env_sim \n")
    #     f.write("cd C:/Users/sohyon/PycharmProjects/Simulation_Module \n")
    #     f.write("call python main.py C:/Users/sohyon/PycharmProjects/Simulation_Module/data/input_data_tp.json \n")
    #     f.write("\npause")

