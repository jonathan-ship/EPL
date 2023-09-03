import copy
import json, math, os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import font_manager, rc
from xlsxwriter import Workbook

# default - in Window
font_name = font_manager.FontProperties(fname="C:\Windows\Fonts\H2GTRM.TTF").get_name()

# At Apple Product
# font_at_prof = '/Users/jonathan/Library/Fonts/NanumSquareB.ttf'
# font_name = font_manager.FontProperties(fname=font_at_prof).get_name()

rc('font', family=font_name)


def get_TP_information(path_tp):
    tp_info = pd.read_excel(path_tp)
    resource_tp = dict()
    for i in range(len(tp_info)):
        temp = tp_info.iloc[i]
        yard = temp['운영구역'][0]
        if yard in ["1", "2"]:
            tp_name = temp['장비번호']
            tp_name = tp_name if tp_name not in resource_tp.keys() else '{0}_{1}'.format(tp_name, 0)
            capacity = temp['최대 적재가능 블록중량(톤)']
            yard = int(yard)
            if yard not in resource_tp.keys():
                resource_tp[yard] = dict()
            resource_tp[yard][tp_name] = capacity

    return resource_tp


def road_usage(event_tracer, network_road, inout, result_path, tp_info):
    '''
    unload, 호출된 TP가 호출한 공장까지 가는 이벤트 : tp_unloaded_start
    load, tp가 블록이랑 같이 이동하는 이벤트 : tp_loaded_start
    '''
    # road_id 불러오기
    road_usage_dict = dict()
    for from_process in network_road.keys():
        for to_process in network_road[from_process].keys():
            object_id_list = network_road[from_process][to_process]
            if len(object_id_list) >= 1:
                for road_id in object_id_list:
                    object_id = road_id[0]
                    if object_id not in road_usage_dict.keys():
                        road_usage_dict[object_id] = 0

    # Select Target Events
    tp_list = list(tp_info[1].keys()) + list(tp_info[2].keys())
    # event_tracer_road = event_tracer[event_tracer['Resource'] == "Transporter"]
    # event_tracer_road = event_tracer_road.reset_index(drop=True)

    # Variables
    for tp_name in tp_list:
        event_tracer_road = event_tracer[(event_tracer["Resource"] == tp_name) & (event_tracer["Distance"] > 0)]
        event_tracer_road = event_tracer_road[
            (event_tracer_road["Event"] == "Transporter Loading Start") | (
                        event_tracer_road["Event"] == "Transporter Load Complete")]
        event_tracer_road = event_tracer_road[event_tracer_road["Distance"] > 0]
        event_tracer_road = event_tracer_road.reset_index(drop=True)
        for i in range(len(event_tracer_road)):
            temp = event_tracer_road.iloc[i]

            from_process = temp["From"]
            to_process = temp["To"]
            used_road = network_road[inout[from_process][1]][inout[to_process][0]]

            for road_id in used_road:
                object_id = road_id[0]
                # 사용 횟수
                road_usage_dict[object_id] += 1

    road_df = pd.DataFrame(road_usage_dict, index=["Times"])
    road_df = road_df.transpose()
    road_df.to_excel(result_path + "/Road Usage.xlsx")
    print("FINISH Calculating Road Usage", flush=True)


def tp_index(event_tracer, tp_info, project_dock, result_path, start_time, finish_time):  # get get tp moving distance and time each day
    # Select Target Events
    event_tracer_road = event_tracer[
        (event_tracer["Event"] == "Transporter Loading Start") | (event_tracer["Event"] == "Transporter Loading Completed")]
    event_tracer_road = event_tracer_road[event_tracer_road["Distance"] > 0]
    event_tracer_road = event_tracer_road.reset_index(drop=True)

    # event_tracer_road = event_tracer_road.reset_index(drop=True)
    time_list = list(np.unique(list(event_tracer_road['Date'])))
    daily_event_grp = event_tracer_road.groupby(event_tracer_road['Date'])

    tp_output_retrival = dict()
    for tp_name in tp_info[1].keys():
        tp_output_retrival["1_{0}_{1}".format(tp_name, tp_info[1][tp_name])] = {time: 0.0 for time in time_list}
    for tp_name in tp_info[2].keys():
        tp_output_retrival["2_{0}_{1}".format(tp_name, tp_info[2][tp_name])] = {time: 0.0 for time in time_list}

    tp_output_loading = copy.deepcopy(tp_output_retrival)

    tp_list = list(tp_info[1].keys()) + list(tp_info[2].keys())
    date_time_list = list()
    for date_time in time_list:
        date_time_list.append(date_time.date())
        daily_event = daily_event_grp.get_group(date_time)
        daily_event = daily_event.reset_index(drop=True)
        for idx in range(len(daily_event)):
            temp = daily_event.iloc[idx]
            if temp["Resource"] in tp_list:
                temp_part = temp['Part'].split("_")
                series = temp_part[0]
                yard = 1 if project_dock[series] in [1, 2, 3] else 2
                if temp["Event"] == "Transporter Loading Start":
                    tp_output_retrival["{0}_{1}_{2}".format(yard, temp["Resource"], tp_info[yard][temp["Resource"]])][date_time] += temp['Distance']
                else:
                    tp_output_loading["{0}_{1}_{2}".format(yard, temp["Resource"], tp_info[yard][temp["Resource"]])][date_time] += temp['Distance']

    yard_tp_retrival = pd.DataFrame.from_dict(tp_output_retrival)
    yard_tp_retrival["Date"] = date_time_list
    yard_tp_retrival = yard_tp_retrival.set_index("Date", drop=True)

    yard_tp_loading = pd.DataFrame.from_dict(tp_output_loading)
    yard_tp_loading["Date"] = date_time_list
    yard_tp_loading = yard_tp_loading.set_index("Date", drop=True)
    # yard_tp = pd.concat([yard_1, yard_2], axis=1)
    # yard_tp = yard_tp.reset_index(drop=True)
    # tp_num_list = list()
    # tp_capa_list = list(yard_tp)
    # for tp_capa in tp_capa_list:
    #     if tp_capa != "Date":
    #         temp = tp_capa.split("_")
    #         capa = int(temp[0])
    #         yard = int(temp[1])
    #         tp_num = int((tp_info[yard][capa]) / (8 * 4 * 1000))
    #         tp_num_list.append(tp_num)
    # tp_num_list.append("Number")
    # num_df = pd.DataFrame(columns=tp_capa_list)
    # num_df.loc[0] = tp_num_list
    # yard_tp = pd.concat([num_df, yard_tp], axis=0)
    # yard_tp = yard_tp.set_index("Date", drop=True)
    # if len(yard_tp) == 1:  # Data 가 없을 때
    #     yard_tp.loc[start_time.date()] = [0 for _ in range(len(yard_tp.columns))]
    #     yard_tp.loc[finish_time.date()] = [0 for _ in range(len(yard_tp.columns))]
    # capa_yard_list = list(yard_tp.columns)
    # # for capa_yard in capa_yard_list:
    # #     temp = capa_yard.split("_")
    # #     capa = int(temp[0])
    # #     yard = int(temp[1])
    # #     capacity_of_tp = tp_info[yard][capa]
    # #     yard_tp.loc[:, capa_yard] = yard_tp.loc[:, capa_yard].apply(lambda x: round((x/capacity_of_tp) * 100, 1))

    yard_tp_retrival.to_excel(result_path + "Transporter Retrival.xlsx")
    yard_tp_loading.to_excel(result_path + "Transporter Loading.xlsx")
    print("FINISH Calculating Transporter Utilization", flush=True)


def calculate_occupied_area(result_path, event_tracer, factory_info, start_date, finish_date):
    if not os.path.exists(result_path + "/Load"):
        os.makedirs(result_path + "/Load")
    event_tracer = event_tracer[
        (event_tracer['Event'] == "Process In") | (event_tracer['Event'] == "Process Out") | (
                    event_tracer['Event'] == "Stockyard In") | (event_tracer['Event'] == "Stockyard Out")]
    factory_type = list(factory_info.keys())
    each_factory_group = event_tracer.groupby(event_tracer['Process'])
    event_factories = list(np.unique(list(event_tracer['Process'])))
    for types in factory_type:
        if types == "Stockyard":
            writer = pd.ExcelWriter(result_path + "/Load/Stock.xlsx", engine='xlsxwriter')
        else:
            writer = pd.ExcelWriter(result_path + "/Load/{0}.xlsx".format(types), engine='xlsxwriter')
        factories = list(factory_info[types].keys())
        for factory in factories:
            factory_df = pd.DataFrame([], columns=[str(factory_info[types][factory]), "Load", "Ratio[%]", "OverLoad[%]"])
            if factory in event_factories:
                factory_event = each_factory_group.get_group(factory)
                factory_event = factory_event.reset_index(drop=True)
                time = list()
                load = list()
                capacity_ratio = list()
                overload = list()
                for i in range(len(factory_event)):
                    temp = factory_event.iloc[i]
                    load.append(temp['Load'])
                    time.append(temp['Date'].date())
                    occupied_ratio = int(temp['Load'] / factory_info[types][factory] * 100) if (factory_info[types][factory] > 0) and (factory_info[types][factory] <= 1e20) else 0.0
                    capacity_ratio.append(occupied_ratio)
                    overload_ratio = occupied_ratio if (factory_info[types][factory] > 0) and (factory_info[types][factory] <= 1e20) and (occupied_ratio >= 100) else None
                    overload.append(overload_ratio)

                factory_df[str(factory_info[types][factory])] = time
                factory_df["Load"] = load
                factory_df["Ratio[%]"] = capacity_ratio
                factory_df["OverLoad[%]"] = overload
            else:
                factory_df.loc[0] = [start_date.date(), 0, 0, None]
                factory_df.loc[1] = [finish_date.date(), 0, 0, None]

            factory_df.to_excel(writer, sheet_name=factory, index=False)
        writer.save()
        print("FINISH Calculating {0} Load".format(types), flush=True)


## 여기부터
def calculate_block_moving_distance(event_tracer, block_info, result_path):
    block_list = list(np.unique(list(event_tracer['Part'])))
    block_group = event_tracer.groupby(event_tracer['Part'])
    block_code_list = list()
    total_distance_list = list()
    avg_distance_list = list()
    for block_code in block_list:
        if block_code in block_info.keys():
            block_event = block_group.get_group(block_code)
            moving_event = block_event[(block_event['Event'] == "Transporter Loading Start") | (
                        block_event['Event'] == "Transporter Loading Completed")]
            if len(moving_event) > 0:
                moving_distance = list(moving_event['Distance'])
                moving_distance = [i for i in moving_distance if i != 0]
                if len(moving_distance) > 0:
                    total_distance_list.append(np.sum(moving_distance))
                    avg_distance_list.append(np.mean(moving_distance))
                    block_code_list.append(block_code)
                else:
                    total_distance_list.append(0)
                    avg_distance_list.append(0)
                    block_code_list.append(block_code)
            else:
                total_distance_list.append(0)
                avg_distance_list.append(0)
                block_code_list.append(block_code)

    if len(block_code_list) == 0:
        block_code_list.append("No Blocks in Period")
        total_distance_list.append(0)
        avg_distance_list.append(0)
    distance_df = pd.DataFrame(columns=["Block", "Total.Distance", "Avg.Distance"])
    distance_df["Block"] = block_code_list
    distance_df["Total.Distance"] = total_distance_list
    distance_df["Avg.Distance"] = avg_distance_list
    distance_df.to_excel(result_path + "Block Moving Distance.xlsx", index=False)
    print("FINISH Calculating Average Moving Distance of Each Block", flush=True)


def road_warning(event_tracer, network_road, inout, block_info, warning_diff, result_path, start_date):
    # Select Target Events]
    event_tracer_road = event_tracer[event_tracer['Resource'] == "Transporter"]
    event_tracer_road = event_tracer_road.reset_index(drop=True)

    # Variables
    date_list = list()
    block_list = list()
    block_size_list = list()
    road_id_list = list()
    road_size_list = list()
    size_diff_list = list()
    from_factory_list = list()
    to_factory_list = list()

    for i in range(len(event_tracer_road)):
        temp = event_tracer_road.iloc[i]

        block_code = temp['Part']
        from_process = temp["From"]
        to_process = temp["To"]
        used_road = network_road[inout[from_process][1]][inout[to_process][0]]

        for road_id in used_road:
            object_id = road_id[0]
            road_size = road_id[1]
            block_size = block_info[block_code]['size']
            if (block_size - road_size) >= warning_diff:
                date_list.append(temp['Date'].date())
                block_list.append(temp['Part'])
                block_size_list.append(block_size)
                road_id_list.append(object_id)
                road_size_list.append(road_size)
                size_diff_list.append(block_size-road_size)
                from_factory_list.append(from_process)
                to_factory_list.append(to_process)

    road_warning_df = pd.DataFrame()
    road_warning_df['Date'] = date_list
    road_warning_df['Block'] = block_list
    road_warning_df['Road ID'] = road_id_list
    road_warning_df['Block Size'] = block_size_list
    road_warning_df['Road Size'] = road_size_list
    road_warning_df['Difference'] = size_diff_list
    road_warning_df['From'] = from_factory_list
    road_warning_df['To'] = to_factory_list
    if len(road_warning_df) == 0:
        road_warning_df.loc[0] = [start_date.date(), "No Block", None, None, None, None, None, None]

    road_warning_df.to_excel(result_path + "Road Warning.xlsx", index=False)
    print("FINISH Calculating Over Size of Block than Road", flush=True)


def post_processing(json_path):
    #with open(sys.argv[1], 'r') as f:
    with open(json_path, 'r') as f:
        result_path = json.load(f)

    event_tracer = pd.read_csv(result_path['event_tracer'])

    with open(result_path['input_path'], 'r') as f:
        input_data = json.load(f)

    with open(input_data['default_input'] + 'network_edge.json', 'r') as f:
        network_road = json.load(f)

    preproc_data_path = result_path['path_preprocess']
    with open(preproc_data_path, 'r') as f:
        preproc_data = json.load(f)


    dock_data = pd.read_excel(input_data['path_dock_series_data'])
    dock = {str(dock_data.iloc[i]['호선']): dock_data.iloc[i]['도크'] for i in range(len(dock_data))}

    start_date_time = pd.to_datetime(preproc_data['simulation_initial_date'], format='%Y-%m-%d')
    finish_date_time = pd.to_datetime(preproc_data['simulation_finish_date'], format='%Y-%m-%d')
    post_range_days = (finish_date_time - start_date_time).days + 1

    event_tracer.loc[:, 'Date'] = pd.to_datetime(event_tracer['Date'], format='%Y-%m-%d')
    event_tracer = event_tracer[(event_tracer['Date'] >= start_date_time) & (event_tracer['Date'] <= finish_date_time)]
    block_info = preproc_data['block_info']

    tp_info = get_TP_information(input_data['path_transporter'])

    # 1. Road
    road_usage(event_tracer, network_road, result_path['inout'], input_data['default_result'], tp_info)

    # 2. Transporter
    tp_index(event_tracer, tp_info, dock, input_data['default_result'], start_date_time, finish_date_time)

    # 3. Block Moving Distance
    calculate_block_moving_distance(event_tracer, block_info, input_data['default_result'])

    # 4. Occupied Area
    factory_info = pd.read_excel(input_data['path_process_info'])
    types = list(np.unique(list(factory_info['type'])))
    factory_grp = factory_info.groupby(factory_info['type'])
    factory_dict = dict()
    for type in types:
        factory_dict[type] = dict()
        each_type = factory_grp.get_group(type)
        each_type = each_type.reset_index(drop=True)
        for idx in range(len(each_type)):
            temp = each_type.iloc[idx]
            factory_dict[type][temp['name']] = temp['Capacity']
        if type == "Stockyard":
            factory_dict[type]['Stockyard'] = float("inf")
        elif type == "Painting":
            factory_dict[type]["Painting"] = float("inf")
        elif type == "Shelter":
            factory_dict[type]["Shelter"] = float("inf")

    calculate_occupied_area(input_data['default_result'], event_tracer, factory_dict, start_date_time, finish_date_time)

    # road_warning(event_tracer, network_road, result_path['inout'], block_info, input_data['parameter_road_warning'],
    #              input_data['default_result'], start_date_time)
    print("FINISH POST-PROCESSING", flush=True)

if __name__ == "__main__":
    # with open('./As-Is/Result/result_path.json', 'r') as f:
    #     result_path = json.load(f)
    post_processing('./Transporter/Result/result_path.json')
    # event_tracer = pd.read_csv(result_path['event_tracer'])
    #
    # with open(result_path['input_path'], 'r') as f:
    #     input_data = json.load(f)
    #
    # with open(input_data['default_input'] + 'network_edge.json', 'r') as f:
    #     network_road = json.load(f)
    #
    # preproc_data_path = result_path['path_preprocess']
    # with open(preproc_data_path, 'r') as f:
    #     preproc_data = json.load(f)
    #
    # tp_df = pd.read_excel(input_data['path_transporter'])
    #
    # dock_data = pd.read_excel(input_data['path_dock_series_data'])
    # dock = {str(dock_data.iloc[i]['호선']): dock_data.iloc[i]['도크'] for i in range(len(dock_data))}
    #
    # start_date_time = pd.to_datetime(preproc_data['simulation_initial_date'], format='%Y-%m-%d')
    # finish_date_time = pd.to_datetime(preproc_data['simulation_finish_date'], format='%Y-%m-%d')
    # post_range_days = (finish_date_time - start_date_time).days + 1
    #
    # event_tracer.loc[:, 'Date'] = pd.to_datetime(event_tracer['Date'], format='%Y-%m-%d')
    # event_tracer = event_tracer[(event_tracer['Date'] >= start_date_time) & (event_tracer['Date'] <= finish_date_time)]
    # block_info = preproc_data['block_info']
    #
    # # 1. Road
    # #road_usage(event_tracer, network_road, result_path['inout'], input_data['default_result'])
    #
    # # 2. Transporter
    # # tp_info = get_TP_information(tp_df)
    # # tp_index(event_tracer, tp_info, dock, input_data['default_result'], start_date_time, finish_date_time)
    #
    # # 3. Block Moving Distance
    # calculate_block_moving_distance(event_tracer, block_info, input_data['default_result'])
    #
    # # 4. Occupied Area
    # # factory_info = pd.read_excel(input_data['path_process_info'])
    # # types = list(np.unique(list(factory_info['type'])))
    # # factory_grp = factory_info.groupby(factory_info['type'])
    # # factory_dict = dict()
    # # for type in types:
    # #     factory_dict[type] = dict()
    # #     each_type = factory_grp.get_group(type)
    # #     each_type = each_type.reset_index(drop=True)
    # #     for idx in range(len(each_type)):
    # #         temp = each_type.iloc[idx]
    # #         factory_dict[type][temp['name']] = temp['Capacity']
    # #     if type == "Stockyard":
    # #         factory_dict[type]['Stockyard'] = float("inf")
    # #     elif type == "Painting":
    # #         factory_dict[type]["Painting"] = float("inf")
    # #     elif type == "Shelter":
    # #         factory_dict[type]["Shelter"] = float("inf")
    # #
    # # calculate_occupied_area(input_data['default_result'], event_tracer, factory_dict, start_date_time, finish_date_time)
    # #
    # # road_warning(event_tracer, network_road, result_path['inout'], block_info, input_data['parameter_road_warning'],
    # #              input_data['default_result'], start_date_time)
    # print("FINISH POST-PROCESSING", flush=True)
