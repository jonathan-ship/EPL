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


def get_TP_information(tp_df):
    tp_info = dict()
    for i in range(len(tp_df)):
        temp = tp_df.iloc[i]
        yard = temp['운영구역'][0]
        if yard in ["1", "2"]:
            capacity = temp['최대 적재가능 블록중량(톤)']
            yard = int(yard)
            if yard not in tp_info.keys():
                tp_info[yard] = dict()

            if capacity not in tp_info[yard].keys():
                tp_info[yard][capacity] = 0

            tp_info[yard][capacity] += 4 * 8 * 1000

    return tp_info


def road_usage(event_tracer, network_road, inout, result_path):
    '''
    unload, 호출된 TP가 호출한 공장까지 가는 이벤트 : tp_unloaded_start
    load, tp가 블록이랑 같이 이동하는 이벤트 : tp_loaded_start
    '''

    # Select Target Events]
    event_tracer_road = event_tracer[event_tracer['Resource'] == "Transporter"]
    event_tracer_road = event_tracer_road.reset_index(drop=True)

    # Variables
    road_usage_dict = dict()

    for i in range(len(event_tracer_road)):
        temp = event_tracer_road.iloc[i]

        from_process = temp["From"]
        to_process = temp["To"]
        used_road = network_road[inout[from_process][1]][inout[to_process][0]]

        for road_id in used_road:
            object_id = road_id[0]
            if object_id not in road_usage_dict.keys():
                road_usage_dict[object_id] = 0
            # 사용 횟수
            road_usage_dict[object_id] += 1

    road_df = pd.DataFrame(road_usage_dict, index=["Times"])
    road_df = road_df.transpose()
    road_df.to_excel(result_path + "/Road Usage.xlsx")


def tp_index(event_tracer, tp_info, project_dock, result_path):  # get get tp moving distance and time each day
    # Select Target Events
    event_tracer_road = event_tracer[event_tracer['Resource'] == "Transporter"]
    event_tracer_road.loc[:, 'Date'] = pd.to_datetime(event_tracer_road['Date'], format='%Y-%m-%d')

    # event_tracer_road = event_tracer_road.reset_index(drop=True)
    time_list = list(np.unique(list(event_tracer_road['Date'])))
    daily_event_grp = event_tracer_road.groupby(event_tracer_road['Date'])

    capacity_1 = list(tp_info[1].keys())
    capacity_1.sort()
    capacity_2 = list(tp_info[2].keys())
    capacity_2.sort()
    capacity_dict = {1: capacity_1, 2: capacity_2}
    tp_output = {1: {time: {capacity: 0 for capacity in capacity_1} for time in time_list},
                 2: {time: {capacity: 0 for capacity in capacity_2} for time in time_list}}
    date_time_list = list()
    for date_time in time_list:
        date_time_list.append(date_time.date())
        daily_event = daily_event_grp.get_group(date_time)
        daily_event = daily_event.reset_index(drop=True)
        for idx in range(len(daily_event)):
            temp = daily_event.iloc[idx]
            series = temp['Part'][:5]
            yard = 1 if project_dock[series] in [1, 2, 3] else 2

            load = temp['Load'] if temp['Load'] <= max(tp_info[yard].keys()) else max(tp_info[yard].keys())
            available_tp = min(list(filter(lambda x: x >= load, tp_info[yard].keys())))
            tp_output[yard][date_time][available_tp] += temp['Distance']


        tp_output[1][date_time] = {
            "{0}_1".format(capacity): round(tp_output[1][date_time][capacity] / tp_info[1][capacity], 1) for capacity
            in capacity_dict[1]}
        tp_output[2][date_time] = {
            "{0}_2".format(capacity): round(tp_output[2][date_time][capacity] * 100 / tp_info[2][capacity], 1) for capacity
            in capacity_dict[2]}

    yard_1 = pd.DataFrame.from_dict(tp_output[1])
    yard_1 = yard_1.transpose()

    yard_2 = pd.DataFrame.from_dict(tp_output[2])
    yard_2 = yard_2.transpose()
    yard_2['Date'] = date_time_list

    yard_tp = pd.concat([yard_1, yard_2], 1)
    yard_tp = yard_tp.reset_index(drop=True)
    tp_num_list = list()
    tp_capa_list = list(yard_tp)
    for tp_capa in tp_capa_list:
        if tp_capa != "Date":
            temp = tp_capa.split("_")
            capa = int(temp[0])
            yard = int(temp[1])
            tp_num = int((tp_info[yard][capa]) / (8 * 4 * 1000))
            tp_num_list.append(tp_num)
    tp_num_list.append("Number")
    num_df = pd.DataFrame(columns=tp_capa_list)
    num_df.loc[0] = tp_num_list
    yard_tp = pd.concat([num_df, yard_tp], 0)
    yard_tp = yard_tp.set_index("Date", drop=True)
    yard_tp.to_excel(result_path + "Transporter.xlsx")


def calculate_occupied_area(result_path, event_tracer, factory_info):
    if not os.path.exists(result_path + "/Load"):
        os.makedirs(result_path + "/Load")
    event_tracer = event_tracer[
        (event_tracer['Event'] == "Process_entered") | (event_tracer['Event'] == "part_transferred_to_out_buffer") | (
                    event_tracer['Event'] == "Stock_in") | (event_tracer['Event'] == "Stock out")]
    factory_type = list(np.unique(list(event_tracer['Process Type'])))
    event_tracer_grp = event_tracer.groupby(event_tracer['Process Type'])
    for types in factory_type:
        factory_grp = event_tracer_grp.get_group(types)
        factories = list(np.unique(list(factory_grp['Process'])))
        each_factory_group = factory_grp.groupby(factory_grp['Process'])

        writer = pd.ExcelWriter(result_path + "/Load/{0}.xlsx".format(types), engine='xlsxwriter')

        for factory in factories:
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
                occupied_ratio = int(temp['Load'] / factory_info[factory] * 100) if factory not in ["Stock", "Shelter", "Painting"] else None
                capacity_ratio.append(occupied_ratio)
                overload_ratio = occupied_ratio if (factory not in ["Stock", "Shelter", "Painting"]) and (occupied_ratio >= 100) else None
                overload.append(overload_ratio)

            factory_df = pd.DataFrame([], columns=[str(factory_info[factory]), "Load", "Ratio[%]", "OverLoad[%]"])
            factory_df[str(factory_info[factory])] = time
            factory_df["Load"] = load
            factory_df["Ratio[%]"] = capacity_ratio
            factory_df["OverLoad[%]"] = overload

            factory_df.to_excel(writer, sheet_name=factory, index=False)
        writer.save()


def calculate_block_moving_distance(event_tracer, block_info, result_path):
    block_list = list(np.unique(list(event_tracer['Part'])))
    block_group = event_tracer.groupby(event_tracer['Part'])
    block_code_list = list()
    distance_list = list()
    for block_code in block_list:
        if block_code in block_info.keys():
            block_event = block_group.get_group(block_code)
            moving_event = block_event[block_event['Resource']=="Transporter"]
            if len(moving_event) > 0:
                moving_distance = list(moving_event['Distance'])
                moving_distance = [i for i in moving_distance if i != 0]
                if len(moving_distance) > 0:
                    avg_distance = np.mean(moving_distance)
                    distance_list.append(avg_distance)
                    block_code_list.append(block_code)

    fig, ax = plt.subplots()
    bins = np.linspace(0, max(distance_list), 50)
    ax.hist(distance_list, density=False, histtype='stepfilled', alpha=0.2, bins=bins)
    plt.savefig(result_path+'/Block Moving Distance.png', dpi=600)

    distance_df = pd.DataFrame(columns=["Block", "Avg.Distance"])
    distance_df["Block"] = block_code_list
    distance_df["Avg.Distance"] = distance_list
    distance_df.to_excel(result_path + "/Block Moving Distance.xlsx", index=False)


def road_warning(event_tracer, network_road, inout, block_info, warning_diff, result_path):
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

    road_warning_df.to_excel(result_path + "Road Warning.xlsx", index=False)


def post_processing(json_path):
    #with open(sys.argv[1], 'r') as f:
    with open(json_path, 'r') as f:
        result_path = json.load(f)

    event_tracer = pd.read_csv(result_path['event_tracer'])

    with open(result_path['input_path'], 'r') as f:
        input_data = json.load(f)

    with open(input_data['default_input'] + 'network_edge.json', 'r') as f:
        network_road = json.load(f)

    preproc_data_path = input_data['path_preprocess']
    with open(preproc_data_path, 'r') as f:
        preproc_data = json.load(f)

    tp_df = pd.read_excel(input_data['path_transporter'])

    dock_data = pd.read_excel(input_data['path_dock_series_data'])
    dock = {dock_data.iloc[i]['호선']: dock_data.iloc[i]['도크'] for i in range(len(dock_data))}

    start_date_time = pd.to_datetime(preproc_data['simulation_initial_date'], format='%Y-%m-%d')
    finish_date_time = pd.to_datetime(preproc_data['simulation_finish_date'], format='%Y-%m-%d')
    post_range_days = (finish_date_time - start_date_time).days + 1

    event_tracer.loc[:, 'Date'] = pd.to_datetime(event_tracer['Date'], format='%Y-%m-%d')
    #event_tracer = event_tracer[(event_tracer['Date'] >= start_date_time) & (event_tracer['Date'] <= finish_date_time)]
    block_info = preproc_data['block_info']

    # 1. Road
    road_usage(event_tracer, network_road, result_path['inout'], input_data['default_result'])

    # 2. Transporter
    tp_info = get_TP_information(tp_df)
    tp_index(event_tracer, tp_info, dock, input_data['default_result'])

    # 3. Block Moving Distance
    calculate_block_moving_distance(event_tracer, block_info, input_data['default_result'])

    # 4. Occupied Area
    factory_info = pd.read_excel(input_data['path_process_info'])
    factory_dict = dict()
    for i in range(len(factory_info)):
        temp = factory_info.iloc[i]
        factory_dict[temp['name']] = temp['Capacity']
    factory_dict['Stock'] = float("inf")
    factory_dict["Painting"] = float("inf")
    factory_dict["Shelter"] = float("inf")
    calculate_occupied_area(input_data['default_result'], event_tracer, factory_dict)

    road_warning(event_tracer, network_road, result_path['inout'], block_info, input_data['parameter_road_warning'],
                 input_data['default_result'])
