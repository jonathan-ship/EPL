import simpy, time, sys

from network import *
from 2023_simulation import *
from Preprocessing import *
from Postprocessing import *


def read_process_info(path):
    process_info_data = pd.read_excel(path)
    process_info = {}
    inout = {}

    for i in range(len(process_info_data)):
        temp = process_info_data.iloc[i]
        name = temp['name']
        process_type = temp['type']
        if process_type not in process_info.keys():
            process_info[process_type] = {}
        process_info[process_type][name] = temp['Capacity']
        inout[name] = [temp['in'], temp['out']]

    virtual_list = ['Stockyard', 'Shelter', 'Painting']
    for virtual in virtual_list:
        if virtual == "Stockyard":
            process_info['Stockyard'][virtual] = float('inf')
        elif virtual == 'Shelter':
            process_info['Shelter'][virtual] = float('inf')
        else:
            process_info['Painting'][virtual] = float('inf')

        inout[virtual] = [virtual, virtual]

    return process_info, inout


def read_converting(path):
    converting_df = pd.read_excel(path)
    converting_dict = dict()
    for idx in range(len(converting_df)):
        department = converting_df.iloc[idx]['Department']
        if department not in converting_dict.keys():
            converting_dict[department] = converting_df.iloc[idx]['Factory']
        else:
            factory = converting_dict[department]
            if type(factory) == str:
                converting_dict[department] = [factory]
            converting_dict[department].append(converting_df.iloc[idx]['Factory'])
    return converting_dict


def read_dock_series(path: object) -> object:
    data = pd.read_excel(path)
    dock_series_mapping = {str(data.iloc[i]['호선']): data.iloc[i]['도크'] for i in range(len(data))}

    return dock_series_mapping


def read_road(path_distance, path_objectid, data_path):
    network_distance = convert_to_json_road(path_distance, path_objectid, data_path)

    return network_distance


'''
Modeling
* monitor -> repetitive action of every simulation => no module 
* Resource -> need modeling just once => no modeling module, but hv resource_information module
- Part
- Process 
- Stock yard 
'''


def modeling_TP(path_tp, env, inout):
    tp_info = pd.read_excel(path_tp)
    resource_tp = dict()
    for i in range(len(tp_info)):
        temp = tp_info.iloc[i]
        yard = temp['운영구역'][0]
        if yard in ["1", "2"]:
            tp_name = temp['장비번호']
            tp_name = tp_name if tp_name not in resource_tp.keys() else '{0}_{1}'.format(tp_name, 0)
            capacity = temp['최대 적재가능 블록중량(톤)']
            v_loaded = temp['만차 이동 속도(km/h)'] * 24 * 1000
            v_unloaded = temp['공차 이동 속도(km/h)'] * 24 * 1000
            yard = int(yard)
            if yard not in resource_tp.keys():
                resource_tp[yard] = dict()
            # 초기 위치 무작위 설정
            resource_tp[yard][tp_name] = Transporter(env, tp_name, capacity, v_unloaded, v_loaded,
                                                     location=inout[random.choice(list(inout.keys()))][
                                                         random.choice([0, 1])])

    return resource_tp


def modeling_blocks(environment, data, process_dict, block_dict, monitor_class, resource_management, sink,
                    dock_dict=None):
    for part_name in data:
        part_data = data[part_name]
        temp_part = part_name.split("_")
        series = temp_part[0]

        if (series in dock_dict.keys()) and (dock_dict[series] in [1, 2, 3, 4, 5, 8, 9]):
            yard = 1 if dock_dict[series] in [1, 2, 3, 4, 5] else 2
            operation_list = list()
            idx = 0
            while True:
                operation_list.append(Operation(part_data['data'][idx], part_data['data'][idx + 1], part_data['data'][idx + 2], part_data['data'][idx + 3]))
                if part_data['data'][idx + 2] != "Sink":
                    idx += 4
                else:
                    break

            block_name = "{0}_{1}".format(temp_part[0], temp_part[1])
            block_dict[block_name] = Block(env, block_name, part_data['area'], part_data['weight'], operation_list,
                                           dock_dict[series], child=part_data['child_block'],
                                           parent=part_data['parent_block'])
            is_start = False if part_data['child_block'] is not None else True
            source_dict[block_name] = Source(env, block_dict[block_name], process_dict, resource_management, monitor,
                                             sink, is_start=is_start)
            # part_dict[part_name] = Part(environment, part_name, block, part_data['data'], block_dict, part_dict,
            #                             process_dict, convert_dict, inout, distance_matrix, dock_dict[series],
            #                             part_data['size'], part_data['area'], monitor_class, resource_class,
            #                             child=part_data['child_block'], parent=part_data['parent_block'],
            #                             stock_lag=lag_time)

    return block_dict, source_dict


if __name__ == "__main__":
    start = time.time()
    # 1. read input data
    with open('./Entire/input_data.json', 'r') as f:
        input_data = json.load(f)

    process_info, inout = read_process_info(input_data['path_process_info'])
    converting = read_converting(input_data['path_converting_data'])
    dock = read_dock_series(input_data['path_dock_series_data'])

    # if need to preprocess with activity and bom
    if input_data['use_prior_process']:
        with open(input_data['path_preprocess'], 'r') as f:
            sim_data = json.load(f)
        print("Finish data loading at {0} seconds".format(round(time.time() - start, 2)), flush=True)
    else:
        print("Start combining Activity and BOM data at {0} seconds".format(round(time.time() - start, 2)), flush=True)
        data_preprocess_path = processing_with_activity_N_bom(input_data, dock, converting)

        print("Finish data preprocessing at {0} seconds".format(round(time.time() - start, 2)), flush=True)

        with open(data_preprocess_path, 'r') as f:
            sim_data = json.load(f)
        print("Finish data loading at {0} seconds".format(round(time.time() - start, 2)), flush=True)

    initial_date = sim_data['initial_date']
    block_data = sim_data['block_info']
    print("Total Blocks = {0}".format(len(block_data)), flush=True)
    network_distance = read_road(input_data['path_distance'], input_data['path_road'], input_data['default_input'])

    # define simulation environment
    env = simpy.Environment()
    process_dict = dict()
    source_dict = dict()
    block_dict = dict()

    # Modeling
    # 1. Monitor, Sink
    monitor = Monitor(input_data['project_name'], pd.to_datetime(initial_date))
    sink = Sink(env, monitor, len(block_data))

    # 2. Resource
    tp_dict = modeling_TP(input_data['path_transporter'], env, inout)
    resource_management = Management(env, tp_dict, process_dict, source_dict, network_distance, inout, block_dict,
                                     converting, monitor)

    # 3. Process - Factory, Stockyard
    for process_type in process_info.keys():
        process_dict[process_type] = dict()
        for process_name in process_info[process_type].keys():
            if process_type != "Stockyard":
                process_dict[process_type][process_name] = Factory(env, process_name, process_type,
                                                                   process_info[process_type][process_name],
                                                                   block_dict, process_dict, source_dict,
                                                                   resource_management, sink, monitor)
            else:
                process_dict[process_type][process_name] = Stockyard(env, process_name,
                                                                     process_info[process_type][process_name],
                                                                     process_dict, block_dict, source_dict,
                                                                     resource_management, monitor)

    # 4. Block, Source
    block_dict, source_dict = modeling_blocks(env, block_data, process_dict, block_dict, monitor,
                                              resource_management, sink, dock_dict=dock)

    start_sim = time.time()
    print("START Simulation")
    env.run()
    finish_sim = time.time()
    print("FINISH SIMULATION", flush=True)
    print("Execution time: {0} seconds".format(round(finish_sim - start_sim, 2)), flush=True)
    # path_event_tracer, path_tp_info, path_road_info = monitor.save_information()
    path_event_tracer = monitor.get_logs(input_data['default_result'])
    print("number of part created = ", monitor.created, flush=True)
    print("number of completed = ", monitor.completed, flush=True)

    # output_path = dict()
    # output_path['input_path'] = './230827/input_data.json'
    # output_path['event_tracer'] = path_event_tracer
    # output_path['inout'] = inout
    # output_path['path_preprocess'] = input_data['path_preprocess'] if input_data['use_prior_process'] else data_preprocess_path
    #
    # with open(input_data['default_result'] + 'result_path.json', 'w') as f:
    #     json.dump(output_path, f)
    # print("Start Post-Processing", flush=True)
    # post_processing(input_data['default_result'] + 'result_path.json')
