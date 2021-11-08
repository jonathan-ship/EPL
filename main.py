import simpy, time, sys

from Sim_Kernel import Resource, Part, Sink, StockYard, Monitor, Process, Block, Transporter
from Preprocessing import processing_with_activity_N_bom
from Postprocessing import *


def read_process_info(path):
    process_info_data = pd.read_excel(path)
    process_info = {}
    inout = {}

    for i in range(len(process_info_data)):
        temp = process_info_data.iloc[i]
        name = temp['name']
        if ('대공장' in name) and ('1공장' in name):
            print(0)
        process_type = temp['type']
        if process_type not in process_info.keys():
            process_info[process_type] = {}
        process_info[process_type][name] = {}
        process_info[process_type][name]['capacity'] = temp['Capacity']
        process_info[process_type][name]['unit'] = temp['unit']

        inout[name] = [temp['in'], temp['out']]

    virtual_list = ['Stock', 'Shelter', 'Painting']
    for virtual in virtual_list:
        if virtual == "Stock":
            process_info['Stockyard'][virtual] = {'capacity' : float('inf'), 'unit': 'm2'}
        elif virtual == 'Shelter':
            process_info['Shelter'][virtual] = {'capacity' : float('inf'), 'unit': 'm2'}
        else:
            process_info['Painting'][virtual] = {'capacity' : float('inf'), 'unit': 'm2'}

        inout[virtual]= [virtual, virtual]

    return process_info, inout


# def read_inout(path, virtual_list=None):
#     inout_table = pd.read_excel(path)
#     process_inout = {}
#     for i in range(len(inout_table)):
#         temp = inout_table.iloc[i]
#         process_inout[temp['LOC']] = [temp['IN'], temp['OUT']]
#
#     if virtual_list is not None:
#         for virtual in virtual_list:
#             process_inout[virtual] = [virtual, virtual]
#
#     return process_inout
#

# def set_area(default_area, process_list, path_stock_area=None, path_process_area=None):
#     area = dict()
#     stock_list = None
#
#     if path_stock_area is not None:
#         stock_area_data = pd.read_excel(path_stock_area)
#         stock_list = list(stock_area_data['name'])
#         for i in range(len(stock_list)):
#             area[stock_area_data['name'][i]] = stock_area_data['area'][i]
#         if "Stock" in process_list:
#             stock_list.append("Stock")
#
#     if path_process_area is not None:
#         process_area_data = pd.read_excel(path_process_area)
#         proc_list = list(process_area_data['name'])
#         for i in range(len(proc_list)):
#             area[process_area_data['name'][i]] = process_area_data['area'][i]
#
#     for process in process_list:
#         if process not in area.keys():
#             area[process] = default_area
#
#     return area, stock_list
def read_converting(path):
    with open(path, 'r') as f:
        converting_dict = json.load(f)

    return converting_dict


def read_dock_series(path: object) -> object:
    data = pd.read_excel(path)
    dock_series_mapping = {data.iloc[i]['호선']: data.iloc[i]['도크'] for i in range(len(data))}

    return dock_series_mapping


# def read_network(path, padding=None):
#     # network data
#     network = {}
#     for i in range(10, 31):
#         road_path = path + 'distance_above_{0}_meters.xlsx'.format(i)
#         from_to_matrix = pd.read_excel(road_path, index_col=0)
#         # Virtual Stockyard까지의 거리 추가 --> 거리 = 0 (가상의 공간이므로)
#
#     # from_to_matrix = pd.read_excel(path, index_col=0)
#     #
#     # # basic padding -> Source, Sink
#         basic = ["Source", "Sink"]
#         if padding is not None:
#             basic += padding
#
#         # padding: process list that its distance equals 0
#         for padding_process in basic:
#             from_to_matrix[padding_process] = 0.0
#             from_to_matrix.loc[padding_process] = 0.0
#
#         from_to_matrix['선실공장'] = 0.0
#         from_to_matrix.loc['선실공장'] = 0.0
#
#         network[i] = from_to_matrix
#     #
#     # network_dict = dict()
#     # network_dict[12] = from_to_matrix
#
#     return network


'''
Modeling
* monitor -> repetitive action of every simulation => no module 
* Resource -> need modeling just once => no modeling module, but hv resource_information module
- Part
- Process 
- Stock yard 
'''


def modeling_TP(path_tp):
    tp_info = pd.read_excel(path_tp)
    tps = dict()
    tp_minmax = dict()
    for i in range(len(tp_info)):
        temp = tp_info.iloc[i]
        yard = temp['운영구역'][0]
        if yard in ["1", "2"]:
            tp_name = temp['장비번호']
            tp_name = tp_name if tp_name not in tps.keys() else '{0}_{1}'.format(tp_name, 0)
            capacity = temp['최대 적재가능 블록중량(톤)']
            v_loaded = temp['만차 이동 속도(km/h)'] * 24 * 1000
            v_unloaded = temp['공차 이동 속도(km/h)'] * 24 * 1000
            yard = int(yard)
            tps[tp_name] = Transporter(tp_name, yard, capacity, v_loaded, v_unloaded)
            if yard not in tp_minmax.keys():
                tp_minmax[yard] = {"min": 1e8, "max": 0}

            tp_minmax[yard]["min"] = capacity if capacity < tp_minmax[yard]["min"] else tp_minmax[yard]["min"]
            tp_minmax[yard]["max"] = capacity if capacity > tp_minmax[yard]["max"] else tp_minmax[yard]["max"]

    return tps, tp_minmax


def modeling_parts(environment, data, process_dict, monitor_class, resource_class=None, distance_matrix=None,
                   stock_dict=None, inout=None, convert_dict=None, dock_dict=None):
    part_dict = dict()
    blocks = dict()
    for part in data:
        part_data = data[part]
        series = part[:5]
        block = Block(part, part_data['area'], part_data['size'], part_data['weight'])
        blocks[block.name] = block
        part_dict[part] = Part(part, environment, part_data['data'], process_dict, monitor_class,
                               resource=resource_class, network=distance_matrix, block=block, blocks=blocks,
                               child=part_data['child_block'], parent=part_data['parent_block'], stocks=stock_dict,
                               Inout=inout, convert_to_process=convert_dict, dock=dock_dict[series],
                               source_location=part_data['source_location'])

    return part_dict


def modeling_processes(process_dict, stock_dict, process_info, environment, parts, monitor_class, machine_num,
                       resource_class, convert_process):
    # 1. Stockyard
    stockyard_info = process_info['Stockyard']
    for stock in stockyard_info.keys():
        stock_dict[stock] = StockYard(environment, stock, parts, monitor_class,
                                      capacity=stockyard_info[stock]['capacity'], unit=stockyard_info[stock]['unit'])
    # 2. Shelter
    shelter_info = process_info['Shelter']
    for shelter in shelter_info.keys():
        process_dict[shelter] = Process(environment, shelter, machine_num, process_dict, parts, monitor,
                                        resource=resource_class, capacity=shelter_info[shelter]['capacity'],
                                        convert_dict=convert_process, unit=shelter_info[shelter]['unit'],
                                        process_type="Shelter")

    # 3. Painting
    painting_info = process_info['Painting']
    for painting in painting_info.keys():
        process_dict[painting] = Process(environment, painting, machine_num, process_dict, parts, monitor,
                                        resource=resource_class, capacity=painting_info[painting]['capacity'],
                                        convert_dict=convert_process, unit=painting_info[painting]['unit'],
                                        process_type="Painting")

    # 4. Factory
    factory_info = process_info['Factory']
    for factory in factory_info.keys():
        if ('대조립' in factory) and ('1공장' in factory):
            print(0)
        process_dict[factory] = Process(environment, factory, machine_num, process_dict, parts, monitor,
                                        resource=resource_class, capacity=factory_info[factory]['capacity'],
                                        convert_dict=convert_process, unit=factory_info[factory]['unit'],
                                        process_type="Factory")
    # for factory in list_for_process:
    #     if factory in list_for_stock:
    #         stock_dict[factory] = StockYard(environment, factory, part_modeling_data, monitor_class,
    #                                         capacity=area[factory])
    #     elif (area[factory] < 1e8) and ('도장' in factory):
    #         process_dict[factory] = Process(environment, factory, machine_number, process_dict, part_modeling_data,
    #                                         monitor_class, resource=resource_class, convert_dict=convert_dict,
    #                                         area=area[factory], process_type="Painting")
    #     elif (area[factory] < 1e8) and ('도장' not in factory):
    #         process_dict[factory] = Process(environment, factory, machine_number, process_dict, part_modeling_data,
    #                                         monitor_class, resource=resource_class, convert_dict=convert_dict,
    #                                         area=area[factory], process_type="Shelter")
    #     elif factory == "Painting":
    #         process_dict[factory] = Process(environment, factory, machine_number, process_dict, part_modeling_data,
    #                                         monitor_class, resource=resource_class, convert_dict=convert_dict,
    #                                         area=area[factory], process_type="Painting")
    #     elif factory == "Shelter":
    #         process_dict[factory] = Process(environment, factory, machine_number, process_dict, part_modeling_data,
    #                                         monitor_class, resource=resource_class, convert_dict=convert_dict,
    #                                         area=area[factory], process_type="Shelter")
    #     else:
    #         process_dict[factory] = Process(environment, factory, machine_number, process_dict, part_modeling_data,
    #                                         monitor_class, resource=resource_class, convert_dict=convert_dict,
    #                                         area=area[factory])
    #
    process_dict['Sink'] = Sink(environment, process_dict, parts, monitor_class)

    return process_dict, stock_dict


if __name__ == "__main__":
    start = time.time()
    # print(sys.argv[1])
    # 1. read input data
    with open('./result/Series20/input_data.json', 'r') as f:
        input_data = json.load(f)
    # with open(sys.argv[1], 'r') as f:
    #     input_data = json.load(f)

    process_info, inout = read_process_info(input_data['path_process_info'])
    # if need to preprocess with activity and bom
    if input_data['use_prior_process']:
        with open(input_data['path_preprocess'], 'r') as f:
            sim_data = json.load(f)
        print("Finish data loading at ", time.time() - start)
    else:
        print("Start combining Activity and BOM data at ", time.time() - start)
        data_path = processing_with_activity_N_bom(input_data['path_activity_data'], input_data['path_bom_data'],
                                                   input_data['path_dock_series_data'],
                                                   input_data['path_converting_data'], input_data['series_to_preproc'],
                                                   input_data['default_result'])

        print("Finish data preprocessing at ", time.time() - start)

        with open(data_path, 'r') as f:
            sim_data = json.load(f)
        print("Finish data loading at ", time.time() - start)

    initial_date = sim_data['initial_date']
    block_data = sim_data['block_info']

    # read entrance and exit
    # inout_dict = read_inout(input_data['path_inout_data'], virtual_list=virtual)
    # process_list = list(inout_dict.keys())

    # set area of process
    # process_area, stock_list = set_area(input_data['process_area'], process_list,
    #                                     path_stock_area=input_data['path_stock_area'],
    #                                     path_process_area=input_data['path_process_area'])

    converting = read_converting(input_data['path_converting_data'])
    dock = read_dock_series(input_data['path_dock_series_data'])

    # network = read_network(input_data['path_road_data'], padding=virtual)
    ## network
    with open(input_data['path_distance'], 'r') as f:
        network_distance = json.load(f)


    # define simulation environment
    env = simpy.Environment()

    # Modeling
    stock_yard = dict()
    processes = dict()

    # 0. Monitor
    monitor = Monitor(input_data['default_result'], input_data['project_name'], network_distance)

    # 1. Resource
    tps, tp_minmax = modeling_TP(input_data['path_transporter'])
    resource = Resource(env, processes, stock_yard, monitor, tps=tps, tp_minmax=tp_minmax, network=network_distance,
                        inout=inout)

    # 2. Block
    parts = modeling_parts(env, block_data, processes, monitor, resource_class=resource, distance_matrix=network_distance,
                           stock_dict=stock_yard, inout=inout, convert_dict=converting, dock_dict=dock)

    # 3. Process and StockYard
    processes, stock_yard = modeling_processes(processes, stock_yard, process_info, env, parts, monitor,
                                               input_data['machine_num'], resource, converting)

    start_sim = time.time()
    env.run()
    finish_sim = time.time()

    print("Execution time:", finish_sim - start_sim)
    # path_event_tracer, path_tp_info, path_road_info = monitor.save_information()
    path_event_tracer = monitor.save_information()
    print("number of part created = ", monitor.created)
    print("number of completed = ", monitor.completed)

    output_path = dict()
    output_path['input_path'] = './result/Series2_TP/input_data_tp.json'
    output_path['event_tracer'] = path_event_tracer
    # output_path['tp_info'] = path_tp_info
    # output_path['road_info'] = path_road_info

    with open(input_data['default_result'] + 'result_path.json', 'w') as f:
        json.dump(output_path, f)
    print("Finish")