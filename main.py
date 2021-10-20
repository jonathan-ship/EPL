import simpy, time, sys, json
import pandas as pd

from Input import input_main
from Preprocessing import processing_with_activity_N_bom
from Sim_Kernel import Resource, Part, Sink, StockYard, Monitor, Process
from Post import post_main


def set_virtual(virtual_stock=False, virtual_shelter=False, virtual_painting=False):
    if virtual_stock or virtual_shelter or virtual_painting:
        virtual_list = list()
        if virtual_stock:
            virtual_list.append("Stock")
        if virtual_shelter:
            virtual_list.append("Shelter")
        if virtual_painting:
            virtual_list.append("Painting")

        return virtual_list
    else:
        return None


def read_inout(path, virtual_list=None):
    inout_table = pd.read_excel(path)
    process_inout = {}
    for i in range(len(inout_table)):
        temp = inout_table.iloc[i]
        process_inout[temp['LOC']] = [temp['IN'], temp['OUT']]

    if virtual_list is not None:
        for virtual in virtual_list:
            process_inout[virtual] = [virtual, virtual]

    return process_inout


def set_area(default_area, process_list, path_stock_area=None, path_process_area=None):
    area = dict()
    stock_list = None

    if path_stock_area is not None:
        stock_area_data = pd.read_excel(path_stock_area)
        stock_list = list(stock_area_data['name'])
        for i in range(len(stock_list)):
            area[stock_area_data['name'][i]] = stock_area_data['area'][i]
        stock_list.append("Stock")

    if path_process_area is not None:
        process_area_data = pd.read_excel(path_process_area)
        proc_list = list(process_area_data['name'])
        for i in range(len(proc_list)):
            area[process_area_data['name'][i]] = process_area_data['area'][i]

    for process in process_list:
        if process not in area.keys():
            area[process] = default_area

    return area, stock_list


def read_converting(path):
    with open(path, 'r') as f:
        converting_dict = json.load(f)

    return converting_dict


def read_dock_series(path: object) -> object:
    data = pd.read_excel(path)
    dock_series_mapping = {data.iloc[i]['호선']: data.iloc[i]['도크'] for i in range(len(data))}

    return dock_series_mapping


def read_network(path, padding=None):
    from_to_matrix = pd.read_excel(path, index_col=0)

    # basic padding -> Source, Sink
    basic = ["Source", "Sink"]
    if padding is not None:
        basic += padding

    # padding: process list that its distance equals 0
    for padding_process in basic:
        from_to_matrix[padding_process] = 0.0
        from_to_matrix.loc[padding_process] = 0.0

    network_dict = dict()
    network_dict[12] = from_to_matrix

    return network_dict


'''
Modeling
* monitor -> repetitive action of every simulation => no module 
* Resource -> need modeling just once => no modeling module, but hv resource_information module
- Part
- Process 
- Stock yard 
'''


def resource_information(type, tp_numbers=None, tp_loaded=None, tp_unloaded=None, wf_numbers=None, wf_skills=None):
    info = {}
    if type == 'Transporter':
        for i in range(tp_numbers):
            info["TP_{0}".format(i + 1)] = {"v_loaded": tp_loaded, "v_unloaded": tp_unloaded}
    return info


def modeling_parts(environment, data, process_dict, monitor_class, resource_class=None, distance_matrix=None,
                   stock_dict=None, inout=None, convert_dict=None, dock_dict=None):
    part_dict = dict()
    for part in data:
        part_data = data[part]
        series = part[:5]
        part_dict[part] = Part(part, environment, part_data['data'], process_dict, monitor_class,
                               resource=resource_class, from_to_matrix=distance_matrix, size=part_data['size'],
                               area=part_data['area'], child=part_data['child_block'], parent=part_data['parent_block'],
                               stocks=stock_dict, Inout=inout, convert_to_process=convert_dict, dock=dock_dict[series])

    return part_dict


def modeling_processes(process_dict, stock_dict, list_for_process, list_for_stock, environment, machine_number,
                       part_modeling_data, monitor_class, resource_class=None, convert_dict=None, area=None):
    for factory in list_for_process:
        if factory in list_for_stock:
            stock_dict[factory] = StockYard(environment, factory, part_modeling_data, monitor_class,
                                            capacity=area[factory])
        elif (area[factory] < 1e8) and ('도장' in factory):
            process_dict[factory] = Process(environment, factory, machine_number, process_dict, part_modeling_data,
                                            monitor_class, resource=resource_class, convert_dict=convert_dict,
                                            area=area[factory], process_type="Painting")
        elif (area[factory] < 1e8) and ('도장' not in factory):
            process_dict[factory] = Process(environment, factory, machine_number, process_dict, part_modeling_data,
                                            monitor_class, resource=resource_class, convert_dict=convert_dict,
                                            area=area[factory], process_type="Shelter")
        elif factory == "Painting":
            process_dict[factory] = Process(environment, factory, machine_number, process_dict, part_modeling_data,
                                            monitor_class, resource=resource_class, convert_dict=convert_dict,
                                            area=area[factory], process_type="Painting")
        elif factory == "Shelter":
            process_dict[factory] = Process(environment, factory, machine_number, process_dict, part_modeling_data,
                                            monitor_class, resource=resource_class, convert_dict=convert_dict,
                                            area=area[factory], process_type="Shelter")
        else:
            process_dict[factory] = Process(environment, factory, machine_number, process_dict, part_modeling_data,
                                            monitor_class, resource=resource_class, convert_dict=convert_dict,
                                            area=area[factory])

    process_dict['Sink'] = Sink(environment, process_dict, part_modeling_data, monitor_class)

    return process_dict, stock_dict


if __name__ == "__main__":
    start = time.time()
    # print(sys.argv[1])
    # 1. read input data
    # with open('C:/Users/sohyon/PycharmProjects/Simulation_Module/data/input_data.json', 'r') as f:
    #     input_data = json.load(f)
    # with open(sys.argv[1], 'r') as f:
    #     input_data = json.load(f)
    start_time = time.time()
    path_input_file = input_main()
    with open(path_input_file, 'r') as f:
        input_data = json.load(f)

    # if need to preprocess with activity and bom
    if input_data['use_prior_process']:
        with open(input_data['path_preprocess'], 'r') as f:
            sim_data = json.load(f)
        print("Finish data loading at ", time.time() - start)
        preprocessing_finish_time = time.time()
    else:
        print("Start combining Activity and BOM data at ", time.time() - start)
        data_path = processing_with_activity_N_bom(input_data['path_activity_data'], input_data['path_bom_data'],
                                                   input_data['path_dock_series_data'], input_data['path_inout_data'],
                                                   input_data['path_converting_data'], input_data['path_road_data'],
                                                   input_data['series_to_preproc'], input_data['default_result'])
        preprocessing_finish_time = time.time()
        print("Finish data preprocessing at ", time.time() - start)

        with open(data_path, 'r') as f:
            sim_data = json.load(f)
        print("Finish data loading at ", time.time() - start)

    # set virtual process
    virtual = set_virtual(virtual_stock=input_data['stock_virtual'],
                          virtual_shelter=input_data['shelter_virtual'],
                          virtual_painting=input_data['painting_virtual'])

    # read entrance and exit
    inout_dict = read_inout(input_data['path_inout_data'], virtual_list=virtual)
    process_list = list(inout_dict.keys())

    # set area of process
    process_area, stock_list = set_area(input_data['process_area'], process_list,
                                        path_stock_area=input_data['path_stock_area'],
                                        path_process_area=input_data['path_process_area'])

    converting = read_converting(input_data['path_converting_data'])
    dock = read_dock_series(input_data['path_dock_series_data'])
    network = read_network(input_data['path_road_data'], padding=virtual)

    # define simulation environment
    env = simpy.Environment()

    # Modeling
    stock_yard = dict()
    processes = dict()

    # 0. Monitor
    monitor = Monitor(input_data['default_result'], input_data['project_name'], network)

    # 1. Resource
    resource_info = resource_information("Transporter", input_data['tp_num'], input_data['tp_v_loaded'],
                                         input_data['tp_v_unloaded'])
    resource = Resource(env, processes, stock_yard, monitor, tp_info=resource_info, network=network, inout=inout_dict)

    # 2. Block
    parts = modeling_parts(env, sim_data, processes, monitor, resource_class=resource, distance_matrix=network,
                           stock_dict=stock_yard, inout=inout_dict, convert_dict=converting, dock_dict=dock)

    # 3. Process and StockYard
    processes, stock_yard = modeling_processes(processes, stock_yard, process_list, stock_list, env,
                                               input_data['machine_num'], parts, monitor, resource_class=resource,
                                               convert_dict=converting, area=process_area)

    start_sim = time.time()
    env.run()
    finish_sim = time.time()
    print('#' * 80)
    print("Data Preprocessing time:", preprocessing_finish_time - start_time)
    print("Execution time:", finish_sim - start_sim)
    print("Total time:", finish_sim - start_time)
    path_event_tracer, path_tp_info, path_road_info = monitor.save_information()
    print('#' * 80)
    print("number of part created = ", monitor.created)
    print("number of completed = ", monitor.completed)

    output_path = dict()
    output_path['input_path'] = path_input_file
    output_path['event_tracer'] = path_event_tracer
    output_path['tp_info'] = path_tp_info
    output_path['road_info'] = path_road_info

    result_path = input_data['default_result'] + 'result_path.json'
    with open(result_path, 'w') as f:
        json.dump(output_path, f)
    print("Simulation_Finish")



    # get_result = Get_result(path_event_tracer, path_tp_info, input_data['default_result'], './data/input_data.json')
    # get_result.tp_index()
    # get_result.calculate_stock_occupied_area()
    # get_result.calculate_shelter_occupied_area()
    # get_result.calculate_painting_occupied_area()
    # print(0)

    
