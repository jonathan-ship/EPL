import json
import pandas as pd
import numpy as np

'''
Read and Convert from excel file to json file 
1. Distance File : Convert & Save and Return to Main.py
2. Object Id File : Convert & Save 
'''


def convert_to_json_road(path_distance, path_objectid, data_path):
    padding = ['Stock', 'Shelter', 'Painting']

    from_to_matrix_distance = pd.read_excel(path_distance, index_col=0)
    from_to_matrix_edge = pd.read_csv(path_objectid, index_col=0)
    #
    # # basic padding -> Source, Sink
    basic = ["Source", "Sink"]
    if padding is not None:
        basic += padding

    # padding: process list that its distance equals 0
    for padding_process in basic:
        from_to_matrix_distance[padding_process] = 0.0
        from_to_matrix_distance.loc[padding_process] = 0.0

        from_to_matrix_edge[padding_process] = "[]"
        from_to_matrix_edge.loc[padding_process] = "[]"

    from_to_matrix_distance = from_to_matrix_distance.replace({np.nan: None})

    network_distance = dict()
    network_edge = dict()
    from_list = list(from_to_matrix_distance.index)
    for from_idx in from_list:
        network_distance[from_idx] = dict()
        network_edge[from_idx] = dict()
        temp = from_to_matrix_distance.loc[from_idx]
        to_list = list(temp.index)
        for to_idx in to_list:
            # distance
            network_distance[from_idx][to_idx] = from_to_matrix_distance[to_idx][from_idx]

            # object id
            network_edge[from_idx][to_idx] = eval(''.join(from_to_matrix_edge[to_idx][from_idx]))

    with open(data_path + 'network_distance.json', 'w') as f:
        json.dump(network_distance, f)
    with open(data_path + 'network_edge.json', 'w') as f:
        json.dump(network_edge, f)

    return network_distance