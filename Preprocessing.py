import json, copy
import pandas as pd
import numpy as np

SERIES = []
ONLY_BLOCK = []
BLOCK = []
PROCESS_CODE = []
START_DATE = []
FINISH_DATE = []
LOCATION = []
SIZE = []
AREA = []


# 블록의 크기, 면적 계산
def determine_size(data):
    L = data['길이']
    B = data['폭']
    H = data['높이']
    size_list = [L, B, H]
    if L * B * H == 0:
        size_list.remove(0)
    size = min(size_list) if len(size_list) > 0 else 0.0

    return size


# Activity 전처리 결과 dataframe
def append_list(block_code, process_code, start_date, finish_date, location, size, area):
    SERIES.append(block_code[:5])
    ONLY_BLOCK.append(block_code[-5:])
    BLOCK.append(block_code)
    PROCESS_CODE.append(process_code)
    START_DATE.append(start_date)
    FINISH_DATE.append(finish_date)
    LOCATION.append(location)
    SIZE.append(size)
    AREA.append(area)


# 부서 - 도크 mapping
def convert_process(present_process, block_code, converting, dock_mapping):
    # 현재 Step
    series = block_code[:5]
    process_convert_by_dict = converting[present_process] if present_process != 'Sink' else 'Sink'

    # 1:1 대응
    if type(process_convert_by_dict) == str:
        return process_convert_by_dict
    elif present_process == '건조1부' or present_process == '건조3부' or present_process == '건조2부':
        if series[3] == 0:  # 한 자리수 호선
            dock_num = dock_mapping[int(series[4]) - 1]['도크']
            dock = '{0}도크'.format(dock_num)
        else:  # 두 자리수 호선
            dock_num = dock_mapping[int(series[-2:]) - 1]['도크']
            dock = '{0}도크'.format(dock_num)
        return dock
    else:
        return present_process


'''
INPUT DATA 
- Activity data 
- BOM data 
- mapping data 
   > 부서 - process mapping (.json)
   > 건조부서 - dock mapping (호선도크.xlsx)
   > GIS mapping (process_gis_mapping_table.xlsx) 
'''


def processing_with_activity_N_bom(path_activity: str, path_bom: str, path_dock: str, path_converting: str, series: list, saving_path: str) -> object:
    activity_data_all = pd.read_excel(path_activity, engine='openpyxl')

    bom_data_all = pd.read_excel(path_bom, engine='openpyxl')

    dock_mapping = pd.read_excel(path_dock)
    dock_mapping = dict(dock_mapping.transpose())

    with open(path_converting, 'r') as f:
        converting = json.load(f)

    # network = {}
    #
    # ## from to matrix를 network(json)으로부터 가져오는 것부터
    # from_to_matrix = pd.read_excel(path_road, index_col=0)
    # # Virtual Stockyard까지의 거리 추가 --> 거리 = 0 (가상의 공간이므로)
    # from_to_matrix.loc['Virtual'] = 0.0
    # from_to_matrix.loc['Source'] = 0.0
    # from_to_matrix.loc['Sink'] = 0.0
    # from_to_matrix['Virtual'] = 0.0
    # from_to_matrix['Source'] = 0.0
    # from_to_matrix['Sink'] = 0.0
    #
    # network[12] = from_to_matrix

    if series == "all":
        target_series = list(np.unique(list(activity_data_all['호선'])))
    else:
        target_series = ['A000{0}'.format(series_num) if series_num < 10 else 'A00{0}'.format(series_num) for series_num in series]

    # filtering positive value at start_date and finish_date
    activity_data_all = activity_data_all[(activity_data_all['시작일'] > 0) & (activity_data_all['종료일'] > 0)]
    activity_data_all.loc[:, '시작일'] = pd.to_datetime(activity_data_all['시작일'], format='%Y%m%d')
    activity_data_all.loc[:, '종료일'] = pd.to_datetime(activity_data_all['종료일'], format='%Y%m%d')
    # find initial date
    initial_date = None
    for idx in range(len(target_series)):
        temp_series = activity_data_all[activity_data_all['호선'] == target_series[idx]]
        if idx == 0:
            initial_date = temp_series['시작일'].min()
        else:
            initial_date = temp_series['시작일'].min() if min(temp_series['시작일']) < initial_date else initial_date


    # *2. find initial date of start date
    preproc = dict()
    preproc['initial_date'] = initial_date.strftime('%Y-%m-%d')
    block_info = {}
    for series in target_series:
        activity_data = activity_data_all[activity_data_all['호선'] == series]
        bom_data = bom_data_all[bom_data_all['호선'] == series]

        series_area = []
        series_size = []
        print("start ", series)

        # filtering unused work
        activity_data = activity_data[(activity_data_all['공정공종'] != 'C91') & (activity_data_all['공정공종'] != 'CX3') & \
                                      (activity_data_all['공정공종'] != 'F91') & (activity_data_all['공정공종'] != 'FX3') & \
                                      (activity_data_all['공정공종'] != 'G4A') & (activity_data_all['공정공종'] != 'G4B') & \
                                      (activity_data_all['공정공종'] != 'GX3') & (activity_data_all['공정공종'] != 'HX3') & \
                                      (activity_data_all['공정공종'] != 'K4A') & (activity_data_all['공정공종'] != 'K4B') & \
                                      (activity_data_all['공정공종'] != 'L4B') & (activity_data_all['공정공종'] != 'L4A') & \
                                      (activity_data_all['공정공종'] != 'LX3') & (activity_data_all['공정공종'] != 'JX3')]



        print("공정 골라내기 완료")

        # making the processing position the same with "외부"
        out_of_the_yard = ['KINGS QUAY 공사부', '해양외업생산부', '해양사업부', '포항공장부', '특수선', '용연공장부']
        activity_data['작업부서'] = activity_data['작업부서'].apply(lambda x: x.replace(x, '외부') if x in out_of_the_yard else x)

        # reforming date to integer


        # *3. reform datetime to integer by subtracting
        activity_data.loc[:, '시작일'] = activity_data['시작일'].apply(lambda x: (x - initial_date).days)
        activity_data.loc[:, '종료일'] = activity_data['종료일'].apply(lambda x: (x - initial_date).days)
        activity_data = activity_data.sort_values(by=['시작일'], ascending=True)
        activity_data = activity_data.reset_index(drop=True)
        # making columns need to processing
        activity_data.loc[:, '호선'] = activity_data['호선'].apply(lambda x: str(x))
        activity_data.loc[:, '블록'] = activity_data['ACT_ID'].apply(lambda x: x[:5])
        activity_data.loc[:, 'block code'] = activity_data['호선'] + '_' + activity_data['블록']
        # activity_data.loc[:, 'process_head'] = activity_data['공정공종'].apply(
        #    lambda x: x[0] if type(x) == str else None)  # 공정공종 첫 알파벳
        activity_data.loc[:, 'detail_process'] = activity_data['호선'] + activity_data['ACT_ID'].apply(
            lambda x: x[:8] if type(x) == str else None)

        bom_data.loc[:, '상위블록'] = bom_data['상위블록'].apply(lambda x: str(x))
        bom_data.loc[:, 'child code'] = bom_data['호선'] + '_' + bom_data['블록']
        bom_data.loc[:, 'parent code'] = bom_data['호선'] + '_' + bom_data['상위블록']

        # removing detail process
        detail_list = list(activity_data.drop_duplicates(['detail_process'])['detail_process'])
        for detail in detail_list:
            temp = activity_data[activity_data['detail_process'] == detail]
            if len(temp) > 1:
                req_Index = temp[(temp['작업부서'] == '외부') | (temp['작업부서'] == None)].index.tolist()
                if len(req_Index) == len(temp):
                    req_Index = req_Index[1:]
                activity_data = activity_data.drop(req_Index)

        # grouping block and bom data by block code
        block_list = list(np.unique(list(activity_data['block code'])))
        bom_child = list(np.unique(list(bom_data['child code'])))
        bom_parent = list(np.unique(list(bom_data['parent code'])))

        block_group = activity_data.groupby(activity_data['block code'])
        bom_group_by_child = bom_data.groupby(bom_data['child code'])
        bom_group_by_parent = bom_data.groupby(bom_data['parent code'])

        # Calculating block area and size
        bom_data['size'] = bom_data.apply(lambda x: determine_size(x), 1)
        bom_data['area'] = bom_data.apply(lambda x: x['길이'] * x['폭'], 1)

        size_remove_zero = list(bom_data['size'][bom_data['size'] > 0.0])
        area_remove_zero = list(bom_data['area'][bom_data['area'] > 0.0])
        weight_remove_zero = list(bom_data['중량'][bom_data['중량'] > 0.0])

        size_avg = np.mean(size_remove_zero)
        area_avg = np.mean(area_remove_zero)
        weight_avg = np.mean(weight_remove_zero)

        bom_data['size'] = bom_data['size'].replace(0.0, size_avg)
        bom_data['area'] = bom_data['area'].replace(0.0, area_avg)
        bom_data['중량'] = bom_data['중량'].replace(0.0, weight_avg)

        # recording block information into 'block_info' dictionary
        block_list_for_source = copy.deepcopy(block_list)
        for block_code in block_list:
            if (block_code in bom_child) or (block_code in bom_parent):
                block_info[block_code] = {}
                block_data = block_group.get_group(block_code)
                block_data = block_data.sort_values(by=['시작일'], ascending=True)
                block_data = block_data.reset_index(drop=True)

                # 1. activity processing
                previous_activity = block_data.iloc[0]
                previous_dict = dict(previous_activity)
                start_date = previous_dict['시작일']
                finish_date = previous_dict['종료일']
                work_station = previous_dict['작업부서']

                process_data = [None for _ in range(32)]
                idx = 0
                if len(block_data) == 1:
                    process_data[4 * idx] = float(previous_dict['시작일'])
                    process_time = previous_dict['종료일'] - previous_dict['시작일']
                    process_data[4 * idx + 1] = float(process_time) if process_time > 0 else 0.0
                    process_data[4 * idx + 2] = convert_process(previous_dict['작업부서'], block_code,
                                                                converting=converting, dock_mapping=dock_mapping)
                    process_data[4 * idx + 3] = previous_dict['공정공종']
                    idx += 1
                else:
                    for j in range(1, len(block_data)):
                        post_activity = block_data.iloc[j]
                        post_dict = dict(post_activity)
                        if (post_dict['시작일'] >= start_date) and (post_dict['종료일'] <= finish_date):  # 후행이 선행에 포함
                            previous_dict['공정공종'] = previous_dict['공정공종'] + post_dict['공정공종'][0]
                            post_dict = copy.deepcopy(previous_dict)

                        # 처음이 블록 이동이 없는 데 속하고, 이전 공정이라는 게 존재하지 않을 경우
                        if (j == 1) and (work_station in ['도장1부', '도장2부', '발판지원부']):
                            previous_dict['작업부서'] = post_dict['작업부서']
                        elif (post_dict['작업부서'] in ['도장1부', '도장2부', '발판지원부']) and \
                                (previous_dict['작업부서'] not in ['도장1부', '도장2부', '발판지원부']):
                            post_dict['작업부서'] = previous_dict['작업부서']
                        elif (post_dict['작업부서'] in ['도장1부', '도장2부', '발판지원부']) and \
                                (previous_dict['작업부서'] in ['도장1부', '도장2부', '발판지원부']):
                            post_dict['작업부서'] = work_station

                        if previous_dict['작업부서'] != post_dict['작업부서']:
                            '''append_list(block_code, previous_activity['공정공종'], previous_activity['시작일'],
                                        previous_activity['종료일'],
                                        previous_activity['작업부서'], size, area)'''
                            process_data[4 * idx] = float(previous_dict['시작일'])
                            process_time = previous_dict['종료일'] - previous_dict['시작일']
                            process_data[4 * idx + 1] = float(process_time) if process_time > 0 else 0.0
                            process_data[4 * idx + 2] = convert_process(previous_dict['작업부서'], block_code,
                                                                        converting=converting, dock_mapping=dock_mapping)
                            process_data[4 * idx + 3] = previous_dict['공정공종']
                            idx += 1

                            finish_date = previous_dict['종료일']
                            work_station = previous_dict['작업부서']
                            previous_dict = copy.deepcopy(post_dict)

                        else:
                            previous_dict['종료일'] = post_dict['종료일'] if post_dict['종료일'] > previous_dict['종료일'] else \
                            previous_dict['종료일']
                            previous_dict['공정공종'] = previous_dict['공정공종'][0] + post_dict['공정공종']
                            if previous_dict['시작일'] < finish_date:
                                previous_dict['시작일'] = finish_date + 1
                            previous_activity = pd.Series(previous_dict)
                            work_station = previous_dict['작업부서']

                        if j == len(block_data) - 1:  # 마지막이면
                            '''append_list(block_code, previous_activity['공정공종'], previous_activity['시작일'],
                                        previous_activity['종료일'],
                                        previous_activity['작업부서'], size, area)'''
                            process_data[4 * idx] = float(previous_dict['시작일'])
                            process_time = previous_dict['종료일'] - previous_dict['시작일']
                            process_data[4 * idx + 1] = float(process_time) if process_time > 0 else 0.0
                            process_data[4 * idx + 2] = convert_process(previous_dict['작업부서'], block_code,
                                                                        converting=converting, dock_mapping=dock_mapping)
                            process_data[4 * idx + 3] = previous_dict['공정공종']
                            idx += 1

                process_data[4 * idx + 2] = 'Sink'

                block_info[block_code]['data'] = process_data

                # 2. processing with bom data
                # 2-1. grouping by child code
                if block_code in bom_child:
                    bom_child_data = bom_group_by_child.get_group(block_code)
                    bom_child_data = bom_child_data.reset_index(drop=True)

                    block_info[block_code]['size'] = float(bom_child_data['size'][0])
                    block_info[block_code]['area'] = float(bom_child_data['area'][0])
                    block_info[block_code]['weight'] = float(bom_child_data['중량'][0])

                    parent_block = bom_child_data['parent code'][0]
                    if parent_block in block_list:
                        block_info[block_code]['parent_block'] = parent_block
                    else:
                        block_info[block_code]['parent_block'] = None
                else:
                    block_info[block_code]['parent_block'] = None

                block_info[block_code]['source_location'] = None
                # 2-2. grouping by parent code
                if block_code in bom_parent:
                    bom_parent_data = bom_group_by_parent.get_group(block_code)
                    bom_parent_data = bom_parent_data.reset_index(drop=True)

                    child_list = []
                    # child_last_process_weight = {}
                    for i in range(len(bom_parent_data)):
                        child = bom_parent_data['child code'][i]
                        if child in block_list:
                            child_list.append(child)
                            # last_process = convert_process(list(block_group.get_group(child)['작업부서'])[-1], child,
                            #                                converting=converting, dock_mapping=dock_mapping)
                            # if last_process not in ['선행도장부', '선행의장부', '기장부', '의장1부', '의장2부', '의장3부',
                            #                         '도장1부', '도장2부', '발판지원부']:
                            #     child_weight = list(bom_group_by_child.get_group(child)['중량'])[-1]
                            #     child_size = list(bom_group_by_child.get_group(child)['size'])[-1]
                            #     child_last_process_weight.append([last_process, child_weight, child_size])

                    if len(child_list) > 0:
                        block_info[block_code]['child_block'] = child_list
                    else:
                        block_info[block_code]['child_block'] = None
                        block_list_for_source.remove(block_code)

                    # if len(child_last_process_weight) > 0:
                    #     child_last_process_weight = sorted(child_last_process_weight, key=lambda x: (x[1], x[2]))
                    #     block_info[block_code]['source_location'] = child_last_process_weight[0][0]
                    # if len(child_list) > 0 and len(child_last_process_weight) == 0:
                    #     print(block_code)
                    #     print(0)
                else:
                    block_info[block_code]['child_block'] = None
            else:
                continue
        # determine source location
        for parent_code in block_info.keys():
            if parent_code[:5] == series:
                child_list = block_info[parent_code]['child_block']
                parent_block_grp = block_group.get_group(parent_code)
                parent_first_process = list(parent_block_grp['작업부서'])[0]
                if (child_list is not None) and (parent_first_process in ['선행도장부', '선행의장부', '기장부', '의장1부', '의장2부', '의장3부', '도장1부', '도장2부', '발판지원부']):
                    child_process_weight_dict = dict()
                    for child_code in child_list:
                        child_process = list(block_group.get_group(child_code)['작업부서'])
                        child_weight = list(bom_group_by_child.get_group(child_code)['중량'])[-1]
                        for i in range(len(child_process)):
                            process = child_process[-(i+1)]
                            if process not in ['선행도장부', '선행의장부', '기장부', '의장1부', '의장2부', '의장3부', '도장1부',
                                               '도장2부', '발판지원부']:
                                if i not in child_process_weight_dict.keys():
                                    child_process_weight_dict[i] = list()
                                converted_process = convert_process(process, child_code, converting=converting,
                                                                    dock_mapping=dock_mapping)
                                child_process_weight_dict[i].append([converted_process, child_weight])

                    # determine its source location
                    if len(child_process_weight_dict) == 0:
                        print(parent_code)  # 이럴 경우 하위의 하위까지 내려가서 찾는 거 만들기
                        grandchild_list = list()
                        for child in child_list:
                            grandchild_list += block_info[child]['child_block']
                        for grandchild in grandchild_list:
                            grandchild_process = list(block_group.get_group(grandchild)['작업부서'])
                            grandchild_weight = list(bom_group_by_child.get_group(grandchild)['중량'])[-1]
                            for i in range(len(grandchild_process)):
                                process = grandchild_process[-(i + 1)]
                                if process not in ['선행도장부', '선행의장부', '기장부', '의장1부', '의장2부', '의장3부', '도장1부',
                                                   '도장2부', '발판지원부']:
                                    if i not in child_process_weight_dict.keys():
                                        child_process_weight_dict[i] = list()
                                    converted_process = convert_process(process, grandchild, converting=converting,
                                                                        dock_mapping=dock_mapping)
                                    child_process_weight_dict[i].append([converted_process, grandchild_weight])

                    if len(child_process_weight_dict) == 0:
                        print(0)

                    min_idx = min(child_process_weight_dict.keys())
                    location_list = child_process_weight_dict[min_idx]
                    location_list = sorted(location_list, key=lambda x: x[1], reverse=True)
                    block_info[parent_code]['source_location'] = location_list[0][0]

    preproc['block_info'] = block_info
    # Save data
    with open(saving_path + 'Layout_data.json', 'w') as f:
        json.dump(preproc, f)

    return saving_path + 'Layout_data.json'
