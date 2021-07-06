import pandas as pd
import numpy as np
from datetime import datetime
import time
import random
import math

SERIES = []
ONLY_BLOCK = []
BLOCK = []
PROCESS_CODE = []
START_DATE = []
FINISH_DATE = []
LOCATION = []
SIZE = []
AREA = []


def determine_size_area(data):
    L = data['길이']
    B = data['폭']
    size = min(L, B) if L * B > 0 else max(L, B)
    area = L * B

    return size, area


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


start = time.time()
series_list = []
new_activity = pd.DataFrame(
    columns=['series_block_code', 'series', 'block_code', 'process_code', 'start_date', 'finish_date', 'location',
             'loc_indicator'])

for i in range(1, 84):
    if i < 10:
        series_list.append("A000{0}".format(i))
    else:
        series_list.append("A00{0}".format(i))
activity_data_all = pd.read_excel('./data/Layout_Activity_sample.xlsx', engine='openpyxl')
bom_data_all = pd.read_excel('./data/Layout_BOM_sample.xlsx', engine='openpyxl')

for series_num in series_list:
    activity_data = activity_data_all[activity_data_all['호선'] == series_num]
    bom_data = bom_data_all[bom_data_all['호선'] == series_num]
    print("start series ", series_num)
    # BOM data에 있는 블록만 골라내기
    # 필요없는 공정 골라내기

    activity_data = activity_data[(activity_data_all['공정공종'] != 'C91') & (activity_data_all['공정공종'] != 'CX3') & \
                                (activity_data_all['공정공종'] != 'F91') & (activity_data_all['공정공종'] != 'FX3') & \
                                (activity_data_all['공정공종'] != 'G4A') & (activity_data_all['공정공종'] != 'G4B') & \
                                (activity_data_all['공정공종'] != 'GX3') & (activity_data_all['공정공종'] != 'HX3') & \
                                (activity_data_all['공정공종'] != 'K4A') & (activity_data_all['공정공종'] != 'K4B') & \
                                (activity_data_all['공정공종'] != 'L4B') & (activity_data_all['공정공종'] != 'L4A') & \
                                (activity_data_all['공정공종'] != 'LX3') & (activity_data_all['공정공종'] != 'JX3')]
    activity_data = activity_data[(activity_data['시작일'] > 0) & (activity_data['종료일'] > 0)]

    print("공정 골라내기 완료")

    # 외부물류 -> 작업부서 '외부'로 통일
    activity_data['작업부서'][(activity_data['작업부서'] == 'KINGS QUAY 공사부') | (activity_data['작업부서'] == '해양외업생산부') | (activity_data['작업부서'] == '해양사업부') | (activity_data['작업부서'] == '포항공장부') | (activity_data['작업부서'] == '특수선') | (activity_data['작업부서'] == '용연공장부')] = '외부'

    activity_data.loc[:, '시작일'] = pd.to_datetime(activity_data['시작일'], format='%Y%m%d')
    activity_data.loc[:, '종료일'] = pd.to_datetime(activity_data['종료일'], format='%Y%m%d')
    initial_date = activity_data['시작일'].min()

    activity_data.loc[:, '시작일'] = activity_data['시작일'].apply(lambda x: (x - initial_date).days)
    activity_data.loc[:, '종료일'] = activity_data['종료일'].apply(lambda x: (x - initial_date).days)
    print("timestamp으로 바꾸기 완료")

    activity_data.loc[:, '호선'] = activity_data['호선'].apply(lambda x: str(x))
    activity_data.loc[:, '블록'] = activity_data['ACT_ID'].apply(lambda x: x[:5])
    activity_data.loc[:, 'block code'] = activity_data['호선'] + '_' + activity_data['블록']
    activity_data.loc[:, 'process_head'] = activity_data['공정공종'].apply(lambda x: x[0] if type(x) == str else None)  # 공정공종 첫 알파벳
    activity_data.loc[:, 'detail_process'] = activity_data['호선'] + activity_data['ACT_ID'].apply(lambda x: x[:8] if type(x) == str else None)

    print("블록 코드 구현 완료")

    detail_list = list(activity_data.drop_duplicates(['detail_process'])['detail_process'])
    for detail in detail_list:
        temp = activity_data[activity_data['detail_process'] == detail]
        if len(temp) > 1:
            req_Index = temp[(temp['작업부서'] == '외부') | (temp['작업부서'] == None)].index.tolist()
            if len(req_Index) == len(temp):
                req_Index = req_Index[1:]
            activity_data = activity_data.drop(req_Index)
    block_list = list(activity_data.drop_duplicates(['block code'])['block code'])
    print('세부공종 제거')
    activity_data.to_excel('./data/temp.xlsx')
    block_group = activity_data.groupby(activity_data['block code'])  ## block code에 따른 grouping

    print('전처리 at', time.time() - start)

    for block_code in block_list:
        child_list = list(bom_data.drop_duplicates(['블록'])['블록'])
        parent_list = list(bom_data.drop_duplicates(['상위블록'])['상위블록'])
        block_data = block_group.get_group(block_code)  # block code 별로 grouping 한 결과

        if (block_data.iloc[0]['블록'] in child_list) or (block_data.iloc[0]['블록'] in parent_list):
            size, area = determine_size_area(bom_data[bom_data['블록'] == block_data.iloc[0]['블록']].iloc[0])
            heads = list(block_data.drop_duplicates(['process_head'])['process_head'])
            if len(block_data):
                block_data = block_data.sort_values(by=['시작일'], ascending=True)
                block_data = block_data.reset_index(drop=True)
                previous_activity = block_data.iloc[0]
                finish_date = 0
                for j in range(1, len(block_data)):
                    post_activity = block_data.iloc[j]
                    if (post_activity['작업부서'] in ['도장1부', '도장2부', '발판지원부']) and \
                        (previous_activity['작업부서'] not in ['도장1부', '도장2부', '발판지원부']):
                        post_activity['작업부서'] = previous_activity['작업부서']
                    elif (post_activity['작업부서'] in ['도장1부', '도장2부', '발판지원부']) and \
                        (previous_activity['작업부서'] in ['도장1부', '도장2부', '발판지원부']):
                        print(block_code)

                    if previous_activity['작업부서'] != post_activity['작업부서']:
                        append_list(block_code, previous_activity['공정공종'], previous_activity['시작일'], previous_activity['종료일'],
                                    previous_activity['작업부서'], size, area)
                        finish_date = previous_activity['종료일']
                        previous_activity = post_activity

                    else:
                        previous_dict = dict(previous_activity)
                        post_dict = dict(post_activity)
                        previous_dict['종료일'] = post_dict['종료일'] if post_dict['종료일'] > previous_dict['종료일'] else previous_dict['종료일']
                        previous_dict['공정공종'] = previous_dict['공정공종'][0] + post_dict['공정공종']
                        if previous_dict['시작일'] < finish_date:
                            previous_dict['시작일'] = finish_date + 1
                        previous_activity = pd.Series(previous_dict)

                    if j == len(block_data) - 1:  # 마지막이면
                        append_list(block_code, previous_activity['공정공종'], previous_activity['시작일'],
                                    previous_activity['종료일'],
                                    previous_activity['작업부서'], size, area)

        # else:
        #     print(block_code)

print("데이터 전처리 완료")

new_activity['series'] = SERIES
new_activity['block_code'] = ONLY_BLOCK
new_activity['series_block_code'] = BLOCK
new_activity['process_code'] = PROCESS_CODE
new_activity['start_date'] = START_DATE
new_activity['finish_date'] = FINISH_DATE
new_activity['location'] = LOCATION
new_activity['size'] = SIZE
new_activity['area'] = AREA
new_activity.to_excel('./data/new_activity_A0001.xlsx')

print('Finish at', time.time() - start)

## SIZE, AREA REDETERMINE
data = pd.read_excel('./data/new_activity_A0001.xlsx')
#dock_mapping = pd.read_excel('./data/호선도크.xlsx')
#dock_mapping = dict(dock_mapping)
#mapping_table = pd.read_excel('./data/process_gis_mapping_table.xlsx')
#process_inout = {}
#for i in range(len(mapping_table)):
#    temp = mapping_table.iloc[i]
#    process_inout[temp['LOC']] = [temp['IN'], temp['OUT']]
#
size = list(data['size'])
area = list(data['area'])
size_wo_0 = [item for item in size if item != 0]
area_wo_0 = [item for item in area if item != 0]
size_mean = np.mean(size_wo_0)
area_mean = np.mean(area_wo_0)
#
data['size'] = data['size'].replace(0, size_mean)
data['area'] = data['area'].replace(0, area_mean)

network = {}
for i in range(12, 41):
    from_to_matrix = pd.read_excel('./network/distance_above_{0}_meters.xlsx'.format(i), index_col=0)
    # Virtual Stockyard까지의 거리 추가 --> 거리 = 0 (가상의 공간이므로)
    from_to_matrix.loc['Virtual'] = 0.0
    from_to_matrix.loc['Source'] = 0.0
    from_to_matrix.loc['Sink'] = 0.0
    from_to_matrix['Virtual'] = 0.0
    from_to_matrix['Source'] = 0.0
    from_to_matrix['Sink'] = 0.0
    network[i] = from_to_matrix


PE_Shelter = ['1도크쉘터', '2도크쉘터', '3도크쉘터', '의장쉘터', '특수선쉘터', '선행의장1공장쉘터', '선행의장2공장쉘터',
               '선행의장3공장쉘터', '대조립쉘터', '뉴판넬PE장쉘터', '대조립부속1동쉘터', '대조립2공장쉘터', '선행의장6공장쉘터',
               '화공설비쉘터', '판넬조립5부쉘터', '총조립SHOP쉘터', '대조립5부쉘터']

convert_to_process = {'가공소조립부 1야드' : '선각공장', '가공소조립부 2야드': '2야드 중조공장', '대조립1부': '대조립 1공장',
                      '대조립2부': '대조립 2공장', '대조립3부': '2야드 대조립공장', '의장생산부': '해양제관공장',
                      '판넬조립1부': '선각공장', '판넬조립2부': '2야드 판넬공장', '건조1부': ['1도크', '2도크'], '건조2부': '3도크',
                      '건조3부': ['8도크', '9도크'], '선행도장부': ['도장 1공장', '도장 2공장', '도장 3공장', '도장 4공장',
                                                        '도장 5공장', '도장 6공장', '도장 7공장', '도장 8공장',
                                                        '2야드 도장 1공장', '2야드 도장 2공장', '2야드 도장 3공장',
                                                        '2야드 도장 5공장', '2야드 도장 6공장'],
                      '선실생산부': '선실공장', '선행의장부': PE_Shelter, '기장부': PE_Shelter, '의장1부': PE_Shelter,
                      '의장3부': PE_Shelter, '외부': '외부'}


def convert_process(present_process):
    # 현재 Step
    process_convert_by_dict = convert_to_process[present_process] if present_process != 'Sink' else 'Sink'

    # 1:1 대응
    if type(process_convert_by_dict) == str:
        return process_convert_by_dict
    elif present_process == '건조1부' or present_process == '건조3부':
        return '8도크'
    else:
        return present_process
    # # dock 가야 하는 경우 -> random

    # elif (present_process == "선행도장부") or
    # # 그냥 process인 경우 + 경우의 수가 여러 개인 경우
    # else:
    #     ## 선행이나 후행이 하나로 정해진 경우에만
    #     distance = []
    #     pre_choice = process_convert_by_dict[:]
    #     road_size = 12
    #     from_to_matrix = network[road_size]
    #     compared_process = process_inout[previous_process][1]
    #     for process in process_convert_by_dict:
    #         process_temp = process_inout[process][0]
    #         if (process_temp in from_to_matrix.index) and (compared_process in list(from_to_matrix)) and \
    #                 from_to_matrix[process_temp][compared_process] is not None:
    #             distance.append(from_to_matrix[process_temp][compared_process])
    #         else:
    #             pre_choice.remove(process)
    #     if len(distance):
    #         process_idx = distance.index(min(distance))
    #         process = pre_choice[process_idx]
    #         return process
    #     else:
    #         print("Impossible! cus no next place as no road available", name,
    #               'from = {0}, to = {1}'.format(previous_process, present_process))
    #         return 'Impossible'

new_dataframe = pd.DataFrame(columns=['series_block_code', 'series', 'block_code', 'process_code', 'start_date', 'finish_date', 'location', 'size', 'area'])

block_group = data.groupby(data['series_block_code'])
block_list = list(data.drop_duplicates(['series_block_code'])['series_block_code'])
for block_code in block_list:
    block = block_group.get_group(block_code)
    block = block.sort_values(by=['start_date'], ascending=True)
    block = block.reset_index(drop=True)
    size = 10
    step = 0
    for i in range(len(block)):
        present_data = list(block.iloc[i])
        present_process = block.iloc[i]['location']
        convert_present_process = convert_process(present_process)
        if convert_present_process != 'Impossible':
            present_data[7] = convert_present_process
            block.iloc[i] = present_data
            del present_data[0]
            del present_data[7]
            new_dataframe.loc[len(new_dataframe)] = present_data
    # new_dataframe.loc[(len(new_dataframe)] = [0]

new_dataframe.to_excel('./data/Layout_data.xlsx')
data.to_excel('./data/Layout_data_pre.xlsx')


print(0)
