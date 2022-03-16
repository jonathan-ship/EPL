import json
import pandas as pd
import numpy as np
import copy

def read_inout_info():
    process_info_data = pd.read_excel('./Validation/Input/Factory_info.xlsx')
    inout_dict = dict()
    process_dict = dict()

    for i in range(len(process_info_data)):
        temp = process_info_data.iloc[i]
        name = temp['name']
        factory_type = temp['type']

        if factory_type not in process_dict.keys():
            process_dict[factory_type] = dict()
        inout_dict[name] = [temp['in'], temp['out']]
        process_dict[factory_type][name] = temp['Capacity']

    virtual_list = ['Stock', 'Shelter', 'Painting']
    for virtual in virtual_list:
        inout_dict[virtual] = [virtual, virtual]
    # process_dict['Stockyard'].append('Stock')
    # process_dict['Shelter'].append('Shelter')
    # process_dict['Painting'].append('Painting')

    return inout_dict, process_dict


def read_converting():
    converting_df = pd.read_excel('./Validation/Input/Converting.xlsx')
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


if __name__ == "__main__":
    event_tracer = pd.read_csv("./Validation/Result/result_Validation.csv")
    # 1. 제외 액티비티 확인
    # 파일 불러오기
    raw_data = pd.read_excel("./Validation/Input/Layout_Activity.xlsx")
    raw_data_bom = pd.read_excel("./Validation/Input/Layout_BOM.xlsx")

    # 물류 무관 공정 제외
    raw_data = raw_data[(raw_data['공정공종'] != 'C91') & (raw_data['공정공종'] != 'CX3') & \
                        (raw_data['공정공종'] != 'F91') & (raw_data['공정공종'] != 'FX3') & \
                        (raw_data['공정공종'] != 'G4A') & (raw_data['공정공종'] != 'G4B') & \
                        (raw_data['공정공종'] != 'GX3') & (raw_data['공정공종'] != 'HX3') & \
                        (raw_data['공정공종'] != 'K4A') & (raw_data['공정공종'] != 'K4B') & \
                        (raw_data['공정공종'] != 'L4B') & (raw_data['공정공종'] != 'L4A') & \
                        (raw_data['공정공종'] != 'LX3') & (raw_data['공정공종'] != 'JX3')]

    # 1-1. BOM에 없는 데이터 제거되었는 지 확인
    raw_data.loc[:, '호선'] = raw_data.loc[:, '호선'].apply(lambda x: str(x))
    raw_data.loc[:, '블록'] = raw_data.loc[:, '블록'].apply(lambda x: str(x))
    raw_data.loc[:, '시작일'] = pd.to_datetime(raw_data['시작일'], format='%Y%m%d')
    raw_data.loc[:, '종료일'] = pd.to_datetime(raw_data['종료일'], format='%Y%m%d')
    raw_data.loc[:, '블록코드'] = raw_data.loc[:, '호선'] + '_' + raw_data.loc[:, '블록']

    raw_data_bom.loc[:, '호선'] = raw_data_bom.loc[:, '호선'].apply(lambda x: str(x))
    raw_data_bom.loc[:, '블록'] = raw_data_bom.loc[:, '블록'].apply(lambda x: str(x))
    raw_data_bom.loc[:, '상위블록'] = raw_data_bom.loc[:, '상위블록'].apply(lambda x: str(x))
    raw_data_bom.loc[:, '블록코드'] = raw_data_bom.loc[:, '호선'] + '_' + raw_data_bom.loc[:, '블록']
    raw_data_bom.loc[:, '상위블록코드'] = raw_data_bom.loc[:, '호선'] + '_' + raw_data_bom.loc[:, '상위블록']
    #
    # # Activity Data에 있는 모든 블록코드
    # total_block = list(np.unique(list(raw_data['블록코드'])))
    # # BOM에 있는 블록 코드
    # total_block_bom = list(np.unique(list(raw_data_bom['블록코드']) + list(raw_data_bom['상위블록코드'])))
    #
    # in_bom_block = list()  # BOM에 있는 블록 --> 시뮬레이션에 있어야 함
    # out_bom_block = list()  # BOM에 없는 블록 --> 시뮬레이션에 있으면 안 됨
    #
    # for block_code in total_block:
    #     if block_code in total_block_bom:  # BOM에 있는 블록이면
    #         in_bom_block.append(block_code)
    #     else:  # BOM에 없는 블록이면
    #         out_bom_block.append(block_code)
    #
    # # Simulation Log에 있는 블록 코드
    # total_sim_block = list(np.unique(list(event_tracer['Part'])))
    #
    # # BOM에 있는 블록은 모두 시뮬레이션 되었어야
    # total_sim_block = sorted(total_sim_block)
    # in_bom_block = sorted(in_bom_block)
    # suc_idx_1_1 = True if total_sim_block == in_bom_block else False
    # print("Validation 1-1: BOM에 있는 모든 블록이 시뮬레이션 되었는지 =", suc_idx_1_1)
    #
    # # BOM에 없는 블록은 시뮬레이션 로그에 없어야
    # suc_idx_1_2 = True
    # for out_bom in out_bom_block:
    #     if out_bom in total_sim_block:  # 만약 BOM에 없는 블록이 시뮬레이션 로그에 있으면
    #         suc_idx_1_2 = False
    # print("Validation 1-1: BOM에 없는 모든 블록이 시뮬레이션에서 제외됐는지 =", suc_idx_1_2)

    validation_before = pd.read_excel('Validation_before.xlsx')
    validation_before.loc[:, '호선'] = validation_before.loc[:, '호선'].apply(lambda x: str(x))
    validation_before.loc[:, '블록'] = validation_before.loc[:, '블록'].apply(lambda x: str(x))
    validation_before.loc[:, '블록코드'] = validation_before.loc[:, '호선'] + '_' + validation_before.loc[:, '블록']
    validation_after = pd.read_excel('Validation_after.xlsx')


    # 2-1. 병합 : 작업부서가 동일한 경우
    print("Start Validation 2-1")
    before_21 = validation_before[validation_before['MEMO'] == "병합(작업부서 동일)"]
    before_21 = before_21.reset_index(drop=True)

    block_list_21 = list(np.unique(list(before_21['블록코드'])))
    for block_21 in block_list_21:
        temp = before_21[before_21['블록코드'] == block_21]
        temp_act_id = list(temp['ACT_ID'])

        temp_activity = raw_data[raw_data['블록코드'] == block_21]
        temp_activity = temp_activity.sort_values(by=['시작일', '종료일'], ascending=True)
        temp_activity = temp_activity.reset_index(drop=True)

        # Validation_before - 중일정 비교
        prior_activity = temp_activity.iloc[0]

        merged_start_time = None
        merged_finish_time = None
        merged_department = None

        for i in range(1, len(temp_activity)):
            next_activity = temp_activity.iloc[i]

            # 만약 작업부서가 동일해서 병합되어야 하는 액티비티라면
            if next_activity["ACT_ID"] in temp_act_id:
                # 부서가 같지 않다면 --> 오류
                if prior_activity['작업부서'] != next_activity['작업부서']:
                    print("Fail 2-1,", block_21)

            # 중일정 - Validation_after 비교
            # 병합된다면
            if prior_activity['작업부서'] == next_activity['작업부서']:
                merged_start_time = min(prior_activity['시작일'], next_activity['시작일'])
                merged_finish_time = max(prior_activity['종료일'], next_activity['종료일'])
                merged_department = prior_activity['작업부서']
            # 병합되지 않는다면
            else:
                # 이미 병합된 게 있는 경우
                if merged_start_time  is not None:
                    temp_after = validation_after[(validation_after['block'] == block_21) & (
                                validation_after['start_time'] == merged_start_time) & (
                                                              validation_after['finish_time'] == merged_finish_time) & (
                                                  validation_after['process'] == merged_department)]
                    if len(temp_after) == 0:  # 병합된 액티비티가 없으면
                        print("Fail 2-1,", block_21)
                    merged_start_time, merged_finish_time, merged_department = None, None, None
            prior_activity = next_activity
    print("Finish Validation 2-1")

    # 2-2. 병합 : 한 액티비티가 다른 액티비티에 포함되는 경우
    print("Start Validation 2-2")
    before_22 = validation_before[validation_before['MEMO'] == "병합(선행에 포함)"]
    before_22 = before_21.reset_index(drop=True)
    block_list_22 = list(np.unique(list(before_22['블록코드'])))

    for block_22 in block_list_22:
        temp = before_21[before_22['블록코드'] == block_22]
        temp = temp.reset_index(drop=True)

        temp_activity = raw_data[raw_data['블록코드'] == block_22]
        temp_activity = temp_activity.sort_values(by=['시작일', '종료일'], ascending=True)
        temp_activity = temp_activity.reset_index(drop=True)

        # Validation_before - 중일정 데이터 비교
        # 해당 액티비티가 포함되는 액티비티가 있는 지 확인
        flag = False
        for i in range(len(temp)):
            temp_before = temp.iloc[i]
            temp_before_start = temp_before['시작일']
            temp_before_finish = temp_before['종료일']

            for j in range(len(temp_activity)):
                temp_each_activity = temp_activity.iloc[j]
                temp_each_activity_start = temp_each_activity['시작일']
                temp_each_activity_finish = temp_each_activity['종료일']
                if (temp_before_start > temp_each_activity_start) and (temp_before_finish < temp_each_activity_finish):
                    flag = flag or True
                else:
                    flag = flag or False

        if flag == False:
            print("Fail 2-2", block_22)
    print("Start Validation 2-2")



    print("Finish Validation 2-2")

    # 2-4. 변경 : 액티비티의 종료일이 시작일보다 빠른 경우
    print("Start Validation 2-4")
    before_24 = validation_before[validation_before['MEMO'] == "변경(종료시각<시작시각)"]
    before_24 = before_24.reset_index(drop=True)

    for i in range(len(before_24)):
        temp = before_24.iloc[i]
        block_code = '{0}_{1}'.format(temp['호선'], temp['블록'])
        act_id = temp['ACT_ID']

        # Validation_before - 중일정데이터 비교
        activity_temp = raw_data[raw_data['ACT_ID'] == act_id]
        # 만약 시작일이 종료일보다 빠르면 --> 처리대상이 아님
        if activity_temp["시작일"] < activity_temp["종료일"]:
            print("Fail 2-4,", block_code)

        # Validation_after 확인 - 종료일 = 시작일인지
        temp_after = validation_after[
            (validation_after['block'] == block_code) & (validation_after['start_time'] == temp['시작일']) & (
                        validation_after['finish_time'] == temp['시작일']) & (validation_after['process'] == temp['작업부서'])]

        if len(temp_after) == 0:
            print("Fail 2-4,", block_code)
    print("Finish Validation 2-4")