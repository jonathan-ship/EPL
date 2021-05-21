import pandas as pd
import numpy as np
from datetime import datetime
import time

SERIES = []
ONLY_BLOCK = []
BLOCK = []
PROCESS_CODE = []
START_DATE = []
FINISH_DATE = []
LOCATION = []
LOC_INDICATOR = []


def append_list(block_code, process_code, start_date, finish_date, location, loc_indicator):
    SERIES.append(block_code[:5])
    ONLY_BLOCK.append(block_code[-5:])
    BLOCK.append(block_code)
    PROCESS_CODE.append(process_code)
    START_DATE.append(start_date)
    FINISH_DATE.append(finish_date)
    LOCATION.append(location)
    LOC_INDICATOR.append(loc_indicator)


start = time.time()

activity_data_all = pd.read_excel('../data/Layout_Activity.xlsx')
# 필요없는 공정 골라내기
activity_data = activity_data_all[(activity_data_all['공정공종'] != 'C91') & (activity_data_all['공정공종'] != 'CX3') & \
                                  (activity_data_all['공정공종'] != 'F91') & (activity_data_all['공정공종'] != 'FX3') & \
                                  (activity_data_all['공정공종'] != 'G4A') & (activity_data_all['공정공종'] != 'G4B') & \
                                  (activity_data_all['공정공종'] != 'GX3') & (activity_data_all['공정공종'] != 'HX3') & \
                                  (activity_data_all['공정공종'] != 'K4A') & (activity_data_all['공정공종'] != 'K4B') & \
                                  (activity_data_all['공정공종'] != 'L4B') & (activity_data_all['공정공종'] != 'L4A') & \
                                  (activity_data_all['공정공종'] != 'LX3') & (activity_data_all['공정공종'] != 'JX3')]
# activity_data = activity_data[(type(activity_data['시작일']) == int) & (int(activity_data['종료일']) == int)]
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
activity_data.loc[:, '블록'] = activity_data['블록'].apply(lambda x: str(x) if not (type(x) == int and x < 1000) else str(x) + 'E0')
activity_data.loc[:, 'block code'] = activity_data['호선'] + '_' + activity_data['블록']
activity_data.loc[:, 'process_head'] = activity_data['공정공종'].apply(lambda x: x[0] if type(x) == str else None)  # 공정공종 첫 알파벳
activity_data.loc[:, 'detail_process'] = activity_data['호선'] + activity_data['ACT_ID'].apply(lambda x: x[:8] if type(x) == str else None)
block_list = list(activity_data.drop_duplicates(['block code'])['block code'])
print("블록 코드 구현 완료")

detail_list = list(activity_data.drop_duplicates(['detail_process'])['detail_process'])
for detail in detail_list:
    temp = activity_data[activity_data['detail_process'] == detail]
    if len(temp) > 1:
        req_Index = temp[(temp['작업부서'] == '외부') | (temp['작업부서'] == None)].index.tolist()
        activity_data = activity_data.drop(req_Index)
print('세부공종 제거')
activity_data.to_excel('../data/temp.xlsx')
new_activity = pd.DataFrame(columns=['series_block_code','series', 'block_code', 'process_code', 'start_date', 'finish_date', 'location', 'loc_indicator'])
block_group = activity_data.groupby(activity_data['block code'])  ## block code에 따른 grouping

print('시이이이작 at', time.time() - start)

for block_code in block_list:
    block_data = block_group.get_group(block_code)  # block code 별로 grouping 한 결과
    heads = list(block_data.drop_duplicates(['process_head'])['process_head'])
    for head in heads:  # 공정공종 첫 알파벳
        grouped_by_process = block_data[block_data['process_head'] == head]
        if len(grouped_by_process):
            if head == 'A':
                if 'A01' in grouped_by_process['공정공종'].values:
                    index = grouped_by_process[grouped_by_process['공정공종'] == 'A01'].index.tolist()
                    for idx in index:
                        temp = grouped_by_process.loc[idx]
                        start_date = initial_date + pd.offsets.Day(int(temp['시작일']))
                        finish_date = initial_date + pd.offsets.Day(int(temp['종료일']))
                        append_list(block_code, 'A01', datetime.strftime(start_date, '%Y%m%d'),
                                    datetime.strftime(finish_date, '%Y%m%d'), temp['작업부서'], True)
                        grouped_by_process = grouped_by_process.drop([idx])
            elif head == 'C':
                if ('C12' in grouped_by_process['공정공종'].values) and ('C13' in grouped_by_process['공정공종'].values):
                    if len(grouped_by_process) > 2:
                        print(len(grouped_by_process))
                    for j in range(len(grouped_by_process)):
                        temp = grouped_by_process.iloc[j]
                        start_date = initial_date + pd.offsets.Day(int(temp['시작일']))
                        finish_date = initial_date + pd.offsets.Day(int(temp['종료일']))
                        append_list(block_code, temp['공정공종'], datetime.strftime(start_date, '%Y%m%d'),
                                    datetime.strftime(finish_date, '%Y%m%d'), temp['작업부서'], True)
            elif head == 'L':
                if 'L32' in grouped_by_process['공정공종'].values:
                    index = grouped_by_process[grouped_by_process['공정공종'] == 'L32'].index.tolist()
                    for idx in index:
                        temp = grouped_by_process.loc[idx]
                        start_date = initial_date + pd.offsets.Day(int(temp['시작일']))
                        finish_date = initial_date + pd.offsets.Day(int(temp['종료일']))
                        append_list(block_code, 'L32', datetime.strftime(start_date, '%Y%m%d'),
                                    datetime.strftime(finish_date, '%Y%m%d'), temp['작업부서'], True)
                        grouped_by_process = grouped_by_process.drop([idx])

            if len(grouped_by_process) > 0:
                early_date = initial_date + pd.offsets.Day(int(np.min(grouped_by_process['시작일'])))
                early_date = datetime.strftime(early_date, '%Y%m%d')
                latest_date = initial_date + pd.offsets.Day(int(np.max(grouped_by_process['종료일'])))
                latest_date = datetime.strftime(latest_date, '%Y%m%d')
                location_list = list(grouped_by_process.drop_duplicates(['작업부서'])['작업부서'])
                indicator = True if len(location_list) < 2 else False
                append_list(block_code, head, early_date, latest_date, location_list[0], indicator)
print("데이터 전처리 완료")

new_activity['series'] = SERIES
new_activity['block_code'] = ONLY_BLOCK
new_activity['series_block_code'] = BLOCK
new_activity['process_code'] = PROCESS_CODE
new_activity['start_date'] = START_DATE
new_activity['finish_date'] = FINISH_DATE
new_activity['location'] = LOCATION
new_activity['loc_indicator'] = LOC_INDICATOR
new_activity.to_excel('../data/new_activity_ALL.xlsx')
activity_data.to_excel('../data/Activity.xlsx')
print('Finish at', time.time() - start)
