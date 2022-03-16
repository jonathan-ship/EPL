import simpy, random, copy, math
import pandas as pd
import numpy as np

from collections import OrderedDict
from datetime import timedelta


# region Resource
class Transporter:
    def __init__(self, env, capacity, name, unloading_speed, loading_speed):
        self.capacity = capacity  # 최대적재가능 블록 중량 [ton]
        self.location = None  # 현재 위치
        self.name = name
        self.unloading_speed = unloading_speed
        self.loading_speed = loading_speed

        # number : 해당 capacity의 대수
        self.resource = simpy.Resource(env, capacity=1)


class Process:
    def __init__(self, env, name, capacity, type):
        self.name = name  # 공장 이름
        self.type = type  # Factory / Shelter / Painting

        self.used = 0.0  # 실제 사용 면적

        # capacity : 공장 면적
        self.resource = simpy.Container(env, capacity=capacity, init=capacity)


class Stockyard:
    def __init__(self, env, name, capacity):
        self.name = name
        self.type = "Stockyard"

        self.used = 0.0  # 실제 사용 면적

        # capacity : 적치장 면적
        self.resource = simpy.Container(env, capacity=capacity, init=capacity)


class Resource:
    def __init__(self, env, tp_dict, process_dict, network, inout, block_dict, convert_dict, monitor):
        self.env = env

        self.tp_dict = tp_dict  # {yard : {capacity : {tp_name : Transporter class (), ... } ... }, ... }
        self.process_dict = process_dict  # {type : {process_name : Process class (), ... }... }
        self.network = network  # { 출발지 : { 도착지 : 거리, ... } }
        self.inout = inout  # { 공장 이름 : [공장 입구, 공장 출구], ... }

        # 공장 선정을 위한 추가 변수
        self.block_dict = block_dict  # {'A0001_A11A0': Block class(), ... }, ... }
        self.convert_dict = convert_dict  # {'가공소조립부 1야드': '선각공장', ... }

        self.monitor = monitor  # monitor 클래스

        # TP 야드 별 최저/최대 용량
        self.tp_capacity_dict = dict()
        for yard in self.tp_dict.keys():
            self.tp_capacity_dict[yard] = list(np.unique([tp.capacity for tp in self.tp_dict[yard].values()]))

    # Transporter 호출
    def using_transporter(self, block, next_process):
        # 만약 블록의 무게가 TP 최대 용량보다 큰 경우 최대용량으로 고려
        load = block.weight if block.weight <= np.max(self.tp_capacity_dict[block.yard]) else np.max(self.tp_capacity_dict[block.yard])
        available_capacity = min(list(filter(lambda x: x >= load, self.tp_capacity_dict[block.yard])))
        available_tp = list()
        for tp in self.tp_dict[block.yard].keys():
            if self.tp_dict[block.yard][tp].capacity == available_capacity:
                available_tp.append(tp)
        # 호출될 TP 선정
        if len(available_tp) == 1:  # 해당 capacity Transporter가 한 대인 경우
            tp_name = available_tp[0]
        else:  # 해당 Transporter가 두 대 이상인 경우
            tp_idle = list(filter(lambda x: self.tp_dict[block.yard][x].resource.count == 0, available_tp))
            if len(tp_idle) > 0:  # idle한 TP가 있는 경우
                tp_name = tp_idle[0]
            else:  # idle한 TP가 없는 경우 -> 가까운 TP 호출
                distance_list = list()
                shortest_tp = available_tp[0]
                tp = self.tp_dict[block.yard][available_tp[0]]
                shortest_distance = self.network[tp.location][self.inout[block.location][1]]
                for idx in range(1, len(available_tp)):
                    tp = self.tp_dict[block.yard][available_tp[idx]]
                    distance = self.network[tp.location][self.inout[block.location][1]] if tp.location is not None else 0.0
                    if distance < shortest_distance:
                        shortest_distance = distance
                        shortest_tp = available_tp[idx]

                tp_name = shortest_tp

        # Transporter 호출
        transporter = self.tp_dict[block.yard][tp_name]
        print(self.env.now, "TP {0} request".format(tp_name))
        with transporter.resource.request() as req:

            retrieval_distance = self.network[transporter.location][
                self.inout[block.location][1]] if transporter.location is not None else 0

            self.monitor.record(self.env.now, None, None,part_id=block.name, event="Transporter Requested",
                                resource=tp_name)
            # 트랜스포터 호출
            yield req
            print(self.env.now, "TP {0} Get".format(tp_name))
            self.monitor.record(self.env.now, None, None,part_id=block.name, event="Transporter Assigned",
                                resource=tp_name)
            # 공차 이동 시간만큼 진행
            yield self.env.timeout(retrieval_distance / transporter.unloading_speed)

            transporter_location = None
            for factory_name in self.inout.keys():
                if transporter.location in self.inout[factory_name]:
                    transporter_location = factory_name
            self.monitor.record(self.env.now, None, None, part_id=block.name, event="Transporter Loading Start",
                                resource=tp_name, load=block.weight, unit="ton",
                                from_process=transporter_location, to_process=block.location,
                                distance=retrieval_distance)

            # 트랜스포터의 현재 위치 저장 (블록의 현재 위치의 출구)
            transporter.location = self.inout[block.location][1]

            # 상차 이동 거리
            transporting_distance = self.network[self.inout[block.location][1]][self.inout[next_process][0]]
            # 상차 이동 시간만큼 진행
            yield self.env.timeout(transporting_distance / transporter.loading_speed)
            self.monitor.record(self.env.now, None, None,part_id=block.name, event="Transporter Loading Completed",
                                resource=tp_name, load=block.weight, unit="ton",
                                from_process=block.location, to_process=next_process,
                                distance=transporting_distance)
            # 트랜스포터의 현재 위치 저장 (다음 공장의 입구)
            transporter.location = self.inout[next_process][0]

        return tp_name

    # 부서 -> 공장 변환
    def convert_process(self, block, step):
        present_process = block.data[(step, "process")]

        if present_process != 'Sink':
            working_code = list(block.data[(step, 'work')])
            working_code = sorted(working_code, reverse=True)
        else:
            working_code = 'X'

        present_process_work = '{0}{1}'.format(working_code[0],
                                               present_process) if present_process != 'Sink' else 'Sink'

        # 이미 변환된 경우 혹은 블록 이동 없는 경우
        if present_process_work not in self.convert_dict.keys():
            if present_process in ['도장1부', '도장2부', '발판지원부']:
                converted = self.dispatching_process(block, "not moving", step)
            else:
                converted = present_process

        # 도크인 경우
        elif self.convert_dict[present_process_work] == "Dock":
            converted = "{0}도크".format(block.dock)

        # 1:1 매칭인 경우
        elif type(self.convert_dict[present_process_work]) == str:
            converted = self.convert_dict[present_process_work]

        # 1:다 매칭인 경우 (의장, 도장)
        else:
            if present_process == "선행도장부":
                process_type = "Painting"
            elif present_process in ["선행의장부", "의장1부", "의장2부", "의장3부", "기장부"]:
                process_type = "Shelter"
            else:
                process_type = "Factory"

            converted = self.dispatching_process(block, process_type, step)

        block.data[(step, 'process')] = converted

        return converted

    def dispatching_process(self, block, process_type, step):
        # 비교 공장 선택
        if step == 0:  # 첫 번째 공정일 때
            if block.child is None:  # 하위 블록도 없고, 첫 번째 공장일 때
                source_location = "선각공장" if block.yard == 1 else "2야드 중조공장"
            else:
                source_location_list = list()
                for child in block.child:  # 하위블록이 있는 경우, 하위 블록 중 가장 무거운 블록의 현재 위치를 기준으로 함
                    child_current_location = self.block_dict[child].location
                    if child_current_location is not None:  # 하위 블록이 현재 작업 중인 경우
                        source_location_list.append([self.block_dict[child].weight, child_current_location])
                    else:  # 아직 하위블록의 작업이 시작하지 않은 경우, 하위블록의 하위블록의 현재 위치를 기준으로 함
                        if self.block_dict[child].child is not None:  # 하위블록의 하위블록이 있는 경우
                            for grand_child in self.block_dict[child].child:
                                if self.block_dict[grand_child].location is not None:
                                    source_location_list.append(
                                        [self.block_dict[grand_child].weight, self.block_dict[grand_child].location])

                if len(source_location_list) == 0:  # 하위 블록과 하위블록의 하위블록도 모두 작업 전일 때
                    source_location = "선각공장" if block.yard == 1 else "2야드 중조공장"

                else:  # 기준 목록 중 블록의 무게가 가장 큰 것을 source location으로 정의
                    source_location_list = sorted(source_location_list, key=lambda x: x[0], reverse=True)
                    source_location = source_location_list[0][1]
        else:  # 이전 공정이 있을 때
            source_location = self.block_dict[block.name].location

        if process_type == "not moving":
            return source_location

        else:
            converted_process = list(self.process_dict[process_type].keys())
            if process_type != "Factory":
                converted_process.remove(process_type)
            ## Case 2
            # if process_type == "Painting":
            #     if block.dock == 3:
            #         converted_process = ['도장4공장', '도장8공장', '도장5공장']
            #     else:
            #         converted_process.remove('도장4공장')
            #         converted_process.remove('도장5공장')
            #         converted_process.remove('도장8공장')
            # elif process_type == "Shelter":
            #     if '3도크PE' in converted_process:
            #         if block.dock == 3:
            #             converted_process = ['3도크PE']
            #         else:
            #             converted_process.remove('3도크PE')
            #     else:
            #         if block.dock == 3:
            #             converted_process = ["의장쉘터", '선행의장쉘터', '선대PE']
            #         else:
            #             converted_process.remove('의장쉘터')
            #             converted_process.remove('선행의장쉘터')
            #             converted_process.remove('선대PE')
            distance = list()
            for compared in converted_process:
                # network matrix에 점이 존재하고, 현재 잔여 면적이 블록의 면적보다 클 때 distance list에 추가
                if (self.inout[source_location][1] in self.network.keys()) and \
                        (self.inout[compared][0] in self.network.keys()) and \
                        (self.process_dict[process_type][compared].resource.level >= block.area):
                    distance.append([compared, self.network[self.inout[source_location][1]][self.inout[compared][0]]])

            if len(distance) > 0:
                # 거리 순으로 정렬 후, 가장 가까운 공장으로 배정
                distance = sorted(distance, key=lambda x: x[1])
                return distance[0][0]
            else:
                # 만약 잔여면적이 블록 면적보다 큰 공장이 없을 경우, 가상의 공장으로 배정
                return process_type

    def dispatching_stockyard(self, block, step):
        next_process = self.convert_process(block, step + 1)
        if next_process == "Sink":
            next_process = self.convert_process(self.block_dict[block.parent], 0)

        stock_list = list(self.process_dict["Stockyard"].keys())
        stock_list.remove("Stockyard")

        ## Case 2
        # if block.dock == 3:
        #     stock_list = ['E6', 'E7', 'E8', 'E9']
        # else:
        #     stock_list.remove('E6')
        #     stock_list.remove('E7')
        #     stock_list.remove('E8')
        #     stock_list.remove('E9')

        distance = list()
        for stockyard in stock_list:
            if next_process == "선행도장부":
                print(0)
            # network matrix에 점이 존재하고, 현재 잔여 면적이 블록의 면적보다 클 때 distance list에 추가
            if (self.inout[stockyard][1] in self.network.keys()) and (
                    self.inout[next_process][0] in self.network.keys()) and (
                    self.process_dict["Stockyard"][stockyard].resource.level >= block.area):
                distance.append([stockyard, self.network[self.inout[stockyard][1]][self.inout[next_process][0]]])

        if len(distance) > 0:
            # 거리 순으로 정렬 후, 가장 가까운 적치장으로 배정
            distance = sorted(distance, key=lambda x: x[1])
            return distance[0][0]
        else:
            # 만약 잔여면적이 블록 면적보다 큰 적치장이 없을 경우, 가상의 적치장으로 배정
            return "Stockyard"

# endregion

# region Part&Source&Process
class Block:
    def __init__(self, name, area, size, weight, data, dock, child=None, parent=None):
        self.name = name
        self.area = area
        self.size = size  # 임시
        self.weight = weight
        columns = pd.MultiIndex.from_product([[i for i in range(8)], ['start_time', 'process_time', 'process', 'work']])
        self.data = pd.Series(data, index=columns)
        self.dock = dock
        self.child = child
        self.parent = parent

        self.yard = 1 if dock in [1, 2, 3, 4, 5] else 2
        self.location = None # 공장에 가 있는 경우
        self.process_type = None
        self.stock = False  # 적치장에 가 있으면 True
        self.working_code = 'A'  # 현재 공정공종 상태 -> initial state : 가장 작은 단계인 가공(A)
        self.erecting = True if self.child is None else False


class Part:
    def __init__(self, env, name, block, working_data, block_dict, source_dict, process_dict, process_converting, inout,
                 network, dock, size, area, monitor, resource, child=None, parent=None, stock_lag=2):

        # Variables from input
        self.env = env
        self.name = name  # 블록 이름
        self.block = block  # Block class로 모델링 한 블록
        columns = pd.MultiIndex.from_product([[i for i in range(8)], ['start_time', 'process_time', 'process', 'work']])
        self.data = pd.Series(working_data, index=columns)

        self.block_dict = block_dict  # { "A0001_A11A0" : Block class (), ... }
        self.source_dict = source_dict  # { "A0001_A11A0" : Part class (), ...}
        self.process_dict = process_dict # {type : {process_name : Process class (), ... }... }
        self.process_converting = process_converting  # { "가공소조립부1야드" : "선각공장", ... }
        self.inout = inout  # { 공장 이름 : [공장 입구, 공장 출구], ... }
        self.network = network  # { 출발지 : { 도착지 : 거리, ... } }

        self.dock = dock
        self.yard = 1 if dock in [1, 2, 3, 4, 5] else 2
        self.size = size
        self.area = area

        self.monitor = monitor  # monitor 클래스
        self.resource = resource

        self.child = child
        self.parent = parent

        self.stock_lag = stock_lag

        # Variables defined
        self.child_store = simpy.Store(env)
        self.store = simpy.Store(env)
        if self.child is None:
            self.store.put(self.block)

        self.tp_flag = False
        self.in_stock = False
        self.where_stock = None
        self.finish = False
        self.starting = True if self.child is None else False

        self.env.process(self._sourcing())

    def _sourcing(self):
        # 하위블록이 있는 경우 하위블록 중 하나라도 도착해야(종료해야) 상위블록 시작
        step = 0
        while not self.finish:

            planned_process = self.data[(step, 'process')]
            start_time = self.data[(step, 'start_time')]
            working_code = self.data[(step - 1, 'work')] if step >= 1 else 'A'
            working_code_list = list(working_code)
            working_code_list = sorted(working_code_list, reverse=True)
            working_code = working_code_list[0]

            self.block_dict[self.name].working_code = working_code

            # 1. 다음 시작 시간까지 timeout
            # 대조립 이후 공정이면 Transporter 사용
            if working_code in ['F', 'G', 'H', 'J', 'M', 'K', 'L', 'N']:
                self.tp_flag = True
                # 하루 전에 미리 호출
                lag_time = start_time - 1 - self.env.now
                # 적정 TP capacity
            else:
                self.tp_flag = False
                # 정시에 호출
                lag_time = start_time - self.env.now
            lag_time = lag_time if lag_time >= 0 else 0
            yield self.env.timeout(lag_time)

            # 2. Department -> Factory로 변환 및 면적 선점
            process = self.resource.convert_process(self.block, step)

            if process in self.process_dict["Shelter"].keys():
                process_type = "Shelter"
            elif process in self.process_dict["Painting"].keys():
                process_type = "Painting"
            else:
                process_type = "Factory"

            # 첫 단계면 -->
            if step == 0:
                part = yield self.store.get()
                self.starting = True
                self.monitor.record(self.env.now, None, None, part_id=self.name, event="Block Created")
                self.block_dict['Created Part'] += 1
                print("{0} created at {1}, Creating Part is finished {2}/{3}".format(self.name, self.env.now,
                                                                                     self.block_dict['Created Part'],
                                                                                     len(self.block_dict) - 1))
                if self.child is not None:
                    self.env.process(self._erecting(process))

            # 면적 선점
            print(self.env.now, "Area {0} request".format(process))
            if process == "가공소조립부 1야드":
                print(0)
            self.monitor.record(self.env.now, None, None, part_id=self.name, event="Area Request",
                                process_type=process_type, resource=process, load=self.area,
                                memo=self.process_dict[process_type][process].resource.level)

            yield self.process_dict[process_type][process].resource.get(self.area)
            print(self.env.now, "Area {0} Get".format(process))
            self.monitor.record(self.env.now, None, None, part_id=self.name, event="Area Preempted",
                                process_type=process_type, resource=process,
                                load=self.process_dict[process_type][process].resource.level)

            # Transporter를 사용할 경우, Transporter를 이용하여 다음 공정으로 이동
            # 블록이 적치장이 있으면, 다음 공정 시작 직전에 블록 반출 후 다음 공정으로 이동 => 적치장에 면적 반환
            if self.in_stock:
                # 면적 반환
                self.process_dict["Stockyard"][self.where_stock].resource.put(self.area)
                self.process_dict["Stockyard"][self.where_stock].used -= self.area

                self.monitor.record(self.env.now, self.where_stock, None, part_id=self.name, event="Stockyard Out",
                                    process_type="Stockyard", load=self.process_dict["Stockyard"][self.where_stock].used,
                                    memo="for next process")

                # 적치장에서 공정으로 이동
                if any(code in ['F', 'G', 'H', 'J', 'M', 'K', 'L', 'N'] for code in
                       list(self.data[(step - 1, 'work')])):  # 이전 공정까지의 블록이 트랜스포터를 필요로 하면
                    tp_name = yield self.env.process(self.resource.using_transporter(self.block, process))
                    self.monitor.record(self.env.now, None, None, part_id=self.name, event="Stockyard to Process",
                                        resource=tp_name, load=self.block.weight, from_process=self.where_stock,
                                        to_process=process,
                                        distance=self.network[self.inout[self.where_stock][1]][self.inout[process][0]])
                    # 적치장 관련 변수 초기화
                    self.in_stock = False
                    self.where_stock = None

            # 적치장을 들리지 않고 바로 공장 - 공장으로 이동
            elif (not self.in_stock) and self.tp_flag:
                tp_name = yield self.env.process(self.resource.using_transporter(self.block, process))
                self.monitor.record(self.env.now, None, None, part_id=self.name, event="Process to Process",
                                    resource=tp_name, load=self.block.weight, from_process=self.block.location,
                                    to_process=process,
                                    distance=self.network[self.inout[self.block.location][1]][self.inout[process][0]])

            # 블록 위치 업데이트
            self.block_dict[self.name].location = process
            self.block_dict[self.name].process_type = process_type

            # 공장에서 작업
            self.process_dict[process_type][process].used += self.area  # 실제 사용 면적
            self.monitor.record(self.env.now, process, None, part_id=self.name, event="Process In", resource=process,
                                process_type=process_type, load=self.process_dict[process_type][process].used,
                                unit="m2")
            self.monitor.record(self.env.now, process, None, part_id=self.name, event="Work Start", resource=process,
                                process_type=process_type)

            # 작업 시간만큼 진행
            yield self.env.timeout(self.data[(step, 'process_time')])
            self.monitor.record(self.env.now, process, None, part_id=self.name, event="Work Finish", resource=process,
                                process_type=process_type)

            # 만약 다음 공정의 시작시간 혹은 상위블록 첫 시작시간까지 stock lag 이하로 남았을 때 -> 그 시간만큼 현재 공장에서 stay
            if (self.name == "A0004_A11A0") and (step == 2):
                print(0)
            next_start_time = 0
            stock_reason = None
            if self.data[(step + 1, 'process')] != 'Sink':  # 다음 작업이 남아있는 경우
                next_start_time = self.data[(step + 1, 'start_time')]
                stock_reason = "for next process"
            elif self.parent is not None:  # 다음 작업 없이 상위블록으로 가야할 경우
                next_start_time = self.block_dict[self.parent].data[(0, 'start_time')]
                stock_reason = "for parent block"

            # stay 할 때
            if next_start_time - self.env.now < self.stock_lag:
                if next_start_time - self.env.now >= 0:
                    yield self.env.timeout(next_start_time - self.env.now)

                if stock_reason == "for parent block":
                    self.source_dict[self.parent].child_store.put(self.name)
                    self.source_dict[self.parent].store.put(self.block_dict[self.parent])
                    self.finish = True
                else:
                    self.process_dict[process_type][process].resource.put(self.area)
                    self.process_dict[process_type][process].used -= self.area
                    self.monitor.record(self.env.now, None, None, part_id=self.name, event="Area Release",
                                        resource=process,
                                        load=self.area, memo=self.process_dict[process_type][process].used)

            # 적치장으로 보내야 할 때
            elif next_start_time - self.env.now >= self.stock_lag:
                # 적치장 탐색
                self.where_stock = self.resource.dispatching_stockyard(self.block, step)
                # 적치장 면적 선점
                yield self.process_dict["Stockyard"][self.where_stock].resource.get(self.area)

                # 적치장으로 이동
                if self.tp_flag:
                    tp_name = yield self.env.process(self.resource.using_transporter(self.block, self.where_stock))
                    self.monitor.record(self.env.now, None, None, part_id=self.name, event="Process to Stockyard",
                                        resource=tp_name, load=self.block.weight, from_process=self.block.location,
                                        to_process=self.where_stock,
                                        distance=self.network[self.inout[self.block.location][1]][
                                            self.inout[self.where_stock][0]])

                # 적치장 관련 변수 업데이트
                self.in_stock = True
                self.process_dict["Stockyard"][self.where_stock].used += self.area
                self.block_dict[self.name].location = self.where_stock
                self.block_dict[self.name].process_type = "Stockyard"
                self.block_dict[self.name].stock = True

                # 이벤트 기록
                self.monitor.record(self.env.now, self.where_stock, None, part_id=self.name, event="Stockyard In",
                                    load=self.process_dict["Stockyard"][self.where_stock].used)

                # 선행 공장 면적 반납
                self.process_dict[process_type][process].resource.put(self.area)
                self.process_dict[process_type][process].used -= self.area
                self.monitor.record(self.env.now, None, None, part_id=self.name, event="Area Release", resource=process,
                                    load=self.area, memo=self.process_dict[process_type][process].used)

                # 적치장 입고 정보를 상위블록으로 넘겨줌
                if stock_reason == "for parent block":
                    self.source_dict[self.parent].child_store.put(self.name)
                    self.source_dict[self.parent].store.put(self.block_dict[self.parent])
                    self.finish = True

            # Sink로의 이동여부 결정
            if (self.parent is None) and (self.data[(step + 1, 'process')] == "Sink"):
                self.process_dict['Sink'].put(self.block)
                    # self.process_dict[process_type][process].resource.put(self.area)
                    # self.monitor.record(self.env.now, None, None, part_id=self.name, event="Area Release",
                    #                     resource=process,
                    #                     load=self.area, memo=self.process_dict[process_type][process].used)
                self.finish = True
            else:
                step += 1

    def _erecting(self, first_process):
        child_num = 0
        while child_num != len(self.child):
            child_block = yield self.child_store.get()
            child_block_class = self.block_dict[child_block]
            # 면적 반환
            self.process_dict[child_block_class.process_type][child_block_class.location].resource.put(
                child_block_class.area)
            self.process_dict[child_block_class.process_type][
                child_block_class.location].used -= child_block_class.area
            self.monitor.record(self.env.now, None, None, part_id=child_block, event="Area Release",
                                resource=child_block_class.location, load=child_block_class.area,
                                memo=self.process_dict[child_block_class.process_type][
                                    child_block_class.location].used)
            record_event = "Process Out" if child_block_class.process_type != "Stockyard" else "Stockyard Out"
            self.monitor.record(self.env.now, child_block_class.location, None, part_id=child_block,
                                event=record_event, process_type=child_block_class.process_type,
                                load=self.process_dict[child_block_class.process_type][
                                    child_block_class.location].used, memo="for parent block")

            child_tp_flag = True if child_block_class.working_code in ['F', 'G', 'H', 'J', 'M', 'K', 'L',
                                                                       'N'] else False
            used_resource = None
            if child_tp_flag:
                used_resource = yield self.env.process(
                    self.resource.using_transporter(self.block_dict[child_block], first_process))

            self.monitor.record(self.env.now, None, None, part_id=child_block, event="Child to Parent",
                                process_type=child_block_class.process_type, resource=used_resource,
                                from_process=child_block_class.location, to_process=first_process,
                                distance=self.network[self.inout[child_block_class.location][1]][
                                    self.inout[first_process][0]], load=child_block_class.weight)

            self.process_dict['Sink'].put(self.block_dict[child_block])
            child_num += 1

# endregion

# region Sink
class Sink:
    def __init__(self, env, monitor, tot_num):
        self.env = env
        self.monitor = monitor
        self.tot_num = tot_num  # 전체 블록 개수

        self.parts_rec = 0

    def put(self, block):
        self.monitor.record(self.env.now, "Sink", None, part_id=block.name, event="Block Completed")
        self.parts_rec += 1

        print("{0} is finished at {1}, {2}/{3}".format(block.name, self.env.now, self.parts_rec, self.tot_num))
# endregion

# region Monitor
class Monitor:
    def __init__(self, result_path, project_name, initial_date):
        self.file_path = result_path
        self.project_name = project_name
        self.initial_date = initial_date

        self.time = list()
        self.date = list()
        self.event = list()
        self.part_id = list()
        self.process = list()
        self.subprocess = list()
        self.resource = list()
        self.memo = list()
        self.process_type = list()
        self.load = list()
        self.unit = list()
        self.from_process = list()
        self.to_process = list()
        self.distance = list()
        self.num = list()
        self.work = list()
        self.department = list()

        self.created = 0
        self.completed = 0
        self.tp_used = dict()
        self.road_used = dict()

    def record(self, time, process, subprocess, part_id=None, event=None, resource=None, memo=None, process_type=None,
               load=None, unit=None, from_process=None, to_process=None, distance=None, num=None, work=None,
               department=None):
        self.time.append(time)
        date_time = self.initial_date + timedelta(days=math.floor(time))
        self.date.append(date_time.date())
        self.event.append(event)
        self.part_id.append(part_id)
        self.process.append(process)
        self.subprocess.append(subprocess)
        self.resource.append(resource)
        self.memo.append(memo)
        self.process_type.append(process_type)
        self.load.append(load)
        self.unit.append(unit)
        self.from_process.append(from_process)
        self.to_process.append(to_process)
        self.distance.append(distance)
        self.num.append(num)
        self.work.append(work)
        self.department.append(department)

        if event == 'Block Created':
            self.created += 1
        elif event == 'Block Completed':
            self.completed += 1

    def save_information(self):
        # 1. event_tracer
        event_tracer = pd.DataFrame()
        event_tracer['Date'] = self.date
        event_tracer['Part'] = self.part_id
        event_tracer['Process'] = self.process
        event_tracer['SubProcess'] = self.subprocess
        event_tracer['Resource'] = self.resource
        event_tracer['Event'] = self.event
        event_tracer['Process Type'] = self.process_type
        event_tracer['Load'] = self.load
        event_tracer['Unit'] = self.unit
        event_tracer['From'] = self.from_process
        event_tracer['To'] = self.to_process
        event_tracer['Distance'] = self.distance
        event_tracer['Memo'] = self.memo
        event_tracer['Work'] = self.work  # 공정공종
        event_tracer['Department'] = self.department
        event_tracer['Simulation time'] = self.time
        event_tracer['Number'] = self.num

        path_event_tracer = self.file_path + 'result_{0}.csv'.format(self.project_name)

        event_tracer.to_csv(path_event_tracer, encoding='utf-8-sig')

        return path_event_tracer
# endregion















