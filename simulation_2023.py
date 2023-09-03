import simpy, random, copy, math
import pandas as pd
import numpy as np

from collections import OrderedDict
from datetime import timedelta


# region Operation
class Operation:
    def __init__(self, start_time, process_time, process, work):
        self.start_time = start_time  # 작업 시작 시간
        self.process_time = process_time  # 작업 시간
        self.process = process
        self.work = work  # 공정공종

        self.area_requested = False
        self.area_preempted = False
        self.area_preempted_waiting_list = list()


# endregion

# region Job
class Block:
    def __init__(self, env, name, area, weight, schedule, dock, child=None, parent=None):
        self.env = env
        self.name = name
        self.area = area  # Process/Stockyard 점유 면적
        self.weight = weight  # 무게 -> transporter 사용 시 고려
        self.schedule = schedule  # operation list
        self.dock = dock  # 작업 도크
        self.child = child  # 하위블록, type: list
        self.parent = parent  # 상위블록, type: str

        self.yard = 1 if dock in [1, 2, 3, 4, 5] else 2  # 작업 야드, 사용가능한 transporter 및 도장공장, 쉘터, 적치장 결정
        self.location = None  # 현재 위치
        self.process_type = None  # 현재 위치하고 있는 공장의 process type, [Factory, Stockyard, Painting, Shelter] 중 하나
        self.in_stock = False  # 적치장에 있으면 True
        self.working_code = 'A'  # 현재 작업 중인 스케줄의 공정공종 / initial state: 'A' (가장 낮은 단계)

        self.step = 0  # 처리한 Operation의 개수
        self.finished_child = 0  # 작업을 완료한 Child Block의 개수
        self.is_finished_first_process = False if self.child is not None else True
        self.wait_idx = False
        self.wait_event = self.env.event()


# endregion

# region Resource
class Transporter:
    def __init__(self, env, name, capacity, unloading_speed, loading_speed, number=1, location=None):
        self.env = env
        self.name = name
        self.capacity = capacity
        self.unloading_speed = unloading_speed
        self.loading_speed = loading_speed

        self.location = location  # 현재 location / 처음 위치 무작위 결정
        self.next_location = None
        self.planned_finish_time = 0.0

        # number : 해당 capacity를 가진 TP의 대수
        self.resource = simpy.Resource(self.env, capacity=number)


class Management:
    def __init__(self, env, tp_dict, process_dict, source_dict, distance_matrix, inout, block_dict, dep_fac_dict,
                 monitor):
        self.env = env
        self.tp_dict = tp_dict  # transporter dictionary / {yard : {capacity : {tp_name : Transporter class (), ... } ... }, ... }
        self.process_dict = process_dict  # process/stockyard dictionary / {process_type(Factory, Stockyard, Painting, Shelter): {process_name: process class(), ... } ... }
        self.source_dict = source_dict  # {'A0001_A11A0': Source class(), ... }
        self.distance_matrix = distance_matrix  # From-To Distance Matrix / {출발지: {도착지: 거리, ... } }
        # self.road_matrix = road_matrix  # From-To Edge List Matrix  / {출발지: {도착치: 도착지까지 거쳐야 하는 도로 리스트, ...} ... }
        self.inout = inout  # 공장별 입출구 이름 / {공장 이름: [공장 입구, 공장 출구], ... }
        self.block_dict = block_dict  # {'A0001_A11A0': Block class(), ... }
        self.dep_fac_dict = dep_fac_dict  # {'A가공소조립부 1야드': '선각공장', ... }
        self.monitor = monitor  # Monitor Class

        # 야드 별 TP 최저/최대 용량
        self.tp_capacity = dict()
        for yard in self.tp_dict.keys():
            self.tp_capacity[yard] = list(np.unique([tp.capacity for tp in self.tp_dict[yard].values()]))

    def get_process_type(self, process):
        its_process_type = None
        for process_type in self.process_dict.keys():
            if process in self.process_dict[process_type].keys():
                its_process_type = process_type
                break

        return its_process_type

    def assign_transporter(self, block, next_process, idx=None):
        # 만약 블록의 무게가 TP 최대 용량보다 큰 경우 최대용량으로 고려
        load = block.weight if block.weight <= np.max(self.tp_capacity[block.yard]) else np.max(
            self.tp_capacity[block.yard])
        available_capacity = min(list(filter(lambda x: x >= load, self.tp_capacity[block.yard])))


        # Capacity 조건을 만족하는 TP list
        available_tp = list()
        for tp in self.tp_dict[block.yard].keys():
            if self.tp_dict[block.yard][tp].capacity == available_capacity:
                available_tp.append(tp)

        tp_idle = list()
        # 호출할 TP 결정
        if len(available_tp) == 1:  # 해당 Capacity를 가지는 TP가 한 대만 있는 경우
            selected_tp = self.tp_dict[block.yard][available_tp[0]]
            selected_tp_start_location = selected_tp.location if selected_tp.resource.count > 0 else selected_tp.next_location

        else:  # 해당 Capacity를 가지는 TP가 여러 대인 경우
            tp_idle = list(
                filter(lambda x: self.tp_dict[block.yard][x].resource.count == 0, available_tp))  # idle한 TP 리스트

        # 현재 블록 위치로 도착 시간이 가장 빠른 TP 선택
        # 선택 대상 TP: 유휴 TP가 있으면 tp_idle 중에서, 유휴 TP가 없으면 현재 이동 중인 TP 대상
        tp_list = tp_idle if len(tp_idle) > 0 else available_tp

        expected_arrived_time = 1e10
        selected_tp = None
        selected_tp_start_location = None
        for tp_name in tp_list:
            tp = self.tp_dict[block.yard][tp_name]
            # TP가 쉬고 있으면 현재 시각에 출발 / 이동 중이면 미리 기록한 planned_finish time에 출발
            start_time = self.env.now if tp.resource.count == 0 else tp.planned_finish_time
            # TP가 쉬고 있으면 현재 위치에서 / 이동 중이면 현대 도착 위치에서 출발
            start_location = tp.location if tp.resource.count == 0 else tp.next_location
            distance = self.distance_matrix[start_location][self.inout[block.location][1]]
            moving_time = distance / tp.unloading_speed
            if expected_arrived_time > start_time + moving_time:
                selected_tp = tp
                expected_arrived_time = start_time + moving_time
                selected_tp_start_location = start_location

        # transporter 정보 업데이트
        selected_tp.next_location = self.inout[next_process][0]
        retrieval_distance = self.distance_matrix[selected_tp_start_location][self.inout[block.location][1]]
        retrieval_time = retrieval_distance / selected_tp.unloading_speed
        transporting_distance = self.distance_matrix[self.inout[block.location][1]][self.inout[next_process][0]]
        transporting_time = transporting_distance / selected_tp.loading_speed
        selected_tp.planned_finish_time = self.env.now + retrieval_time + transporting_time

        self.monitor.record(self.env.now, "Management", part_id=block.name, event="Transporter Requested",
                            resource=selected_tp.name, work=block.schedule[block.step].work)
        with selected_tp.resource.request() as req:
            yield req
            self.monitor.record(self.env.now, "Management", part_id=block.name, event="Transporter Assigned",
                                resource=selected_tp.name, work=block.schedule[block.step].work)

            # 블록 위치로 이동
            yield self.env.timeout(retrieval_time)
            selected_tp.location = self.inout[block.location][1]

            transporter_location = None
            for factory_name in self.inout.keys():
                if selected_tp_start_location in self.inout[factory_name]:
                    transporter_location = factory_name
            self.monitor.record(self.env.now, "Management", part_id=block.name, event="Transporter Loading Start",
                                resource=selected_tp.name, load=block.weight, unit="ton",
                                from_process=transporter_location, to_process=block.location,
                                distance=retrieval_distance, work=block.schedule[block.step].work)
            # 출발 공정의 면적 반환
            current_process_type = self.get_process_type(block.location)
            self.process_dict[current_process_type][block.location].preemp_used.put(block.area)
            self.process_dict[current_process_type][block.location].actual_used.put(block.area)
            self.monitor.record(self.env.now, None, part_id=block.name, resource=block.location,
                                event="Area Released",
                                load=self.process_dict[current_process_type][block.location].area -
                                     self.process_dict[current_process_type][block.location].actual_used.level,
                                unit="m2")

            # 다음 위치로 이동
            yield self.env.timeout(transporting_time)
            self.monitor.record(self.env.now, "Management", part_id=block.name,
                                event="Transporter Loading Complete",
                                resource=selected_tp.name, load=block.weight, unit="ton",
                                from_process=block.location, to_process=next_process,
                                distance=transporting_distance, work=block.schedule[block.step].work)
            # TP, 블록 위치 업데이트
            selected_tp.location = self.inout[next_process][0]

            self.monitor.record(self.env.now, None, part_id=block.name, event=idx,
                                resource=selected_tp.name, load=block.weight, from_process=block.location,
                                to_process=next_process)

            # 블록 위치 업데이트 및 다음 공정에 입고
            block.location = next_process
            next_process_type = self.get_process_type(next_process)
            block.process_type = next_process_type

            # 블록 입고
            if idx == "Process to Stockyard":
                block.in_stock = True
                self.process_dict[next_process_type][next_process].queue.put(block)
                self.monitor.record(self.env.now, next_process, part_id=block.name, event="Stockyard In",
                                    load=self.process_dict[next_process_type][next_process].area -
                                         self.process_dict[next_process_type][next_process].preemp_used.level,
                                    unit='m2')
                self.env.process(self.process_dict[next_process_type][next_process].run())
            elif idx == "Process to Process" or idx == "Stockyard to Process":
                self.process_dict[next_process_type][next_process].queue.put(block)
                self.monitor.record(self.env.now, next_process, part_id=block.name, event="Process In",
                                    load=self.process_dict[next_process_type][next_process].area -
                                         self.process_dict[next_process_type][next_process].preemp_used.level,
                                    unit='m2')
            elif idx == "Child to Parent":
                self.source_dict[block.parent].erect(block)


        self.monitor.record(self.env.now, "Management", part_id=block.name, event="Transporter Released",
                            resource=selected_tp.name)

    def convert_process(self, block, step):
        present_dep = block.schedule[step].process

        if present_dep != 'Sink':
            working_code = list(block.schedule[step].work)  # 현재 공정공종 알파벳 리스트, ex. 'FCFC' -> ['F', 'C', 'F', 'C']
            working_code = sorted(working_code, reverse=True)[0]  # 가장 큰 알파벳 선택 -> 가장 큰 알파벳 기준으로 TP 사용 여부 결정
        else:
            working_code = 'X'  # Sink로 보냄

        present_dep_work = '{0}{1}'.format(working_code,
                                           present_dep) if present_dep != 'Sink' else 'Sink'  # ex) 공정공종 'A' + 부서 '가공소조립부1야드' -> 'A가공소조립부1야드'

        # 이미 변환된 경우 혹은 블록 이동 없는 경우
        if present_dep_work not in self.dep_fac_dict.keys():
            if present_dep in ['도장1부', '도장2부', '발판지원부']:
                converted = self.dispatching_process(block, "not moving", step)
                process_type = "Factory"
            else:
                converted = present_dep

        # 도크인 경우
        elif self.dep_fac_dict[present_dep_work] == "Dock":
            converted = "{0}도크".format(block.dock)

        # 1:1 매칭인 경우
        elif type(self.dep_fac_dict[present_dep_work]) == str:
            converted = self.dep_fac_dict[present_dep_work]

        # 1:다 매칭인 경우 (의장, 도장)
        else:
            if present_dep == "선행도장부":
                process_type = "Painting"
            elif present_dep in ["선행의장부", "의장1부", "의장2부", "의장3부", "기장부"]:
                process_type = "Shelter"
            else:
                process_type = "Factory"

            converted = self.dispatching_process(block, process_type, step)

        block.schedule[step].process = converted

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
                # distance matrix에 점이 존재하고, 현재 잔여 면적이 블록의 면적보다 클 때 distance list에 추가
                if (self.inout[source_location][1] in self.distance_matrix.keys()) and \
                        (self.inout[compared][0] in self.distance_matrix.keys()) and \
                        (self.process_dict[process_type][compared].preemp_used.level >= block.area):
                    distance.append(
                        [compared, self.distance_matrix[self.inout[source_location][1]][self.inout[compared][0]]])

            if len(distance) > 0:
                # 거리 순으로 정렬 후, 가장 가까운 공장으로 배정
                distance = sorted(distance, key=lambda x: x[1])
                next_process = distance[0][0]
                return next_process
            else:
                # 만약 잔여면적이 블록 면적보다 큰 공장이 없을 경우, 가상의 공장으로 배정
                return process_type

    def dispatching_stockyard(self, block, step):
        next_process = self.convert_process(block, step + 1)

        if next_process == "Sink":  # 다음 공정 -> Parent block의 첫 번째 공정
            next_process = self.convert_process(self.block_dict[block.parent], 0)
            next_process_type = self.get_process_type(next_process)

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
            # network matrix에 점이 존재하고, 현재 잔여 면적이 블록의 면적보다 클 때 distance list에 추가
            if (self.inout[stockyard][1] in self.distance_matrix.keys()) and (
                    self.inout[next_process][0] in self.distance_matrix.keys()) and (
                    self.process_dict["Stockyard"][stockyard].preemp_used.level >= block.area):
                distance.append(
                    [stockyard, self.distance_matrix[self.inout[stockyard][1]][self.inout[next_process][0]]])

        if len(distance) > 0:
            # 거리 순으로 정렬 후, 가장 가까운 적치장으로 배정
            distance = sorted(distance, key=lambda x: x[1])
            next_stockyard = distance[0][0]

            # 면적 선점
            self.monitor.record(self.env.now, None, part_id=block.name, event="Area Requested",
                                resource=next_stockyard, process_type="Stockyard",
                                load=self.process_dict['Stockyard'][next_stockyard].preemp_used.level, unit='m2',
                                memo="requested area: {0}".format(block.area))
            yield self.process_dict['Stockyard'][next_stockyard].preemp_used.get(block.area)
            self.monitor.record(self.env.now, None, part_id=block.name, event="Area Preempted",
                                resource=next_stockyard, process_type="Stockyard",
                                load=self.process_dict['Stockyard'][next_stockyard].area -
                                     self.process_dict['Stockyard'][next_stockyard].preemp_used.level, unit='m2')
            return next_stockyard
        else:
            # 만약 잔여면적이 블록 면적보다 큰 적치장이 없을 경우, 가상의 적치장으로 배정
            next_stockyard = "Stockyard"
            # 면적 선점
            self.monitor.record(self.env.now, None, part_id=block.name, event="Area Requested",
                                resource="Stockyard", process_type="Stockyard",
                                load=self.process_dict['Stockyard'][next_stockyard].preemp_used.level, unit='m2',
                                memo="requested area: {0}".format(block.area))
            yield self.process_dict['Stockyard'][next_stockyard].preemp_used.get(block.area)
            self.monitor.record(self.env.now, None, part_id=block.name, event="Area Preempted",
                                resource="Stockyard", process_type="Stockyard",
                                load=self.process_dict['Stockyard'][next_stockyard].area -
                                     self.process_dict['Stockyard'][next_stockyard].preemp_used.level, unit='m2')
            return next_stockyard


# endregion

# region Source
class Source:
    def __init__(self, env, block, process_dict, resource_management, monitor, sink, is_start=False):
        self.env = env
        self.block = block  # Source에서 생성시킬 Block class
        self.process_dict = process_dict  # process/stockyard dictionary / {process_type(Factory, Stockyard, Painting, Shelter): {process_name: process class(), ... } ... }
        self.resource_management = resource_management  # Management Class
        self.monitor = monitor  # Monitor Class
        self.sink = sink

        self.queue = simpy.Store(self.env)
        if is_start:
            self.queue.put(block)
            self.env.process(self.run())

    def run(self):
        block = yield self.queue.get()
        iat = block.schedule[0].start_time - self.env.now
        if iat > 0:
            yield self.env.timeout(block.schedule[0].start_time - self.env.now)
        self.monitor.record(self.env.now, "Source", part_id=block.name, event="Block Created")
        # 첫 번째 공정 결정
        first_process = self.resource_management.convert_process(block, 0)
        # 면적 선점
        first_process_type = self.resource_management.get_process_type(first_process)
        if not block.schedule[0].area_requested and not block.schedule[0].area_preempted:
            self.monitor.record(self.env.now, None, part_id=block.name, event="Area Requested", resource=first_process,
                                load=self.process_dict[first_process_type][first_process].preemp_used.level, unit='m2',
                                memo="requested area: {0}".format(block.area))
            block.schedule[0].area_requested = True
            yield self.process_dict[first_process_type][first_process].preemp_used.get(block.area)
            block.schedule[0].area_preempted = True
            self.monitor.record(self.env.now, None, part_id=block.name, event="Area Preempted", resource=first_process,
                                load=self.process_dict[first_process_type][first_process].area -
                                     self.process_dict[first_process_type][first_process].preemp_used.level, unit='m2')
            for event in block.schedule[0].area_preempted_waiting_list:  # 면적 선점 기다린 이벤트가 있다면
                event.succeed()
        elif block.schedule[0].area_requested and not block.schedule[0].area_preempted:
            waiting_event = self.env.event()
            block.schedule[0].area_preempted_waiting_list.append(waiting_event)
            yield waiting_event

        # 면적 선점이 끝나면
        self.process_dict[first_process_type][first_process].queue.put(block)
        self.monitor.record(self.env.now, first_process, part_id=block.name, event="Process In",
                            load=self.process_dict[first_process_type][first_process].area -
                                 self.process_dict[first_process_type][first_process].actual_used.level,
                            unit='m2')


    def erect(self, child_block):
        self.block.finished_child += 1
        self.monitor.record(self.env.now, None, part_id=self.block.name, event="Child Block Arrived",
                            memo="{0} is arrived, {1}/{2}".format(child_block.name, self.block.finished_child,
                                                                  len(self.block.child)))
        if self.block.finished_child == 1:
            self.queue.put(self.block)
            self.env.process(self.run())
        if self.block.finished_child == len(self.block.child):
            self.block.is_finished_first_process = True
            if self.block.wait_idx:
                self.block.wait_event.succeed()

        self.sink.terminate(child_block)


# endregion

# region Process
class Factory:
    def __init__(self, env, name, process_type, area, block_dict, process_dict, source_dict, resource_management, sink,
                 monitor, stock_lag=2):
        self.env = env
        self.name = name
        self.process_type = process_type
        self.area = area

        self.preemp_used = simpy.Container(env, capacity=area, init=area)  # 선점 면적
        self.actual_used = simpy.Container(env, capacity=area, init=area)  # 실제 사용 면적

        self.block_dict = block_dict  # {"A0001_A11A0": Block class, ... }
        self.process_dict = process_dict  # process/stockyard dictionary / {process_type(Factory, Stockyard, Painting, Shelter): {process_name: process class(), ... } ... }
        self.source_dict = source_dict  # {"A0001_A11A0": Source class, ... }
        self.resource_management = resource_management  # Management Class
        self.sink = sink  # Sink Class
        self.monitor = monitor  # Monitor Class
        self.stock_lag = stock_lag

        self.queue = simpy.Store(self.env)  # 작업 대기열

        self.env.process(self.run())

    def run(self):
        while True:
            block = yield self.queue.get()
            block.location = self.name
            block.process_type = self.process_type
            block.in_stock = False
            block.working_code = block.schedule[block.step].work
            area = yield self.actual_used.get(block.area)
            self.env.process(self.work(block))

    def work(self, block):
        self.monitor.record(self.env.now, self.name, part_id=block.name, event="Work Start",
                            process_type=self.process_type, load=self.area - self.actual_used.level, unit="m2",
                            work=block.schedule[block.step].work)
        # 작업 시간만큼 timeout
        yield self.env.timeout(block.schedule[block.step].process_time)

        if not block.is_finished_first_process:
            block.wait_idx = True
            self.monitor.record(self.env.now, self.name, part_id=block.name, event="Wait for Child Blocks")
            yield block.wait_event

        self.monitor.record(self.env.now, self.name, part_id=block.name, event="Work Finish",
                            process_type=self.process_type, work=block.schedule[block.step].work)
        self.env.process(self.to_next_process(block))

    def to_next_process(self, block):
        idx = None

        if block.schedule[block.step + 1].process != "Sink":  # 다음 opearation이 남아있는 경우
            next_schedule = block.schedule[block.step + 1]
            idx = "Process to Process"
        elif block.schedule[block.step + 1].process == "Sink" and block.parent is not None:  # 해당 블록의 모든 Operation이 끝나고, 상위 블록으로 조립될 경우
            next_schedule = self.block_dict[block.parent].schedule[0]
            idx = "Child to Parent"
        else:  # 다음 operation도 없고, 상위 블록도 없는 경우 -> 바로 Sink로 이동
            next_schedule = block.schedule[block.step + 1]
            idx = "To Sink"

        next_start_time = next_schedule.start_time if next_schedule.process != "Sink" else 0
        lag_time = next_start_time - math.floor(self.env.now)  # Transporter 이동 등으로 생긴 소수점 시간 무시

        # 다음 목적지 결정
        next_destination = None
        if idx == "To Sink":  # 바로 Sink로 갈 경우
            next_destination = "Sink"
        elif lag_time < self.stock_lag:  # 바로 다음 Process로 이동할 경우
            # 다음 일정으로 갈 경우: 다음 공정 결정 후 면적 선점
            if idx == "Process to Process":
                if next_start_time - self.env.now >= 0:  # 다음 시작시간까지 stay
                    yield self.env.timeout(next_start_time - self.env.now)
                next_destination = self.resource_management.convert_process(block, block.step + 1)
                next_process_type = self.resource_management.get_process_type(next_destination)

                # 면적 선점
                if not block.schedule[block.step + 1].area_requested and not block.schedule[block.step + 1].area_preempted:
                    self.monitor.record(self.env.now, None, part_id=block.name, event="Area Requested",
                                        resource=next_destination,
                                        load=self.process_dict[next_process_type][next_destination].preemp_used.level,
                                        unit="m2", memo="requested area: {0}".format(block.area))
                    block.schedule[block.step + 1].area_requested = True
                    yield self.process_dict[next_process_type][next_destination].preemp_used.get(block.area)
                    block.schedule[block.step + 1].area_preempted = True
                    self.monitor.record(self.env.now, None, part_id=block.name, event="Area Preempted",
                                        process_type=next_process_type, resource=next_destination,
                                        load=self.process_dict[next_process_type][next_destination].area -
                                             self.process_dict[next_process_type][next_destination].preemp_used.level,
                                        unit="m2")
                    for event in block.schedule[block.step + 1].area_preempted_waiting_list:
                        event.succeed()
                elif block.schedule[block.step + 1].area_requested and not block.schedule[block.step + 1].area_preempted:
                    waiting_event = self.env.event()
                    block.schedule[block.step + 1].area_preempted_waiting_list.append(waiting_event)
                    yield waiting_event

            # Parent 블록으로 갈 경우: Parent 블록의 첫 번째 공정으로 이동 후 Parent 블록의 크기만큼 면적 선점
            elif idx == "Child to Parent":
                if self.block_dict[block.parent].finished_child == 0:
                    if next_start_time - self.env.now >= 0:  # 다음 시작시간까지 stay
                        yield self.env.timeout(next_start_time - self.env.now)
                    next_destination = self.resource_management.convert_process(self.block_dict[block.parent], 0)
                    next_process_type = self.resource_management.get_process_type(next_destination)

                    if not self.block_dict[block.parent].schedule[0].area_requested and not self.block_dict[block.parent].schedule[0].area_preempted:
                        self.monitor.record(self.env.now, None, part_id=block.parent, resource=next_destination,
                                            event="Area Requested",
                                            load=self.process_dict[next_process_type][next_destination].preemp_used.level,
                                            unit="m2",
                                            memo="requested area: {0}".format(self.block_dict[block.parent].area))
                        self.block_dict[block.parent].schedule[0].area_requested = True
                        # Parent 블록 크기만큼 면적 선점
                        yield self.process_dict[next_process_type][next_destination].preemp_used.get(
                            self.block_dict[block.parent].area)
                        self.block_dict[block.parent].schedule[0].area_preempted = True
                        self.monitor.record(self.env.now, None, part_id=block.parent, resource=next_destination,
                                            event="Area Preempted", process_type=next_process_type,
                                            load=self.process_dict[next_process_type][next_destination].area -
                                                 self.process_dict[next_process_type][next_destination].preemp_used.level,
                                            unit="m2")
                        for event in self.block_dict[block.parent].schedule[0].area_preempted_waiting_list:
                            event.succeed()
                    elif self.block_dict[block.parent].schedule[0].area_requested and not self.block_dict[block.parent].schedule[0].area_preempted:
                        waiting_event = self.env.event()
                        self.block_dict[block.parent].schedule[0].area_preempted_waiting_list.append(waiting_event)
                        yield waiting_event
                else:
                    next_destination = self.resource_management.convert_process(self.block_dict[block.parent], 0)
        elif lag_time >= self.stock_lag and idx != "To Sink":  # Stockyard로 이동해야 할 경우
            # Parent Block으로 가고, 이미 Parent 블록이 생성되어 있다면 -> 적치장 안 들리고 바로 Parent 블록으로 이동
            if idx == "Child to Parent" and self.block_dict[block.parent].finished_child > 1:
                next_destination = self.resource_management.convert_process(self.block_dict[block.parent], 0)
                next_process_type = self.resource_management.get_process_type(next_destination)
            else:
                idx = "Process to Stockyard"
                # Management Class의 dispatching_stockyard에서 알아서 다음 Operation인지 아니면 Parent 블록으로 가는 지 구별
                next_destination = yield self.env.process(self.resource_management.dispatching_stockyard(block, block.step))

        block.step += 1
        # 다음 목적지로 이동
        if next_destination == "Sink": # Sink로 이동
            # 면적 반환
            self.preemp_used.put(block.area)
            self.actual_used.put(block.area)
            self.monitor.record(self.env.now, None, part_id=block.name, resource=self.name, event="Area Released",
                                load=self.area - self.actual_used.level, unit='m2')
            self.sink.terminate(block)
        else:
            # Transporter를 사용해야 할 경우
            if any(code in ['F', 'G', 'H', 'J', 'M', 'K', 'L', 'N'] for code in list(block.schedule[block.step - 1].work)):
                # Transporter로 이동
                self.env.process(self.resource_management.assign_transporter(block, next_destination, idx=idx))

            # Transporter를 사용하지 않는 경우
            else:
                its_process_type = self.resource_management.get_process_type(next_destination)
                # 면적 반환
                self.preemp_used.put(block.area)
                self.actual_used.put(block.area)
                self.monitor.record(self.env.now, None, part_id=block.name, resource=self.name, event="Area Released",
                                    load=self.area - self.actual_used.level, unit='m2')

                self.monitor.record(self.env.now, None, part_id=block.name, event=idx, from_process=self.name,
                                    to_process=next_destination)

                if idx == "Process to Process":  # 바로 다음 공정의 대기열로 이동
                    self.process_dict[its_process_type][next_destination].queue.put(block)
                elif idx == "Child to Parent":  # Parent 블록으로 흡수
                    self.source_dict[block.parent].erect(block)
                elif idx == "Process to Stockyard":
                    block.in_stock = True
                    self.process_dict[its_process_type][next_destination].queue.put(block)
                    self.env.process(self.process_dict[its_process_type][next_destination].run())



class Stockyard:
    def __init__(self, env, name, area, process_dict, block_dict, source_dict, resource_management, monitor):
        self.env = env
        self.name = name
        self.area = area
        self.process_dict = process_dict
        self.block_dict = block_dict
        self.source_dict = source_dict
        self.resource_management = resource_management
        self.monitor = monitor

        self.preemp_used = simpy.Container(env, capacity=area, init=area)  # 선점 면적
        self.actual_used = simpy.Container(env, capacity=area, init=area)  # 실제 사용 면적

        self.queue = simpy.Store(self.env)

    def run(self):
        block = yield self.queue.get()
        idx = None
        if block.schedule[block.step].process == 'Sink' and block.parent is not None:
            idx = "Child to Parent"
            next_start_time = self.block_dict[block.parent].schedule[0].start_time
            next_process = self.block_dict[block.parent].schedule[0].process
        elif block.schedule[block.step].process != "Sink":
            idx = "Stockyard to Process"
            next_start_time = block.schedule[block.step].start_time
            next_process = block.schedule[block.step].process
            next_process_type = self.resource_management.get_process_type(next_process)
            if not block.schedule[block.step].area_requested and not block.schedule[block.step].area_preempted:
                self.monitor.record(self.env.now, None, part_id=block.name, event="Area Requested", resource=next_process,
                                    load=self.process_dict[next_process_type][next_process].preemp_used.level, unit="m2",
                                    memo="requested area: {0}".format(block.area))
                block.schedule[block.step].area_requested = True
                yield self.process_dict[next_process_type][next_process].preemp_used.get(block.area)
                block.schedule[block.step].area_preempted = True
                self.monitor.record(self.env.now, None, part_id=block.name, event="Area Preempted", resource=next_process,
                                    load=self.process_dict[next_process_type][next_process].area -
                                         self.process_dict[next_process_type][next_process].preemp_used.level, unit="m2")
                for event in block.schedule[block.step].area_preempted_waiting_list:
                    event.succeed()
            elif block.schedule[block.step].area_requested and not block.schedule[block.step].area_preempted:
                waiting_event = self.env.event()
                block.schedule[block.step].area_preempted_waiting_list.append(waiting_event)
                yield waiting_event
        else:
            print("line 606")

        waiting_time = max(next_start_time - self.env.now, 0)

        yield self.env.timeout(waiting_time)

        # 이전 단계부터 이미 Transporter를 사용해야 하는 단계였다면 Transporter 사용
        if any(code in ['F', 'G', 'H', 'J', 'M', 'K', 'L', 'N'] for code in list(block.schedule[block.step - 1].work)):
            self.monitor.record(self.env.now, self.name, part_id=block.name, event="Stockyard Out",
                                load=self.area - self.actual_used.level, unit='m2')
            self.env.process(self.resource_management.assign_transporter(block, next_process, idx=idx))
        else:
            self.preemp_used.put(block.area)
            self.actual_used.put(block.area)
            self.monitor.record(self.env.now, None, resource=self.name, part_id=block.name, event="Area Released",
                                load=self.area - self.actual_used.level, unit='m2')
            next_process_type = self.resource_management.get_process_type(next_process)

            self.monitor.record(self.env.now, None, part_id=block.name, event=idx,
                                from_process=self.name, to_process=next_process)

            if idx == "Stockyard to Process":
                self.process_dict[next_process_type][next_process].queue.put(block)
            elif idx == "Child to Parent":
                self.source_dict[block.parent].erect(block)
            block.in_stock = False


# endregion

# region Sink
class Sink:
    def __init__(self, env, monitor, tot_num):
        self.env = env
        self.monitor = monitor
        self.tot_num = tot_num

        self.finished = 0

    def terminate(self, block):
        self.finished += 1
        self.monitor.record(self.env.now, "Sink", part_id=block.name, event="Block Completed")

        print("{0} is Finished at {1}, {2}% is finished".format(block.name, self.env.now,
                                                                round(100 * self.finished / self.tot_num)))


class Monitor:
    def __init__(self, project_name, initial_date):
        self.project_name = project_name
        self.initial_date = initial_date

        self.time = list()
        self.date = list()
        self.event = list()
        self.part_id = list()
        self.process = list()
        self.resource = list()
        self.memo = list()
        self.process_type = list()
        self.load = list()
        self.unit = list()
        self.from_process = list()
        self.to_process = list()
        self.distance = list()
        self.work = list()

        self.created = 0
        self.completed = 0
        self.tp_used = dict()
        self.road_used = dict()

    def record(self, time, process, part_id=None, event=None, resource=None, memo=None, process_type=None,
               load=None, unit=None, from_process=None, to_process=None, distance=None, work=None):
        self.time.append(time)
        date_time = self.initial_date + timedelta(days=math.floor(time))
        self.date.append(date_time.date())
        self.event.append(event)
        self.part_id.append(part_id)
        self.process.append(process)
        self.resource.append(resource)
        self.memo.append(memo)
        self.process_type.append(process_type)
        self.load.append(load)
        self.unit.append(unit)
        self.from_process.append(from_process)
        self.to_process.append(to_process)
        self.distance.append(distance)
        self.work.append(work)

        if event == 'Block Created':
            self.created += 1
        elif event == 'Block Completed':
            self.completed += 1

    def get_logs(self, file_path):
        # 1. event_tracer
        event_tracer = pd.DataFrame()
        event_tracer['Date'] = self.date
        event_tracer['Part'] = self.part_id
        event_tracer['Process'] = self.process
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
        event_tracer['Simulation time'] = self.time

        path_event_tracer = file_path + 'result_{0}.csv'.format(self.project_name)

        event_tracer.to_csv(path_event_tracer, encoding='utf-8-sig')

        return path_event_tracer

