import copy
import math
import numpy as np
from agent import Agent
from tools.utility import get_central_vertices
import pickle
import pandas as pd
import xlsxwriter
import scipy.io
import os
import matplotlib.pyplot as plt
from NDS_analysis import analyze_ipv_in_nds


class Scenario:
    def __init__(self, pos, vel, heading, ipv):
        self.position = {'lt': np.array(pos[0]), 'gs': np.array(pos[1])}
        self.velocity = {'lt': np.array(vel[0]), 'gs': np.array(vel[1])}
        self.heading = {'lt': np.array(heading[0]), 'gs': np.array(heading[1])}
        self.ipv = {'lt': np.array(ipv[0]), 'gs': np.array(ipv[1])}


class Simulator:
    def __init__(self, version):
        self.semantic_result = None
        self.version = version
        self.output_directory = None
        self.tag = None
        self.case_id = None
        self.scenario = None
        self.agent_lt = None
        self.agent_gs = None
        self.num_step = 0
        self.ending_point = None
        self.gs_actual_trj = []
        self.lt_actual_trj = []

    def initialize(self, scenario, case_tag):
        self.scenario = scenario
        self.agent_lt = Agent(scenario.position['lt'], scenario.velocity['lt'], scenario.heading['lt'], 'lt')
        self.agent_gs = Agent(scenario.position['gs'], scenario.velocity['gs'], scenario.heading['gs'], 'gs')
        self.agent_lt.estimated_inter_agent = copy.deepcopy(self.agent_gs)
        self.agent_gs.estimated_inter_agent = copy.deepcopy(self.agent_lt)
        self.agent_lt.ipv = self.scenario.ipv['lt']
        self.agent_gs.ipv = self.scenario.ipv['gs']
        self.tag = case_tag

    def ibr_iteration(self, num_step=30, lt_controller_type='VGIM', break_when_finish=False):
        self.num_step = num_step
        iter_limit = 3
        for t in range(self.num_step):
            print('time_step: ', t, '/', self.num_step)

            "==plan for left-turn=="
            if lt_controller_type in {'VGIM-coop', 'VGIM-dyna', 'VGIM'}:

                # ==interaction with parallel virtual agents
                self.agent_lt.interact_with_parallel_virtual_agents(self.agent_gs, iter_limit=iter_limit)

                # ==interaction with estimated agent
                self.agent_lt.interact_with_estimated_agents(iter_limit=iter_limit)

            elif lt_controller_type in {'OPT-coop', 'OPT-dyna', 'OPT-safe'}:

                # ==interaction with estimated agent
                self.agent_lt.interact_with_estimated_agents(controller_type=lt_controller_type)

            "==plan for go straight=="
            # ==interaction with parallel virtual agents
            self.agent_gs.interact_with_parallel_virtual_agents(self.agent_lt, iter_limit)

            # ==interaction with estimated agent
            self.agent_gs.interact_with_estimated_agents(iter_limit)

            "==update state=="
            self.agent_lt.update_state(self.agent_gs, controller_type=lt_controller_type)
            self.agent_gs.update_state(self.agent_lt, controller_type='VGIM')

            if break_when_finish:
                if self.agent_gs.observed_trajectory[-1, 0] < self.agent_lt.observed_trajectory[-1, 0] \
                        or self.agent_lt.observed_trajectory[-1, 1] > self.agent_gs.observed_trajectory[-1, 1]:
                    self.num_step = t + 1
                    break

    def save_data(self, print_semantic_result=False, task_id=1):
        filename = self.output_directory + '/data/' + str(self.tag) \
                   + '_task_' + str(task_id) \
                   + '_case_' + str(self.case_id) \
                   + '.pckl'
        f = open(filename, 'wb')
        pickle.dump([self.agent_lt, self.agent_gs, self.semantic_result, self.tag, self.ending_point], f)
        f.close()
        print('case_' + str(self.tag), ' saved')

        if print_semantic_result:

            # prepare workbook
            workbook = self.output_directory + '/excel/' + self.tag + '.xlsx'
            if not os.path.exists(workbook):
                wb = xlsxwriter.Workbook(workbook)
                ws = wb.add_worksheet(self.tag + '-task-' + str(task_id))
                wb.close()

            # prepare data
            data_interaction_event = [[self.agent_gs.observed_trajectory[0, 0],
                                       self.agent_gs.ipv,
                                       self.semantic_result], ]
            if self.case_id == 0:
                pd_interaction_event = pd.DataFrame(data_interaction_event, columns=['gap', 'gs_ipv', 'result'])
                # write data
                with pd.ExcelWriter(workbook, mode='a', if_sheet_exists="overlay", engine="openpyxl") as writer:

                    pd_interaction_event.to_excel(writer, index=False,
                                                  sheet_name=self.tag + '-task-' + str(task_id),
                                                  startrow=self.case_id)

            else:
                pd_interaction_event = pd.DataFrame(data_interaction_event)
                # write data
                with pd.ExcelWriter(workbook, mode='a', if_sheet_exists="overlay", engine="openpyxl") as writer:

                    pd_interaction_event.to_excel(writer, index=False, header=False,
                                                  sheet_name=self.tag + '-task-' + str(task_id),
                                                  startrow=self.case_id + 1)

    def post_process(self):
        """
        identify semantic interaction results:
        1. crashed or not
        2. the letf-turn vehicle yield or not
        :return:
        """
        track_lt = self.agent_lt.observed_trajectory
        track_gs = self.agent_gs.observed_trajectory
        pos_delta = track_gs - track_lt
        dis_delta = np.linalg.norm(pos_delta[:, 0:2], axis=1)

        if min(dis_delta) < 1:
            self.semantic_result = 'crashed'
            print('interaction is crashed. \n')
        else:
            pos_x_smaller = pos_delta[pos_delta[:, 0] < 0]
            if np.size(pos_x_smaller, 0):

                "whether the LT vehicle yield"
                pos_y_larger = pos_x_smaller[pos_x_smaller[:, 1] > 0]
                yield_points = np.size(pos_y_larger, 0)
                if yield_points:
                    self.semantic_result = 'yield'

                    "where the interaction finish"
                    ind_coll = np.where(pos_y_larger[0, 0] == pos_delta[:, 0])
                    ind = ind_coll[0] - 1
                    self.ending_point = {'lt': self.agent_lt.observed_trajectory[ind, :],
                                         'gs': self.agent_gs.observed_trajectory[ind, :]}

                    print('LT vehicle yielded. \n')
                    # print('interaction finished at No.' + str(ind + 1) + ' frame\n')
                    # print('GS info:' + str(self.ending_point['gs']) + '\n')
                    # print('LT info:' + str(self.ending_point['lt']) + '\n')
                    # print('px py vx vy heading')

                else:
                    self.semantic_result = 'rush'
                    print('LT vehicle rushed. \n')
            else:
                pos_y_smaller = pos_delta[pos_delta[:, 1] < 0]
                if np.size(pos_y_smaller, 0):
                    self.semantic_result = 'rush'
                    print('LT vehicle rushed. \n')
                else:
                    self.semantic_result = 'unfinished'
                    print('interaction is not finished. \n')

    def visualize(self, task_id=0, controller_type='VGIM'):
        """

        :param task_id: used whe conduct multi processed in parallel
        :param controller_type:
        :return:
        """
        # set figures
        fig = plt.figure(figsize=(12, 4))
        fig.suptitle('case_' + str(self.tag))
        ax1 = fig.add_subplot(131, title='trajectory_LT_' + self.semantic_result)

        if self.tag == 'nds-simu':  # nds simulation case
            ax1.set(xlim=[-22, 53], ylim=[-31, 57])
            img = plt.imread('./background_pic/Jianhexianxia.jpg')
            ax1.imshow(img, extent=[-22, 53, -31, 57])
            lt_origin_point = self.agent_lt.observed_trajectory[0, 0:2]
            gs_origin_point = self.agent_gs.observed_trajectory[0, 0:2]
            cv_it, _ = get_central_vertices('lt_nds', origin_point=lt_origin_point)
            cv_gs, _ = get_central_vertices('gs_nds', origin_point=gs_origin_point)
        else:  # simulation case
            cv_it, _ = get_central_vertices('lt')
            cv_gs, _ = get_central_vertices('gs')
            max_x_lt = max(self.agent_lt.observed_trajectory[:, 0])
            max_y_lt = max(self.agent_lt.observed_trajectory[:, 1])
            max_x_gs = max(self.agent_gs.observed_trajectory[:, 0])
            max_y_gs = max(self.agent_gs.observed_trajectory[:, 1])
            max_x = max(max_x_lt, max_x_gs)
            max_y = max(max_y_lt, max_y_gs)

            min_x_lt = min(self.agent_lt.observed_trajectory[:, 0])
            min_y_lt = min(self.agent_lt.observed_trajectory[:, 1])
            min_x_gs = min(self.agent_gs.observed_trajectory[:, 0])
            min_y_gs = min(self.agent_gs.observed_trajectory[:, 1])
            min_x = min(min_x_lt, min_x_gs)
            min_y = min(min_y_lt, min_y_gs)

            ax1.set(xlim=[min_x - 3, max_x + 3], ylim=[min_y - 3, max_y + 3])

        "====show plans at each time step===="
        # central vertices
        ax1.plot(cv_it[:, 0], cv_it[:, 1], 'r-')
        ax1.plot(cv_gs[:, 0], cv_gs[:, 1], 'b-')

        # position at each time step
        ax1.scatter(self.agent_lt.observed_trajectory[:, 0],
                    self.agent_lt.observed_trajectory[:, 1],
                    s=80,
                    alpha=0.6,
                    color='red',
                    label='left-turn')
        ax1.scatter(self.agent_gs.observed_trajectory[:, 0],
                    self.agent_gs.observed_trajectory[:, 1],
                    s=80,
                    alpha=0.6,
                    color='blue',
                    label='go-straight')
        if self.tag == 'nds-simu':
            ax1.scatter(self.lt_actual_trj[:self.num_step, 0],
                        self.lt_actual_trj[:self.num_step, 1],
                        s=50,
                        alpha=0.6,
                        color='purple',
                        label='left-turn-actual')
            ax1.scatter(self.gs_actual_trj[:self.num_step, 0],
                        self.gs_actual_trj[:self.num_step, 1],
                        s=50,
                        alpha=0.6,
                        color='black',
                        label='go-straight-actual')

        # full tracks at each time step
        for t in range(self.num_step):
            lt_track = self.agent_lt.trj_solution_collection[t]
            ax1.plot(lt_track[:, 0], lt_track[:, 1], '--', color='red')
            gs_track = self.agent_gs.trj_solution_collection[t]
            ax1.plot(gs_track[:, 0], gs_track[:, 1], '--', color='blue')

        # connect two agents
        for t in range(self.num_step + 1):
            ax1.plot([self.agent_lt.observed_trajectory[t, 0], self.agent_gs.observed_trajectory[t, 0]],
                     [self.agent_lt.observed_trajectory[t, 1], self.agent_gs.observed_trajectory[t, 1]],
                     color='black',
                     alpha=0.2)

        "====show IPV and uncertainty===="
        ax2 = fig.add_subplot(132, title='ipv')
        x_range = np.array(range(len(self.agent_gs.estimated_inter_agent.ipv_collection)))

        # actual ipv
        ax2.plot(x_range, self.agent_lt.ipv * np.ones_like(x_range),
                 color='red',
                 linewidth=5,
                 label='actual lt IPV')
        ax2.plot(x_range, self.agent_gs.ipv * np.ones_like(x_range),
                 color='blue',
                 linewidth=5,
                 label='actual gs IPV')

        # estimated ipv
        if controller_type in {'VGIM-coop', 'VGIM-dyna', 'VGIM'}:
            y_gs = np.array(self.agent_lt.estimated_inter_agent.ipv_collection)
            ax2.plot(x_range, y_gs, color='blue', label='estimated gs IPV')
            # error bar
            y_error_gs = np.array(self.agent_gs.estimated_inter_agent.ipv_error_collection)
            ax2.fill_between(x_range, y_gs - y_error_gs, y_gs + y_error_gs,
                             alpha=0.3,
                             color='blue',
                             label='lt IPV error')

        y_lt = np.array(self.agent_gs.estimated_inter_agent.ipv_collection)
        ax2.plot(x_range, y_lt, color='red', label='estimated lt IPV')
        # error bar
        y_error_lt = np.array(self.agent_gs.estimated_inter_agent.ipv_error_collection)
        ax2.fill_between(x_range, y_lt - y_error_lt, y_lt + y_error_lt,
                         alpha=0.3,
                         color='red',
                         label='gs IPV error')

        "====show velocity===="
        ax3 = fig.add_subplot(133, title='velocity')
        x_range = np.array(range(np.size(self.agent_lt.observed_trajectory, 0)))
        vel_norm_lt = np.linalg.norm(self.agent_lt.observed_trajectory[:, 2:4], axis=1)
        vel_norm_gs = np.linalg.norm(self.agent_gs.observed_trajectory[:, 2:4], axis=1)
        ax3.plot(x_range, vel_norm_lt, color='red', label='LT velocity')
        ax3.plot(x_range, vel_norm_gs, color='blue', label='FC velocity')

        ax1.legend()
        ax2.legend()
        ax3.legend()

        # plt.ioff()
        plt.savefig(self.output_directory + '/figures/' + str(self.tag)
                    + '_task_' + str(task_id)
                    + '_case_' + str(self.case_id)
                    + '.svg', format='svg')

        plt.pause(1)
        plt.close('all')
        # plt.show()

    def read_nds_scenario(self):
        cross_id, data_cross, _ = analyze_ipv_in_nds(self.case_id)
        # data_cross:
        # 0-ipv_lt | ipv_lt_error | lt_px | lt_py  | lt_vx  | lt_vy  | lt_heading  |...
        # 7-ipv_gs | ipv_gs_error | gs_px | gs_py  | gs_vx  | gs_vy  | gs_heading  |

        if cross_id == -1:
            return None
        else:
            init_position_lt = [data_cross[0, 2], data_cross[0, 3]]
            init_velocity_lt = [data_cross[0, 4], data_cross[0, 5]]
            init_heading_lt = data_cross[0, 6]
            ipv_lt = np.mean(data_cross[4:, 0])
            init_position_gs = [data_cross[0, 9], data_cross[0, 10]]
            init_velocity_gs = [data_cross[0, 11], data_cross[0, 12]]
            init_heading_gs = data_cross[0, 13]
            ipv_gs = np.mean(data_cross[4:, 7])
            self.lt_actual_trj = data_cross[:, 2:4]
            self.gs_actual_trj = data_cross[:, 9:11]

            return Scenario([init_position_lt, init_position_gs],
                            [init_velocity_lt, init_velocity_gs],
                            [init_heading_lt, init_heading_gs],
                            [ipv_lt, ipv_gs])


def main1():
    """
    ==== main for simulating unprotected left-turning ====

    1. model type is controlled by ipv and iteration number (VGIM: 3 iterations , Optimal controller: 0)

    2. manual Continuous Interaction: if LT yielded, print the ending point of the interaction and
    the next interaction starts at the ending point  (used for setting LT vehicle's initial state)

    3. change FC vehicle' ipv and initial state, and scenario tag for each simulation

    4. simulation results are saved in simulation/version28

    5. **** check TARGET in agent.py: TARGET = 'simulation' ****
    :return:
    """

    tag = 'NE-Coop'  # TODO
    controller_type = 'VGIM'

    # initial state of the left-turn vehicle
    init_position_lt = [11, -5.8]
    init_velocity_lt = [1.5, 0.3]
    init_heading_lt = math.pi / 4
    ipv_lt = 0  # TODO
    # initial state of the go-straight vehicle
    init_position_gs = [22, -2]  # TODO
    init_velocity_gs = [-1.5, 0]
    init_heading_gs = math.pi
    ipv_gs = 0  # TODO

    simu_scenario = Scenario([init_position_lt, init_position_gs],
                             [init_velocity_lt, init_velocity_gs],
                             [init_heading_lt, init_heading_gs],
                             [ipv_lt, ipv_gs])

    simu = Simulator(34)
    simu.initialize(simu_scenario, tag)

    simu.ibr_iteration(lt_controller_type=controller_type, num_step=1)
    simu.post_process()
    # simu.save_data(print_semantic_result=False)
    simu.visualize()


def main2():
    """
    ==== main for simulating RANDOM* unprotected left-turning ====

    1. model type is controlled by ipv and iteration number (VGIM: 3 iterations , Optimal controller: 0)

    2. change tag for each simulation

    3. simulation results are saved in simulation/version29

    4*. straight-through traffic are endless and generated with random ipv and gap

    5. **** check TARGET in agent.py: TARGET = 'simulation' ****
    :return:
    """
    for case_id in range(100):

        task_id = 3

        # controller_tag = 'VGIM-coop'
        # controller_tag = 'VGIM-dyna'
        # controller_tag = 'OPT-coop'
        # controller_tag = 'OPT-safe'
        controller_tag = 'OPT-dyna'

        # generate gs position
        init_gs_px = 2 * (2 * (np.random.random() - 0.5)) + 25
        # init_gs_px = 26

        # initial state of the go-straight vehicle
        init_position_gs = [init_gs_px, -2]
        init_velocity_gs = [-5, 0]
        init_heading_gs = math.pi
        ipv_gs = math.pi * 1 / 4 * (2 * (np.random.random() - 0.5))
        # ipv_gs = -0.5 * math.pi/4

        # initial state of the left-turn vehicle
        init_position_lt = [11, -5.8]
        init_velocity_lt = [1.5, 1]
        init_heading_lt = math.pi / 4
        if controller_tag in {'VGIM-coop', 'OPT-coop'}:
            ipv_lt = math.pi / 8
        elif controller_tag in {'OPT-safe'}:
            ipv_lt = 3 * math.pi / 16
        else:
            if init_gs_px > 35 and ipv_gs > 3 / 16 * math.pi:
                ipv_lt = -0.1
            else:
                ipv_lt = math.pi / 8

        simu_scenario = Scenario([init_position_lt, init_position_gs],
                                 [init_velocity_lt, init_velocity_gs],
                                 [init_heading_lt, init_heading_gs],
                                 [ipv_lt, ipv_gs])
        simu = Simulator(36)
        simu.output_directory = '../data/3_parallel_game_outputs/simulation/version' + str(simu.version)
        simu.case_id = case_id

        print('==== start main for random interaction ====')
        print('task type: ', controller_tag)
        print('task id: ' + str(task_id))
        print('case id: ' + str(case_id))
        print('gs_px: ', init_gs_px)
        print('gs_ipv: ', ipv_gs)

        simu.initialize(simu_scenario, controller_tag)

        simu.ibr_iteration(lt_controller_type=controller_tag, num_step=30, break_when_finish=True)
        simu.post_process()
        simu.save_data(print_semantic_result=True, task_id=task_id)
        simu.visualize(task_id=task_id, controller_type=controller_tag)


def main3():
    """
    ==== main for simulating NDS scenarios ====

    1. model type is controlled by ipv and iteration number (VGIM: 3 iterations , Optimal controller: 0)

    2. simulation results are saved in NDS_simulation

    3.**** check TARGET in agent.py: TARGET = 'nds simulation' ****

    :return:
    """
    for case_id in range(51, 52):
        # case_id = 0
        task_id = 6  # change task id if conduct multi tasks
        simulation_version = 3
        tag = 'nds-simu'

        simu = Simulator(simulation_version)
        simu.output_directory = '../data/3_parallel_game_outputs/NDS_simulation/version' + str(simulation_version)
        simu.case_id = case_id

        print('==== start main for NDS simulation ====')
        print('task type: ', tag)
        print('case id: ' + str(case_id))

        simu_scenario = simu.read_nds_scenario()
        simu_scenario.ipv['lt'] += 0
        simu_scenario.ipv['gs'] -= 0
        if simu_scenario:
            simu.initialize(simu_scenario, tag)
            simu.agent_gs.target = 'gs_nds'
            simu.agent_lt.target = 'lt_nds'
            simu.ibr_iteration(num_step=30)
            simu.post_process()
            simu.save_data(task_id=task_id)
            simu.visualize(task_id=task_id)
        else:
            continue


if __name__ == '__main__':
    "无保护左转实验- 多模型对比"
    # main1()

    "无保护左转实验- 随机交互"
    main2()

    "剑河仙霞场景仿真"
    # main3()
