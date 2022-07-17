import copy
import math
import numpy as np
from agent import Agent
from tools.utility import draw_rectangle
import matplotlib.pyplot as plt
from NDS_analysis import analyze_ipv_in_nds


class Scenario:
    def __init__(self, pos, vel, heading, ipv, conl_type):
        self.position = {'lt': np.array(pos[0]), 'gs': np.array(pos[1])}
        self.velocity = {'lt': np.array(vel[0]), 'gs': np.array(vel[1])}
        self.heading = {'lt': np.array(heading[0]), 'gs': np.array(heading[1])}
        self.ipv = {'lt': np.array(ipv[0]), 'gs': np.array(ipv[1])}
        self.conl_type = {'lt': conl_type[0], 'gs': conl_type[1]}


class Simulator:
    def __init__(self, case_id=None):
        self.semantic_result = None
        self.output_directory = None
        self.tag = None
        self.case_id = case_id
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
        self.agent_lt.conl_type = self.scenario.conl_type['lt']
        self.agent_gs.conl_type = self.scenario.conl_type['gs']
        self.tag = case_tag

    def interact(self, simu_step=30, break_when_finish=False):
        self.num_step = simu_step
        iter_limit = 10
        for t in range(self.num_step):
            print('time_step: ', t, '/', self.num_step)

            "==plan for left-turn=="
            if self.agent_lt.conl_type in {'gt'}:

                # ==interaction with parallel virtual agents
                self.agent_lt.ibr_interact_with_virtual_agents(self.agent_gs)

                # ==interaction with estimated agent
                self.agent_lt.ibr_interact(iter_limit=iter_limit)

            elif self.agent_lt.conl_type in {'opt'}:
                self.agent_lt.opt_plan()

            "==plan for go straight=="
            if self.agent_gs.conl_type in {'gt'}:
                # ==interaction with parallel virtual agents
                self.agent_gs.ibr_interact_with_virtual_agents(self.agent_lt, iter_limit)

                # ==interaction with estimated agent
                self.agent_gs.ibr_interact(iter_limit)

            elif self.agent_gs.conl_type in {'opt'}:
                self.agent_gs.opt_plan()

            "==update state=="
            self.agent_lt.update_state(self.agent_gs)
            self.agent_gs.update_state(self.agent_lt)

            if break_when_finish:
                if self.agent_gs.observed_trajectory[-1, 0] < self.agent_lt.observed_trajectory[-1, 0] \
                        or self.agent_lt.observed_trajectory[-1, 1] > self.agent_gs.observed_trajectory[-1, 1]:
                    self.num_step = t + 1
                    break

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

    def visualize(self):
        # set figures
        fig, ax = plt.subplots()
        plt.title('trajectory_LT_' + self.semantic_result)
        img = plt.imread('background_pic/T_intersection.jpg')
        plt.imshow(img, extent=[-9.1, 24.9, -13, 8])
        plt.xlim([-9.1, 24.9])
        plt.ylim([-13, 8])

        "====data abstraction===="
        # lt track (observed and planned)
        lt_ob_trj = self.agent_lt.observed_trajectory[:, 0:2]
        lt_heading = self.agent_lt.observed_trajectory[:, 4] / math.pi * 180

        # gs track (observed and planned)
        gs_ob_trj = self.agent_gs.observed_trajectory[:, 0:2]
        gs_heading = self.agent_gs.observed_trajectory[:, 4] / math.pi * 180

        num_frame = len(lt_ob_trj)
        "====show plans at each time step===="
        # # central vertices
        # cv_it, _ = get_central_vertices('lt')
        # cv_gs, _ = get_central_vertices('gs')
        # ax1.plot(cv_it[:, 0], cv_it[:, 1], 'r-')
        # ax1.plot(cv_gs[:, 0], cv_gs[:, 1], 'b-')

        # position at each time step
        for t in range(num_frame):
            draw_rectangle(lt_ob_trj[t, 0], lt_ob_trj[t, 1], lt_heading[t], ax,
                           para_alpha=(t + 1) / num_frame, para_color='#0E76CF')
            draw_rectangle(gs_ob_trj[t, 0], gs_ob_trj[t, 1], gs_heading[t], ax,
                           para_alpha=(t + 1) / num_frame, para_color='#7030A0')

        # full tracks at each time step
        for t in range(self.num_step):
            lt_track = self.agent_lt.trj_solution_collection[t]
            plt.plot(lt_track[:, 0], lt_track[:, 1], '--', color='black')
            gs_track = self.agent_gs.trj_solution_collection[t]
            plt.plot(gs_track[:, 0], gs_track[:, 1], '--', color='black')

        # connect two agents
        for t in range(self.num_step + 1):
            plt.plot([self.agent_lt.observed_trajectory[t, 0], self.agent_gs.observed_trajectory[t, 0]],
                     [self.agent_lt.observed_trajectory[t, 1], self.agent_gs.observed_trajectory[t, 1]],
                     color='black',
                     alpha=0.2)

        "====show velocity===="
        plt.figure(2)
        x_range = np.array(range(np.size(self.agent_lt.observed_trajectory, 0)))
        vel_norm_lt = np.linalg.norm(self.agent_lt.observed_trajectory[:, 2:4], axis=1)
        vel_norm_gs = np.linalg.norm(self.agent_gs.observed_trajectory[:, 2:4], axis=1)
        plt.plot(x_range, vel_norm_lt, color='red', label='LT velocity')
        plt.plot(x_range, vel_norm_gs, color='blue', label='FC velocity')
        plt.legend()

        plt.show()


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
    === main for simulating unprotected left-turning ===
    1. set initial motion state before the simulation
    2. change controller type by manually setting controller_type_xx as 'gt' or 'opt'
        * 'gt' is the game-theoretic planner work by solving IBR process
        * 'opt' is the optimal controller work by solving single optimization
    """

    tag = 'test'  # for saving data
    # initial state of the left-turn vehicle
    init_position_lt = [11, -5.8]
    init_velocity_lt = [1.5, 0.3]
    init_heading_lt = math.pi / 4
    ipv_lt = 0
    controller_type_lt = 'opt'

    # initial state of the go-straight vehicle
    init_position_gs = [22, -2]
    init_velocity_gs = [-1.5, 0]
    init_heading_gs = math.pi
    ipv_gs = math.pi/8
    controller_type_gs = 'opt'

    simu_scenario = Scenario([init_position_lt, init_position_gs],
                             [init_velocity_lt, init_velocity_gs],
                             [init_heading_lt, init_heading_gs],
                             [ipv_lt, ipv_gs],
                             [controller_type_lt, controller_type_gs])

    simu = Simulator()
    simu.initialize(simu_scenario, tag)
    simu.interact(simu_step=30)
    simu.post_process()
    simu.visualize()


def main2():
    tag = 'nds-simu'
    simu = Simulator(case_id=0)
    simu_scenario = simu.read_nds_scenario()
    if simu_scenario:
        simu.initialize(simu_scenario, tag)
        simu.agent_gs.target = 'gs_nds'
        simu.agent_lt.target = 'lt_nds'
        simu.interact(simu_step=30)
        simu.post_process()
        simu.visualize()


if __name__ == '__main__':

    'unprotected left-turn at a T-intersection'
    # main1()

    'test with nds data from jianhe-xianxia intersection (demo for VTD)'
    main2()
