"""
Interaction simulator
"""
import copy
import math
import numpy as np
from agent import Agent
from tools.utility import draw_rectangle, get_central_vertices, smooth_ployline
from tools.lattice_planner import lattice_planning
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
        self.sim_type = None
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

    def interact(self, simu_step=30, make_video=False, break_when_finish=False):
        """
        Simulate the given scenario step by step

        Parameters
        ----------
        make_video
        simu_step: number of simulation steps
        break_when_finish: (if set to be True) break the simulation when any agent crossed the conflict point

        """
        self.num_step = simu_step
        iter_limit = 3

        if make_video:
            plt.ion()
            _, ax = plt.subplots()

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

            elif self.agent_lt.conl_type in {'replay'}:
                t_end = t + self.agent_lt.track_len
                self.agent_lt.trj_solution = self.lt_actual_trj[t:t_end, :]

            elif self.agent_lt.conl_type in {'lattice'}:
                path_point, _ = get_central_vertices('lt_nds', origin_point=self.agent_lt.position)
                obstacle_data = {'px': self.agent_gs.position[0],
                                 'py': self.agent_gs.position[1],
                                 'v': np.linalg.norm(self.agent_gs.velocity),
                                 'heading': self.agent_gs.heading}
                initial_state = {'px': self.agent_lt.position[0],
                                 'py': self.agent_lt.position[1],
                                 'v': np.linalg.norm(self.agent_lt.velocity),
                                 'heading': self.agent_lt.heading}
                res = lattice_planning(path_point, obstacle_data, initial_state, show_res=False)
                self.agent_lt.trj_solution = np.array(res[:self.agent_lt.track_len])

            "==plan for go straight=="
            if self.agent_gs.conl_type in {'gt'}:
                # ==interaction with parallel virtual agents
                self.agent_gs.ibr_interact_with_virtual_agents(self.agent_lt, iter_limit)

                # ==interaction with estimated agent
                self.agent_gs.ibr_interact(iter_limit)

            elif self.agent_gs.conl_type in {'opt'}:
                self.agent_gs.opt_plan()

            elif self.agent_gs.conl_type in {'replay'}:
                track_len = self.agent_gs.track_len
                t_end = t + track_len
                self.agent_gs.trj_solution = self.gs_actual_trj[t:t_end, :]

            elif self.agent_gs.conl_type in {'lattice'}:
                path_point, _ = get_central_vertices('gs_nds', origin_point=self.agent_gs.position)
                obstacle_data = {'px': self.agent_lt.position[0],
                                 'py': self.agent_lt.position[1],
                                 'v': np.linalg.norm(self.agent_lt.velocity),
                                 'heading': self.agent_lt.heading}
                initial_state = {'px': self.agent_gs.position[0],
                                 'py': self.agent_gs.position[1],
                                 'v': np.linalg.norm(self.agent_gs.velocity),
                                 'heading': self.agent_gs.heading}
                res = lattice_planning(path_point, obstacle_data, initial_state, show_res=False)
                self.agent_gs.trj_solution = np.array(res[:self.agent_gs.track_len])

            "==update state=="
            self.agent_lt.update_state(self.agent_gs)
            self.agent_gs.update_state(self.agent_lt)

            "==update video=="
            if make_video:
                plt.cla()
                if self.sim_type == 'simu':
                    img = plt.imread('background_pic/T_intersection.jpg')
                    plt.imshow(img, extent=[-9.1, 24.9, -13, 8])
                    plt.xlim([-9.1, 24.9])
                    plt.ylim([-13, 8])
                    # central vertices
                    cv_it, _ = get_central_vertices('lt')
                    cv_gs, _ = get_central_vertices('gs')
                elif self.sim_type == 'nds':
                    plt.xlim([-22 - 13, 53 - 13])
                    plt.ylim([-31 - 7.8, 57 - 7.8])
                    img = plt.imread('background_pic/Jianhexianxia-v2.png')
                    plt.imshow(img, extent=[-28 - 13, 58 - 13, -42 - 7.8, 64 - 7.8])
                    # central vertices
                    lt_origin_point = self.agent_lt.observed_trajectory[0, 0:2]
                    gs_origin_point = self.agent_gs.observed_trajectory[0, 0:2]
                    cv_it, _ = get_central_vertices('lt_nds', origin_point=lt_origin_point)
                    cv_gs, _ = get_central_vertices('gs_nds', origin_point=gs_origin_point)

                draw_rectangle(self.agent_lt.position[0], self.agent_lt.position[1],
                               self.agent_lt.heading / math.pi * 180, ax,
                               para_alpha=1, para_color='#0E76CF')
                draw_rectangle(self.agent_gs.position[0], self.agent_gs.position[1],
                               self.agent_gs.heading / math.pi * 180, ax,
                               para_alpha=1, para_color='#7030A0')
                ax.axis('scaled')
                plt.show()
                plt.pause(0.1)
                plt.savefig('figures/' + self.tag + '-' + str(t) + '.png', dpi=300)

            if break_when_finish:
                if self.agent_gs.observed_trajectory[-1, 0] < self.agent_lt.observed_trajectory[-1, 0] \
                        or self.agent_lt.observed_trajectory[-1, 1] > self.agent_gs.observed_trajectory[-1, 1]:
                    self.num_step = t + 1
                    break

    def get_semantic_result(self):
        """
        Identify semantic interaction results after simulation:
        1. crashed or not (not critical judgement)
        2. the left-turn vehicle yield or not
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

        cv_lt = []
        cv_gs = []
        # set figures
        fig, axes = plt.subplots(1, 2, figsize=[8, 4])
        fig.suptitle('trajectory_LT_' + self.semantic_result)
        axes[0].set_title('trajectory')
        axes[1].set_title('velocity')
        if self.sim_type == 'simu':
            axes[0].set(aspect=1, xlim=(-9.1, 24.9), ylim=(-13, 8))
            img = plt.imread('background_pic/T_intersection.jpg')
            axes[0].imshow(img, extent=[-9.1, 24.9, -13, 8])
            # central vertices
            cv_lt, _ = get_central_vertices('lt')
            cv_gs, _ = get_central_vertices('gs')
        elif self.sim_type == 'nds':
            axes[0].set(aspect=1, xlim=(-22 - 13, 53 - 13), ylim=(-31 - 7.8, 57 - 7.8))
            img = plt.imread('background_pic/Jianhexianxia-v2.png')
            axes[0].imshow(img, extent=[-28 - 13, 58 - 13, -42 - 7.8, 64 - 7.8])
            # central vertices
            lt_origin_point = self.agent_lt.observed_trajectory[0, 0:2]
            gs_origin_point = self.agent_gs.observed_trajectory[0, 0:2]
            cv_lt, _ = get_central_vertices('lt_nds', origin_point=lt_origin_point)
            cv_gs, _ = get_central_vertices('gs_nds', origin_point=gs_origin_point)

        "---- data abstraction ----"
        # lt track (observed in simulation and ground truth in nds)
        lt_ob_trj = self.agent_lt.observed_trajectory[:, 0:2]
        lt_ob_heading = self.agent_lt.observed_trajectory[:, 4] / math.pi * 180
        lt_nds_trj = np.array(self.lt_actual_trj[:, 0:2])
        lt_nds_heading = np.array(self.lt_actual_trj[:, 4]) / math.pi * 180
        vel_ob_vel_norm_lt = np.linalg.norm(self.agent_lt.observed_trajectory[:, 2:4], axis=1)
        vel_nds_vel_norm_lt = np.linalg.norm(self.lt_actual_trj[:, 2:4], axis=1)

        # gs track (observed in simulation and ground truth in nds)
        gs_ob_trj = self.agent_gs.observed_trajectory[:, 0:2]
        gs_ob_heading = self.agent_gs.observed_trajectory[:, 4] / math.pi * 180
        gs_nds_trj = np.array(self.gs_actual_trj[:, 0:2])
        gs_nds_heading = np.array(self.gs_actual_trj[:, 4]) / math.pi * 180
        vel_ob_vel_norm_gs = np.linalg.norm(self.agent_gs.observed_trajectory[:, 2:4], axis=1)
        vel_nds_vel_norm_gs = np.linalg.norm(self.gs_actual_trj[:, 2:4], axis=1)

        num_frame = len(lt_ob_trj)

        "---- show plans at each time step ----"
        axes[0].plot(cv_lt[:, 0], cv_lt[:, 1], 'b-')
        axes[0].plot(cv_gs[:, 0], cv_gs[:, 1], 'r-')

        # ----position at each time step
        # version 1
        for t in range(num_frame):
            # simulation results
            draw_rectangle(lt_ob_trj[t, 0], lt_ob_trj[t, 1], lt_ob_heading[t], axes[0],
                           para_alpha=0.3, para_color='#0E76CF')
            draw_rectangle(gs_ob_trj[t, 0], gs_ob_trj[t, 1], gs_ob_heading[t], axes[0],
                           para_alpha=0.3, para_color='#7030A0')

            # nds ground truth
            draw_rectangle(lt_nds_trj[t, 0], lt_nds_trj[t, 1], lt_nds_heading[t], axes[0],
                           para_alpha=0.3, para_color='blue')
            draw_rectangle(gs_nds_trj[t, 0], gs_nds_trj[t, 1], gs_nds_heading[t], axes[0],
                           para_alpha=0.3, para_color='red')

        # version 2
        # plt.scatter(lt_ob_trj[:, 0],
        #             lt_ob_trj[:, 1],
        #             s=80,
        #             alpha=0.6,
        #             color='#0E76CF',
        #             label='left-turn')
        # plt.scatter(gs_ob_trj[:, 0],
        #             gs_ob_trj[:, 1],
        #             s=80,
        #             alpha=0.6,
        #             color='#7030A0',
        #             label='go-straight')

        # ----full tracks at each time step
        # for t in range(self.num_step):
        #     lt_track = self.agent_lt.trj_solution_collection[t]
        #     plt.plot(lt_track[:, 0], lt_track[:, 1], '--', color='black')
        #     gs_track = self.agent_gs.trj_solution_collection[t]
        #     plt.plot(gs_track[:, 0], gs_track[:, 1], '--', color='black')

        # ----connect two agents
        for t in range(self.num_step + 1):
            axes[0].plot([self.agent_lt.observed_trajectory[t, 0], self.agent_gs.observed_trajectory[t, 0]],
                         [self.agent_lt.observed_trajectory[t, 1], self.agent_gs.observed_trajectory[t, 1]],
                         color='black',
                         alpha=0.2)

        "---- show velocity ----"
        # plt.figure()
        x_range = np.array(range(np.size(self.agent_lt.observed_trajectory, 0)))
        axes[1].plot(x_range, vel_ob_vel_norm_lt, color='red', label='LT velocity simulation')
        axes[1].plot(x_range, vel_ob_vel_norm_gs, color='blue', label='FC velocity simulation')
        axes[1].plot(x_range, vel_nds_vel_norm_lt[x_range], linestyle='--', color='red', label='LT velocity NDS')
        axes[1].plot(x_range, vel_nds_vel_norm_gs[x_range], linestyle='--', color='blue', label='FC velocity NDS')
        axes[1].legend()
        plt.show()
        plt.savefig('figures/' + self.tag + '-final.png', dpi=600)

    def read_nds_scenario(self, controller_type_lt, controller_type_gs):
        cross_id, data_cross, _ = analyze_ipv_in_nds(self.case_id)
        # data_cross:
        # 0-ipv_lt | ipv_lt_error | lt_px | lt_py  | lt_vx  | lt_vy  | lt_heading  |...
        # 7-ipv_gs | ipv_gs_error | gs_px | gs_py  | gs_vx  | gs_vy  | gs_heading  |

        if cross_id == -1:
            return None
        else:
            init_position_lt = [data_cross[0, 2] - 13, data_cross[0, 3] - 7.8]
            init_velocity_lt = [data_cross[0, 4], data_cross[0, 5]]
            init_heading_lt = data_cross[0, 6]
            ipv_lt = np.mean(data_cross[4:, 0])
            init_position_gs = [data_cross[0, 9] - 13, data_cross[0, 10] - 7.8]
            init_velocity_gs = [data_cross[0, 11], data_cross[0, 12]]
            init_heading_gs = data_cross[0, 13]
            ipv_gs = np.mean(data_cross[4:, 7])+0.4
            self.lt_actual_trj = data_cross[:, 2:7]
            self.lt_actual_trj[:, 0] = self.lt_actual_trj[:, 0] - 13
            self.lt_actual_trj[:, 1] = self.lt_actual_trj[:, 1] - 7.8

            self.gs_actual_trj = data_cross[:, 9:14]
            self.gs_actual_trj[:, 0] = self.gs_actual_trj[:, 0] - 13
            self.gs_actual_trj[:, 1] = self.gs_actual_trj[:, 1] - 7.8

            return Scenario([init_position_lt, init_position_gs],
                            [init_velocity_lt, init_velocity_gs],
                            [init_heading_lt, init_heading_gs],
                            [ipv_lt, ipv_gs],
                            [controller_type_lt, controller_type_gs])

    def ipv_analysis(self):
        ipv_collection_lt = []
        ipv_collection_gs = []
        ipv_error_collection_lt = []
        ipv_error_collection_gs = []
        track_lt = self.agent_lt.observed_trajectory
        track_gs = self.agent_gs.observed_trajectory
        for t in range(np.size(track_lt, 0) - 6):
            init_position_lt = track_lt[t, 0:2]
            init_velocity_lt = track_lt[t, 2:4]
            init_heading_lt = track_lt[t, 4]
            agent_lt = Agent(init_position_lt, init_velocity_lt, init_heading_lt, 'lt_nds')
            track_lt_temp = track_lt[t:t + 6, 0:2]

            init_position_gs = track_gs[t, 0:2]
            init_velocity_gs = track_gs[t, 2:4]
            init_heading_gs = track_gs[t, 4]
            agent_gs = Agent(init_position_gs, init_velocity_gs, init_heading_gs, 'gs_nds')
            track_gs_temp = track_gs[t:t + 6, 0:2]

            # estimate ipv
            agent_lt.estimate_self_ipv_in_nds(track_lt_temp, track_gs_temp)
            ipv_collection_lt.append(agent_lt.ipv)
            ipv_error_collection_lt.append(agent_lt.ipv_error)

            agent_gs.estimate_self_ipv_in_nds(track_gs_temp, track_lt_temp)
            ipv_collection_gs.append(agent_gs.ipv)
            ipv_error_collection_gs.append(agent_gs.ipv_error)

        # figure preparation
        x_range = np.array(range(len(ipv_collection_lt)))
        ipv_collection_lt = np.array(ipv_collection_lt)
        ipv_collection_gs = np.array(ipv_collection_gs)
        ipv_error_collection_lt = np.array(ipv_error_collection_lt)
        ipv_error_collection_gs = np.array(ipv_error_collection_gs)

        plt.figure()
        plt.plot(x_range, ipv_collection_lt, color='#0E76CF')
        plt.fill_between(x_range,
                         ipv_collection_lt - ipv_error_collection_lt,
                         ipv_collection_lt + ipv_error_collection_lt,
                         alpha=0.3,
                         color='#0E76CF',
                         label='estimated lt IPV')
        plt.plot(x_range, ipv_collection_gs, color='#7030A0')
        plt.fill_between(x_range,
                         ipv_collection_gs - ipv_error_collection_gs,
                         ipv_collection_gs + ipv_error_collection_gs,
                         alpha=0.3,
                         color='#7030A0',
                         label='estimated gs IPV')

        # y_lt = np.array(ipv_collection_lt)
        # point_smoothed_lt, _ = smooth_ployline(np.array([x_range, y_lt]).T)
        # x_smoothed_lt = point_smoothed_lt[:, 0]
        # y_lt_smoothed = point_smoothed_lt[:, 1]
        # y_error_lt = np.array(ipv_error_collection_lt)
        # error_smoothed_lt, _ = smooth_ployline(np.array([x_range, y_error_lt]).T)
        # y_error_lt_smoothed = error_smoothed_lt[:, 1]
        #
        # y_gs = np.array(ipv_collection_gs)
        # point_smoothed_gs, _ = smooth_ployline(np.array([x_range, y_gs]).T)
        # x_smoothed_gs = point_smoothed_gs[:, 0]
        # y_gs_smoothed = point_smoothed_gs[:, 1]
        # y_error_gs = np.array(ipv_error_collection_gs)
        # error_smoothed_gs, _ = smooth_ployline(np.array([x_range, y_error_gs]).T)
        # y_error_gs_smoothed = error_smoothed_gs[:, 1]
        #
        # plt.figure()
        # plt.fill_between(x_smoothed_lt, y_lt_smoothed - y_error_lt_smoothed, y_lt_smoothed + y_error_lt_smoothed,
        #                  alpha=0.3,
        #                  color='blue',
        #                  label='estimated lt IPV')
        # plt.fill_between(x_smoothed_gs, y_gs_smoothed - y_error_gs_smoothed, y_gs_smoothed + y_error_gs_smoothed,
        #                  alpha=0.3,
        #                  color='red',
        #                  label='estimated gs IPV')


def main1():
    """
    === main for simulating unprotected left-turning ===
    1. Set initial motion state before the simulation
    2. Change controller type by manually setting controller_type_xx as 'gt' or 'opt'
        * 'gt' is the game-theoretic planner work by solving IBR process
        * 'opt' is the optimal controller work by solving single optimization
    """

    tag = 'test'  # tag for data saving

    '---- set initial state of the left-turn vehicle ----'
    init_position_lt = [11, -5.8]
    init_velocity_lt = [1.5, 0.3]
    init_heading_lt = math.pi / 4
    ipv_lt = 0
    controller_type_lt = 'opt'

    '---- set initial state of the go-straight vehicle ----'
    init_position_gs = [22, -2]
    init_velocity_gs = [-1.5, 0]
    init_heading_gs = math.pi
    ipv_gs = math.pi / 8
    controller_type_gs = 'opt'

    '---- generate scenario ----'
    simu_scenario = Scenario([init_position_lt, init_position_gs],
                             [init_velocity_lt, init_velocity_gs],
                             [init_heading_lt, init_heading_gs],
                             [ipv_lt, ipv_gs],
                             [controller_type_lt, controller_type_gs])

    simu = Simulator()
    simu.sim_type = 'simu'
    simu.initialize(simu_scenario, tag)
    simu.interact(simu_step=10)
    simu.get_semantic_result()
    simu.visualize()


def main2():
    """
       === main for simulating unprotected left-turning scenarios in Jianhe-Xianxia dataset ===
       1. Set case_id to get initial scenarios state of a single case
       2. Change controller type by manually setting controller_type_xx as 'gt' or 'opt'
           * 'gt' is the game-theoretic planner work by solving IBR process
           * 'opt' is the optimal controller work by solving single optimization
       """
    tag = 'nds-simu-case71'

    simu = Simulator(case_id=71)
    simu.sim_type = 'nds'
    controller_type_lt = 'gt'
    controller_type_gs = 'gt'
    simu_scenario = simu.read_nds_scenario(controller_type_lt, controller_type_gs)
    if simu_scenario:
        simu.initialize(simu_scenario, tag)
        simu.agent_gs.target = 'gs_nds'
        simu.agent_lt.target = 'lt_nds'
        simu.interact(simu_step=1, make_video=False, break_when_finish=False)
        simu.get_semantic_result()
        simu.visualize()


def main_test():
    """
    === main for testing a planner with replayed trajectory ===
        * set lt agent as the vehicle under test (ipv=pi/4)
    """

    tag = 'gt-test'

    simu = Simulator(case_id=51)
    simu.sim_type = 'nds'
    controller_type_lt = 'lattice'
    controller_type_gs = 'gt'
    simu_scenario = simu.read_nds_scenario(controller_type_lt, controller_type_gs)
    if simu_scenario:
        simu.initialize(simu_scenario, tag)
        simu.agent_gs.target = 'gs_nds'
        simu.agent_lt.target = 'lt_nds'
        # simu.agent_gs.ipv = math.pi/4
        simu.interact(simu_step=15, make_video=False)
        simu.get_semantic_result()
        simu.ipv_analysis()
        simu.visualize()


if __name__ == '__main__':
    'simulate unprotected left-turn at a T-intersection'
    # main1()

    'simulate with nds data from Jianhe-Xianxia intersection'
    main2()

    'test lattice planner with trajectory replay'
    # main_test()
