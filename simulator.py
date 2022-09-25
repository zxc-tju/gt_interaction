"""
Interaction simulator
"""
import copy
import math
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process, Manager
import numpy as np
import pandas as pd
from agent import Agent
from tools.utility import draw_rectangle, get_central_vertices, smooth_ployline
from tools.lattice_planner import lattice_planning
import matplotlib.pyplot as plt
from NDS_analysis import analyze_ipv_in_nds, cal_pet
from viztracer import VizTracer
import time


class Scenario:
    def __init__(self, pos, vel, heading, ipv, conl_type, acc=None):
        self.position = {'lt': np.array(pos[0]), 'gs': np.array(pos[1])}
        self.velocity = {'lt': np.array(vel[0]), 'gs': np.array(vel[1])}
        if acc:
            self.acceleration = {'lt': np.array(acc[0]), 'gs': np.array(acc[1])}
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
        self.simu_time = 0
        self.case_len = 0
        self.ending_point = None
        self.gs_actual_trj = []
        self.lt_actual_trj = []

    def initialize(self, scenario, case_tag):
        self.scenario = scenario
        self.agent_lt = Agent(scenario.position['lt'], scenario.velocity['lt'], scenario.heading['lt'], 'lt',
                              scenario.acceleration['lt'])
        self.agent_gs = Agent(scenario.position['gs'], scenario.velocity['gs'], scenario.heading['gs'], 'gs',
                              scenario.acceleration['gs'])
        self.agent_lt.estimated_inter_agent = copy.deepcopy(self.agent_gs)
        self.agent_gs.estimated_inter_agent = copy.deepcopy(self.agent_lt)
        self.agent_lt.ipv = self.scenario.ipv['lt']
        # self.agent_lt.ipv = -math.pi/8
        self.agent_gs.ipv = self.scenario.ipv['gs']
        # self.agent_gs.ipv = math.pi/4
        self.agent_lt.conl_type = self.scenario.conl_type['lt']
        self.agent_gs.conl_type = self.scenario.conl_type['gs']
        self.tag = case_tag

    def interact(self, simu_step=30, iter_limit=5,
                 make_video=False,
                 break_when_finish=False,
                 interactive=True):
        """
        Simulate the given scenario step by step

        Parameters
        ----------
        interactive
        iter_limit: max iteration number
        make_video
        simu_step: number of simulation steps
        break_when_finish: (if set to be True) break the simulation when any agent crossed the conflict point

        """
        self.num_step = simu_step
        # iter_limit = 3

        if make_video:
            plt.ion()
            _, ax = plt.subplots()

        for t in range(self.num_step):

            # print('time_step: ', t, '/', self.num_step)
            # if t == 28:
            #     print('debug!')

            "==plan for left-turn=="
            if self.agent_lt.conl_type in {'gt'}:

                # ==interaction with parallel virtual agents
                self.agent_lt.ibr_interact_with_virtual_agents(self.agent_gs)
                # self.agent_lt.ibr_interact_with_virtual_agents_parallel(self.agent_gs)

                # ==interaction with estimated agent
                self.agent_lt.ibr_interact(iter_limit=iter_limit)

            elif self.agent_lt.conl_type in {'linear-gt'}:
                # time1 = time.perf_counter()
                self.agent_lt.lp_ibr_interact(iter_limit=iter_limit, interactive=interactive)
                # time2 = time.perf_counter()
                # print('time consumption: ', time2 - time1)

            elif self.agent_lt.conl_type in {'opt'}:
                self.agent_lt.opt_plan()

            elif self.agent_lt.conl_type in {'idm'}:
                self.agent_lt.idm_plan(self.agent_gs)

            elif self.agent_lt.conl_type in {'replay'}:
                t_end = t + self.agent_lt.track_len
                self.agent_lt.trj_solution = self.lt_actual_trj[t:t_end, :]

            elif self.agent_lt.conl_type in {'lattice'}:
                if not self.agent_lt.plan_count:
                    path_point, _ = get_central_vertices('lt_nds',
                                                         origin_point=self.agent_lt.observed_trajectory[0, 0:2])
                    obstacle_data = {'px': self.agent_gs.position[0],
                                     'py': self.agent_gs.position[1],
                                     'v': np.linalg.norm(self.agent_gs.velocity),
                                     'heading': self.agent_gs.heading}
                    initial_state = {'px': self.agent_lt.position[0],
                                     'py': self.agent_lt.position[1],
                                     'v': np.linalg.norm(self.agent_lt.velocity),
                                     'heading': self.agent_lt.heading}
                    res, state = lattice_planning(path_point, obstacle_data, initial_state, show_res=False)
                    if len(res) >= 3:
                        self.agent_lt.trj_solution = np.array(res[:self.agent_lt.track_len])
                        self.agent_lt.plan_count = 2
                        if state == 'planning_back':
                            self.agent_lt.plan_count = 7
                    else:
                        self.agent_lt.cruise_plan()

                else:
                    self.agent_lt.trj_solution = self.agent_lt.trj_solution[1:, :]
                    self.agent_lt.plan_count = self.agent_lt.plan_count - 1

            "==plan for go straight=="
            if self.agent_gs.conl_type in {'gt'}:
                # ==interaction with parallel virtual agents
                self.agent_gs.ibr_interact_with_virtual_agents(self.agent_lt, iter_limit)
                # self.agent_gs.ibr_interact_with_virtual_agents_parallel(self.agent_lt, iter_limit)

                # ==interaction with estimated agent
                self.agent_gs.ibr_interact(iter_limit)

            elif self.agent_gs.conl_type in {'linear-gt'}:
                self.agent_gs.lp_ibr_interact(iter_limit=iter_limit, interactive=interactive)
                # print(self.agent_gs.trj_solution)

            elif self.agent_gs.conl_type in {'opt'}:
                self.agent_gs.opt_plan()

            elif self.agent_gs.conl_type in {'idm'}:
                self.agent_gs.idm_plan(self.agent_lt)

            elif self.agent_gs.conl_type in {'replay'}:
                track_len = self.agent_gs.track_len
                t_end = t + track_len
                self.agent_gs.trj_solution = self.gs_actual_trj[t:t_end, :]

            elif self.agent_gs.conl_type in {'lattice'}:
                if not self.agent_gs.plan_count:
                    path_point, _ = get_central_vertices('gs_nds', origin_point=self.agent_gs.position)
                    obstacle_data = {'px': self.agent_lt.position[0],
                                     'py': self.agent_lt.position[1],
                                     'v': np.linalg.norm(self.agent_lt.velocity),
                                     'heading': self.agent_lt.heading}
                    initial_state = {'px': self.agent_gs.position[0],
                                     'py': self.agent_gs.position[1],
                                     'v': np.linalg.norm(self.agent_gs.velocity),
                                     'heading': self.agent_gs.heading}
                    res, state = lattice_planning(path_point, obstacle_data, initial_state, show_res=False)
                    if len(res) >= 3:
                        self.agent_gs.trj_solution = np.array(res[:self.agent_gs.track_len])
                        self.agent_gs.plan_count = 2
                        if state == 'planning_back':
                            self.agent_gs.plan_count = 7
                    else:
                        self.agent_gs.cruise_plan()

                else:
                    self.agent_gs.trj_solution = self.agent_gs.trj_solution[1:, :]
                    self.agent_gs.plan_count = self.agent_gs.plan_count - 1

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
                plt.pause(0.0001)
                plt.savefig('figures/' + self.tag + '-' + str(t) + '.png', dpi=300)

            if break_when_finish:
                if self.agent_gs.observed_trajectory[-1, 0] < self.agent_lt.observed_trajectory[-1, 0] \
                        or self.agent_lt.observed_trajectory[-1, 1] > self.agent_gs.observed_trajectory[-1, 1]:
                    self.num_step = t + 1
                    break

    def visualize_final_results(self, file_path):

        cv_lt = []
        cv_gs = []
        # set figures
        fig, axes = plt.subplots(1, 2, figsize=[10, 5])
        # fig.suptitle('trajectory_LT_' + self.semantic_result)
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

        vel_ob_vel_norm_lt = np.linalg.norm(self.agent_lt.observed_trajectory[:, 2:4], axis=1)
        if self.sim_type == 'nds':
            vel_nds_vel_norm_lt = np.linalg.norm(self.lt_actual_trj[:, 2:4], axis=1)
            lt_nds_trj = np.array(self.lt_actual_trj[:, 0:2])
            lt_nds_heading = np.array(self.lt_actual_trj[:, 4]) / math.pi * 180

        # gs track (observed in simulation and ground truth in nds)
        gs_ob_trj = self.agent_gs.observed_trajectory[:, 0:2]
        gs_ob_heading = self.agent_gs.observed_trajectory[:, 4] / math.pi * 180

        vel_ob_vel_norm_gs = np.linalg.norm(self.agent_gs.observed_trajectory[:, 2:4], axis=1)
        if self.sim_type == 'nds':
            vel_nds_vel_norm_gs = np.linalg.norm(self.gs_actual_trj[:, 2:4], axis=1)
            gs_nds_trj = np.array(self.gs_actual_trj[:, 0:2])
            gs_nds_heading = np.array(self.gs_actual_trj[:, 4]) / math.pi * 180

        num_frame = len(lt_ob_trj)

        "---- show plans at each time step ----"
        axes[0].plot(cv_lt[:, 0], cv_lt[:, 1], 'b-', linewidth=0.1)
        axes[0].plot(cv_gs[:, 0], cv_gs[:, 1], 'r-', linewidth=0.1)

        # ----position at each time step
        # version 1
        # for t in range(num_frame):
        #     # simulation results
        #     draw_rectangle(lt_ob_trj[t, 0], lt_ob_trj[t, 1], lt_ob_heading[t], axes[0],
        #                    para_alpha=0.3, para_color='#0E76CF')
        #     draw_rectangle(gs_ob_trj[t, 0], gs_ob_trj[t, 1], gs_ob_heading[t], axes[0],
        #                    para_alpha=0.3, para_color='#7030A0')
        # #
        #     # nds ground truth
        #     if self.sim_type == 'nds':
        #         draw_rectangle(lt_nds_trj[t, 0], lt_nds_trj[t, 1], lt_nds_heading[t], axes[0],
        #                        para_alpha=0.3, para_color='blue')
        #
        #         draw_rectangle(gs_nds_trj[t, 0], gs_nds_trj[t, 1], gs_nds_heading[t], axes[0],
        #                        para_alpha=0.3, para_color='red')

        # version 2
        axes[0].scatter(lt_ob_trj[:num_frame, 0],
                        lt_ob_trj[:num_frame, 1],
                        s=50,
                        alpha=0.6,
                        color='#0E76CF',
                        label='left-turn simulation')
        axes[0].scatter(gs_ob_trj[:num_frame, 0],
                        gs_ob_trj[:num_frame, 1],
                        s=50,
                        alpha=0.6,
                        color='#7030A0',
                        label='go-straight simulation')

        # if self.sim_type == 'nds':
        #     axes[0].scatter(lt_nds_trj[:num_frame, 0],
        #                     lt_nds_trj[:num_frame, 1],
        #                     s=50,
        #                     alpha=0.3,
        #                     color='blue',
        #                     label='left-turn NDS')
        #     axes[0].scatter(gs_nds_trj[:num_frame, 0],
        #                     gs_nds_trj[:num_frame, 1],
        #                     s=50,
        #                     alpha=0.3,
        #                     color='red',
        #                     label='go-straight NDS')

        # ----full tracks at each time step
        for t in range(self.num_step):
            if int(t / 2) == t / 2:
                lt_track = self.agent_lt.trj_solution_collection[t]
                axes[0].plot(lt_track[:, 0], lt_track[:, 1], '--', color='black', alpha=0.5)
                gs_track = self.agent_gs.trj_solution_collection[t]
                axes[0].plot(gs_track[:, 0], gs_track[:, 1], '--', color='black', alpha=0.5)
                if self.agent_gs.conl_type in {'gt', 'linear-gt'}:
                    gs_inter_track = self.agent_gs.estimated_inter_agent.trj_solution_collection[t]
                    axes[0].plot(gs_inter_track[:, 0], gs_inter_track[:, 1], '--', color='red', alpha=0.5)
                if self.agent_lt.conl_type in {'gt', 'linear-gt'}:
                    lt_inter_track = self.agent_lt.estimated_inter_agent.trj_solution_collection[t]
                    axes[0].plot(lt_inter_track[:, 0], lt_inter_track[:, 1], '--', color='red', alpha=0.5)

        # ----connect two agents
        for t in range(self.num_step + 1):
            axes[0].plot([self.agent_lt.observed_trajectory[t, 0], self.agent_gs.observed_trajectory[t, 0]],
                         [self.agent_lt.observed_trajectory[t, 1], self.agent_gs.observed_trajectory[t, 1]],
                         color='gray',
                         alpha=0.2)

        "---- show velocity ----"
        x_range = np.array(range(np.size(self.agent_lt.observed_trajectory, 0)))
        axes[1].plot(x_range, vel_ob_vel_norm_lt, linestyle='--',
                     color='blue', label='left-turn simulation')
        axes[1].plot(x_range, vel_ob_vel_norm_gs, linestyle='--',
                     color='red', label='go-straight simulation')

        if self.sim_type == 'nds':
            x_range_nds = self.simu_time + x_range
            axes[1].plot(x_range, vel_nds_vel_norm_gs[x_range_nds],
                         color='red', label='go-straight NDS')
            axes[1].plot(x_range, vel_nds_vel_norm_lt[x_range_nds],
                         color='blue', label='left-turn NDS')

        axes[1].legend()
        axes[0].legend()
        plt.show()
        plt.savefig(file_path + self.tag + '-final.png', dpi=600)
        # plt.savefig(file_path + self.tag + '-final.svg')
        plt.close()

    def visualize_single_step(self, file_path):

        cv_lt = []
        cv_gs = []
        # set figures
        fig, axes = plt.subplots(1, 1, figsize=[5, 5])
        # fig.suptitle('trajectory_LT_' + self.semantic_result)
        axes.set_title('trajectory')
        if self.sim_type == 'simu':
            axes.set(aspect=1, xlim=(-9.1, 24.9), ylim=(-13, 8))
            img = plt.imread('background_pic/T_intersection.jpg')
            axes.imshow(img, extent=[-9.1, 24.9, -13, 8])
            # central vertices
            cv_lt, _ = get_central_vertices('lt')
            cv_gs, _ = get_central_vertices('gs')
        elif self.sim_type == 'nds':
            axes.set(aspect=1, xlim=(-22 - 13, 53 - 13), ylim=(-31 - 7.8, 57 - 7.8))
            img = plt.imread('background_pic/Jianhexianxia-v2.png')
            axes.imshow(img, extent=[-28 - 13, 58 - 13, -42 - 7.8, 64 - 7.8])
            # central vertices
            lt_origin_point = self.agent_lt.observed_trajectory[0, 0:2]
            gs_origin_point = self.agent_gs.observed_trajectory[0, 0:2]
            cv_lt, _ = get_central_vertices('lt_nds', origin_point=lt_origin_point)
            cv_gs, _ = get_central_vertices('gs_nds', origin_point=gs_origin_point)

        "---- data abstraction ----"
        # lt track (observed in simulation and ground truth in nds)
        lt_ob_trj = self.agent_lt.observed_trajectory[:, 0:2]
        # gs track (observed in simulation and ground truth in nds)
        gs_ob_trj = self.agent_gs.observed_trajectory[:, 0:2]

        # solution at this single time point
        lt_plan = self.agent_lt.trj_solution_collection[0]
        lt_inter_plan = self.agent_lt.estimated_inter_agent.trj_solution_collection[0]
        gs_plan = self.agent_gs.trj_solution_collection[0]
        gs_inter_plan = self.agent_gs.estimated_inter_agent.trj_solution_collection[0]

        # find the closest point pairs
        dis_lt_plan = np.linalg.norm(lt_plan[:, 0:2] - lt_inter_plan[:, 0:2], axis=1)
        lt_index_min_dis = np.where(dis_lt_plan == min(dis_lt_plan))

        dis_gs_plan = np.linalg.norm(gs_plan[:, 0:2] - gs_inter_plan[:, 0:2], axis=1)
        gs_index_min_dis = np.where(dis_gs_plan == min(dis_gs_plan))

        "---- show initial state ----"
        axes.plot(cv_lt[:, 0], cv_lt[:, 1], 'b-', linewidth=0.1)
        axes.plot(cv_gs[:, 0], cv_gs[:, 1], 'r-', linewidth=0.1)

        axes.scatter(lt_ob_trj[0, 0],
                     lt_ob_trj[0, 1],
                     s=50,
                     alpha=0.6,
                     color='#0E76CF',
                     label='left-turn-ipv-' + str(self.agent_lt.ipv))
        axes.scatter(gs_ob_trj[0, 0],
                     gs_ob_trj[0, 1],
                     s=50,
                     alpha=0.6,
                     color='#7030A0',
                     label='go-straight-ipv-' + str(self.agent_gs.ipv))

        # ----full tracks at each time step
        axes.scatter(lt_plan[:, 0], lt_plan[:, 1], color='blue', alpha=0.5, s=3)
        axes.scatter(gs_plan[:, 0], gs_plan[:, 1], color='red', alpha=0.5, s=3)
        if self.agent_gs.conl_type in {'gt', 'linear-gt'}:
            axes.scatter(gs_inter_plan[:, 0], gs_inter_plan[:, 1], color='red', alpha=0.2, s=3)
        if self.agent_lt.conl_type in {'gt', 'linear-gt'}:
            axes.scatter(lt_inter_plan[:, 0], lt_inter_plan[:, 1], color='blue', alpha=0.2, s=3)

        # ----connect two agents
        lt_index = int(lt_index_min_dis[0])
        for t in {lt_index}:
            axes.plot([lt_plan[t, 0], lt_inter_plan[t, 0]],
                      [lt_plan[t, 1], lt_inter_plan[t, 1]],
                      color='gray',
                      alpha=0.4,
                      linewidth=0.2)
        gs_index = int(gs_index_min_dis[0])
        for t in {gs_index}:
            axes.plot([gs_plan[t, 0], gs_inter_plan[t, 0]],
                      [gs_plan[t, 1], gs_inter_plan[t, 1]],
                      color='purple',
                      alpha=0.4,
                      linewidth=0.2)

        # plt.show()
        axes.legend()
        plt.savefig(file_path + self.tag + '-final.png', dpi=300)
        # plt.savefig(file_path + self.tag + '-final.svg')
        # plt.close()

    def read_nds_scenario(self, controller_type_lt, controller_type_gs, t=0):
        cross_id, data_cross, _ = analyze_ipv_in_nds(self.case_id)
        # data_cross:
        # 0-ipv_lt | ipv_lt_error | lt_px | lt_py  | lt_vx  | lt_vy  | lt_heading  |...
        # 7-ipv_gs | ipv_gs_error | gs_px | gs_py  | gs_vx  | gs_vy  | gs_heading  |

        if cross_id == -1:
            print('-----no trajectory crossed in case' + str(self.case_id))
            return None
        else:
            init_position_lt = [data_cross[t, 2] - 13, data_cross[t, 3] - 7.8]
            init_velocity_lt = [data_cross[t, 4], data_cross[t, 5]]
            init_acceleration_lt = [(data_cross[t + 1, 4] - data_cross[t + 1, 4]) / 0.12,
                                    (data_cross[t + 1, 5] - data_cross[t + 1, 5]) / 0.12]
            init_heading_lt = data_cross[t, 6]
            if controller_type_lt in {'opt', 'gt', 'linear-gt'}:
                ipv_weight_lt = 1 - data_cross[4:, 1]
                ipv_weight_lt = ipv_weight_lt / ipv_weight_lt.sum()
                ipv_lt = sum(ipv_weight_lt * data_cross[4:, 0])
                # ipv_lt = max(sum(ipv_weight_lt * data_cross[4:, 0])-0.2, -math.pi*3/8)
            else:
                ipv_lt = 0

            init_position_gs = [data_cross[t, 9] - 13, data_cross[t, 10] - 7.8]
            init_velocity_gs = [data_cross[t, 11], data_cross[t, 12]]
            init_acceleration_gs = [(data_cross[t + 1, 11] - data_cross[t + 1, 11]) / 0.12,
                                    (data_cross[t + 1, 12] - data_cross[t + 1, 12]) / 0.12]
            init_heading_gs = data_cross[t, 13]
            if controller_type_gs in {'opt', 'gt', 'linear-gt'}:
                ipv_weight_gs = 1 - data_cross[4:, 8]
                ipv_weight_gs = ipv_weight_gs / ipv_weight_gs.sum()
                ipv_gs = sum(ipv_weight_gs * data_cross[4:, 7])
                # ipv_gs = max(sum(ipv_weight_gs * data_cross[4:, 7])-0.2, -math.pi*3/8)
            else:
                ipv_gs = 0
            self.lt_actual_trj = data_cross[:, 2:7]
            self.lt_actual_trj[:, 0] = self.lt_actual_trj[:, 0] - 13
            self.lt_actual_trj[:, 1] = self.lt_actual_trj[:, 1] - 7.8

            self.gs_actual_trj = data_cross[:, 9:14]
            self.gs_actual_trj[:, 0] = self.gs_actual_trj[:, 0] - 13
            self.gs_actual_trj[:, 1] = self.gs_actual_trj[:, 1] - 7.8

            self.case_len = np.size(data_cross, 0)

            return Scenario([init_position_lt, init_position_gs],
                            [init_velocity_lt, init_velocity_gs],
                            [init_heading_lt, init_heading_gs],
                            [ipv_lt, ipv_gs],
                            [controller_type_lt, controller_type_gs],
                            [init_acceleration_lt, init_acceleration_gs])

    def ipv_analysis(self):
        """
        estimate self ipv expression from a third-party view
        (On answering what is my ipv like from the view of others?)

        note that due to the dependence on [last self track], this process is not possible when using linearized
        optimization.
        Returns -------

        """
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
            agent_lt.estimate_self_ipv(track_lt_temp, track_gs_temp)
            ipv_collection_lt.append(agent_lt.ipv)
            ipv_error_collection_lt.append(agent_lt.ipv_error)

            agent_gs.estimate_self_ipv(track_gs_temp, track_lt_temp)
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

    def save_event_meta(self, num_failed, file_name, sheet_name):
        """

        Returns
        -------

        """

        "---- event data abstraction ----"
        # lt track (observed in simulation and ground truth in nds)
        ob_trj_lt = self.agent_lt.observed_trajectory[:, 0:2]
        nds_trj_lt = np.array(self.lt_actual_trj[:, 0:2])
        vel_ob_vel_norm_lt = np.linalg.norm(self.agent_lt.observed_trajectory[:, 2:4], axis=1)
        vel_nds_vel_norm_lt = np.linalg.norm(self.lt_actual_trj[:, 2:4], axis=1)
        acc_ob_lt = (vel_ob_vel_norm_lt[1:] - vel_ob_vel_norm_lt[:-1]) / 0.12
        acc_nds_lt = (vel_nds_vel_norm_lt[1:] - vel_nds_vel_norm_lt[:-1]) / 0.12
        jerk_ob_lt = (acc_ob_lt[1:] - acc_ob_lt[:-1]) / 0.12
        jerk_nds_lt = (acc_nds_lt[1:] - acc_nds_lt[:-1]) / 0.12

        # gs track (observed in simulation and ground truth in nds)
        ob_trj_gs = self.agent_gs.observed_trajectory[:, 0:2]
        nds_trj_gs = np.array(self.gs_actual_trj[:, 0:2])
        vel_ob_vel_norm_gs = np.linalg.norm(self.agent_gs.observed_trajectory[:, 2:4], axis=1)
        vel_nds_vel_norm_gs = np.linalg.norm(self.gs_actual_trj[:, 2:4], axis=1)
        acc_ob_gs = (vel_ob_vel_norm_gs[1:] - vel_ob_vel_norm_gs[:-1]) / 0.12
        acc_nds_gs = (vel_nds_vel_norm_gs[1:] - vel_nds_vel_norm_gs[:-1]) / 0.12
        jerk_ob_gs = (acc_ob_gs[1:] - acc_ob_gs[:-1]) / 0.12
        jerk_nds_gs = (acc_nds_gs[1:] - acc_nds_gs[:-1]) / 0.12

        "---- meta data ----"
        # 1
        case_id = self.case_id

        # ----semantic result
        seman_res_simu = self.semantic_result
        seman_res_nds = get_semantic_result(nds_trj_lt, nds_trj_gs, case_type='nds')
        # 2
        seman_right = bool(seman_res_nds == seman_res_simu)

        # ----velocity
        # 3
        mean_vel_simu_lt = vel_ob_vel_norm_lt.mean()
        # 4
        mean_vel_nds_lt = vel_nds_vel_norm_lt.mean()
        # 5
        mean_vel_simu_gs = vel_ob_vel_norm_gs.mean()
        # 6
        mean_vel_nds_gs = vel_nds_vel_norm_gs.mean()
        # 7
        vel_rmse_lt = np.sqrt(
            ((vel_ob_vel_norm_lt - vel_nds_vel_norm_lt) ** 2).sum()
            / np.size(vel_ob_vel_norm_lt, 0))
        # 8
        vel_rmse_gs = np.sqrt(
            ((vel_ob_vel_norm_gs - vel_nds_vel_norm_gs) ** 2).sum()
            / np.size(vel_ob_vel_norm_gs, 0))

        # ----position deviation
        # 9
        ave_pos_dev_lt = np.linalg.norm(ob_trj_lt - nds_trj_lt, axis=1).mean()
        # 10
        pos_rmse_lt = np.sqrt(
            ((ave_pos_dev_lt - np.linalg.norm(ob_trj_lt - nds_trj_lt, axis=1)) ** 2).sum()
            / np.size(ob_trj_lt, 0))
        # 11
        ave_pos_dev_gs = np.linalg.norm(ob_trj_gs - nds_trj_gs, axis=1).mean()
        # 12
        pos_rmse_gs = np.sqrt(
            ((ave_pos_dev_gs - np.linalg.norm(ob_trj_gs - nds_trj_gs, axis=1)) ** 2).sum()
            / np.size(ob_trj_gs, 0))

        # ----APET
        apet_nds, _, _ = cal_pet(nds_trj_lt, nds_trj_gs, type_cal='apet')
        # 13
        min_apet_nds = apet_nds.min()
        # 14
        mean_apet_nds = min(apet_nds.mean(), 15)

        apet_simu, _, _ = cal_pet(ob_trj_lt, ob_trj_gs, type_cal='apet')
        # 15
        min_apet_simu = apet_simu.min()
        # 16
        mean_apet_simu = min(apet_simu.mean(), 15)

        # ----PET
        # 17
        pet_nds, _ = cal_pet(nds_trj_lt, nds_trj_gs, type_cal='pet')
        # 18
        pet_simu, _ = cal_pet(ob_trj_lt, ob_trj_gs, type_cal='pet')

        # ----max acc and jerk
        # 19
        max_acc_simu_lt = max(max(acc_ob_lt), -min(acc_ob_lt))
        # 20
        max_acc_simu_gs = max(max(acc_ob_gs), -min(acc_ob_gs))
        # 21
        max_acc_nds_lt = max(max(acc_nds_lt), -min(acc_nds_lt))
        # 22
        max_acc_nds_gs = max(max(acc_nds_gs), -min(acc_nds_gs))

        # 23
        max_jerk_simu_lt = max(max(jerk_ob_lt), -min(jerk_ob_lt))
        # 24
        max_jerk_simu_gs = max(max(jerk_ob_gs), -min(jerk_ob_gs))
        # 25
        max_jerk_nds_lt = max(max(jerk_nds_lt), -min(jerk_nds_lt))
        # 26
        max_jerk_nds_gs = max(max(jerk_nds_gs), -min(jerk_nds_gs))

        "---- sava data ----"
        # prepare data
        df = pd.DataFrame([[case_id, seman_right, seman_res_simu,
                            mean_vel_simu_lt, mean_vel_nds_lt,
                            mean_vel_simu_gs, mean_vel_nds_gs,
                            vel_rmse_lt, vel_rmse_gs,
                            ave_pos_dev_lt, pos_rmse_lt,
                            ave_pos_dev_gs, pos_rmse_gs,
                            min_apet_nds, min_apet_simu,
                            mean_apet_nds, mean_apet_simu,
                            pet_nds, pet_simu,
                            max_acc_simu_lt, max_acc_simu_gs,
                            max_acc_nds_lt, max_acc_nds_gs,
                            max_jerk_simu_lt, max_jerk_simu_gs,
                            max_jerk_nds_lt, max_jerk_nds_gs
                            ], ],
                          columns=['case id', 'semantic', ' result',
                                   'simu. v. lt', 'nds v. lt',
                                   'simu. v. gs', 'nds v. gs',
                                   'v. RMSE lt', 'velocity RMSE gs',
                                   'ave. POS deviation lt', 'POS RMSE lt',
                                   'ave. POS deviation gs', 'POS RMSE gs',
                                   'MIN APET NDS', 'MIN APET SIMU',
                                   'mean APET NDS', 'mean APET SIMU',
                                   'PET NDS', 'PET SIMU',
                                   'MAX acc. SIMU lt', 'MAX acc. SIMU gs',
                                   'MAX acc. NDS lt', 'MAX acc. NDS gs',
                                   'MAX jerk SIMU lt', 'MAX jerk SIMU gs',
                                   'MAX jerk NDS lt', 'MAX jerk NDS gs',
                                   ])

        # write data
        if case_id == 0:
            header_flag = True
            start_row = 0
        else:
            header_flag = False
            start_row = case_id + 1

        with pd.ExcelWriter(file_name, mode='a', if_sheet_exists="overlay", engine="openpyxl") as writer:
            df.to_excel(writer, header=header_flag, index=False,
                        sheet_name=sheet_name,
                        startcol=0, startrow=start_row - num_failed)

    def save_conv_meta(self, trajectory_collection, file_name):
        """

        Returns
        -------

        """

        "---- event data abstraction ----"
        # data needed for each time step
        ave_plan_dev_lt = []  # distance between self-opt and interactive plan for quantifying interaction strength
        ave_plan_dev_gs = []
        min_overall_strength_ipv_lt = []
        min_overall_strength_ipv_gs = []
        min_self_strength_ipv_lt = []
        min_self_strength_ipv_gs = []
        ave_semantic_res = []
        semantic_res_std = []

        for t in range(len(trajectory_collection)):
            trj_package = trajectory_collection[t]

            # optimal plan in non-interactive simulation
            self_opt_lt = trj_package['non-interactive']['lt']
            self_opt_gs = trj_package['non-interactive']['gs']

            self_plan_dev_lt = []
            inter_plan_dev_lt = []
            self_plan_dev_gs = []
            inter_plan_dev_gs = []
            semantic_res = []
            for task_id in range(len(trj_package) - 1):
                # lt track (observed in simulation and ground truth in nds)
                self_plan_lt = trj_package['task' + str(task_id)]['lt-self']
                inter_plan_lt = trj_package['task' + str(task_id)]['lt-inter']
                nds_trj_lt = np.array(self.lt_actual_trj[:, 0:2])

                # gs track (observed in simulation and ground truth in nds)
                self_plan_gs = trj_package['task' + str(task_id)]['gs-self']
                inter_plan_gs = trj_package['task' + str(task_id)]['gs-inter']
                nds_trj_gs = np.array(self.gs_actual_trj[:, 0:2])

                "interaction strength indicated by plan deviation in each task"
                self_plan_dev_lt.append(np.mean(np.linalg.norm(self_opt_lt - self_plan_lt, axis=1)))
                inter_plan_dev_lt.append(np.mean(np.linalg.norm(self_opt_gs - inter_plan_lt, axis=1)))

                self_plan_dev_gs.append(np.mean(np.linalg.norm(self_opt_gs - self_plan_gs, axis=1)))
                inter_plan_dev_gs.append(np.mean(np.linalg.norm(self_opt_lt - inter_plan_gs, axis=1)))

                "closest time point"
                dis_lt = np.linalg.norm(np.array(self_plan_lt)-np.array(inter_plan_lt), axis=1)
                min_dis_index_lt = np.where(min(dis_lt) == dis_lt)
                dis_gs = np.linalg.norm(np.array(self_plan_gs) - np.array(inter_plan_gs), axis=1)
                min_dis_index_gs = np.where(min(dis_gs) == dis_gs)

                "yield or rush?"
                if self_plan_lt[min_dis_index_lt[0][0], 0] > inter_plan_lt[min_dis_index_lt[0][0], 0]:
                    semantic_res.append(1)  # left-turn agent rushed
                else:
                    semantic_res.append(0)

                if self_plan_gs[min_dis_index_gs[0][0], 0] < inter_plan_gs[min_dis_index_gs[0][0], 0]:
                    semantic_res.append(1)  # left-turn agent rushed
                else:
                    semantic_res.append(0)

            "save interaction strength for each agent"
            ave_plan_dev_lt.append(np.mean(self_plan_dev_lt))
            ave_plan_dev_gs.append(np.mean(self_plan_dev_gs))

            "find the ipv that minimize the ## overall ## interaction strength"
            overall_plan_dev_lt = np.array(self_plan_dev_lt) + np.array(inter_plan_dev_lt)
            min_overall_strength_id_lt = np.where(min(overall_plan_dev_lt) == overall_plan_dev_lt)
            if min_overall_strength_id_lt[0][0] in {0, 1, 2}:
                min_overall_strength_ipv_lt.append(-1)
            elif min_overall_strength_id_lt[0][0] in {3, 4, 5}:
                min_overall_strength_ipv_lt.append(0)
            else:
                min_overall_strength_ipv_lt.append(1)

            overall_plan_dev_gs = np.array(self_plan_dev_gs) + np.array(inter_plan_dev_gs)
            min_overall_strength_id_gs = np.where(min(overall_plan_dev_gs) == overall_plan_dev_gs)
            if min_overall_strength_id_gs[0][0] in {0, 3, 6}:
                min_overall_strength_ipv_gs.append(-1)
            elif min_overall_strength_id_gs[0][0] in {1, 4, 7}:
                min_overall_strength_ipv_gs.append(0)
            else:
                min_overall_strength_ipv_gs.append(1)

            "find the ipv that minimize the ## self ## interaction strength"
            min_self_strength_id_lt = np.where(min(self_plan_dev_lt) == self_plan_dev_lt)
            if min_self_strength_id_lt[0][0] in {0, 1, 2}:
                min_self_strength_ipv_lt.append(-1)
            elif min_self_strength_id_lt[0][0] in {3, 4, 5}:
                min_self_strength_ipv_lt.append(0)
            else:
                min_self_strength_ipv_lt.append(1)

            min_self_strength_id_gs = np.where(min(self_plan_dev_gs) == self_plan_dev_gs)
            if min_self_strength_id_gs[0][0] in {0, 3, 6}:
                min_self_strength_ipv_gs.append(-1)
            elif min_self_strength_id_gs[0][0] in {1, 4, 7}:
                min_self_strength_ipv_gs.append(0)
            else:
                min_self_strength_ipv_gs.append(1)

            "find semantic results"
            ave_semantic_res.append(np.mean(semantic_res))
            semantic_res_std.append(np.std(semantic_res))

        "---- sava data ----"
        # prepare data
        df = pd.DataFrame([[ave_plan_dev_lt[i], ave_plan_dev_gs[i],
                            min_overall_strength_ipv_lt[i], min_overall_strength_ipv_gs[i],
                            min_self_strength_ipv_lt[i], min_self_strength_ipv_gs[i],
                            ave_semantic_res[i], semantic_res_std[i],
                            ] for i in range(len(ave_plan_dev_lt))],
                          columns=['ave_plan_dev_lt', 'ave_plan_dev_gs',
                                   'min_overall_strength_ipv_lt', 'min_overall_strength_ipv_gs',
                                   'min_self_strength_ipv_lt', 'min_self_strength_ipv_gs',
                                   'ave_semantic_res', 'semantic_res_std',
                                   ])

        # write data
        with pd.ExcelWriter(file_name, engine="xlsxwriter") as writer:
            df.to_excel(writer, header=True, index=False,
                        sheet_name=str(self.case_id),
                        startcol=0)

            for column in df:
                column_length = max(df[column].astype(str).map(len).max(), len(column))
                col_idx = df.columns.get_loc(column)
                writer.sheets[str(self.case_id)].set_column(col_idx, col_idx, 1.2 * column_length)

            writer.save()


def get_semantic_result(track_lt, track_gs, case_type='simu'):
    """
        Identify semantic interaction results after simulation:
        1. crashed or not (not critical judgement)
        2. the left-turn vehicle yield or not
        """

    pos_delta = track_gs - track_lt
    dis_delta = np.linalg.norm(pos_delta[:, 0:2], axis=1)

    if min(dis_delta) < 1:
        semantic_result = 'crashed'
        # print('interaction is crashed. \n')
    elif case_type == 'nds':
        pos_y_larger = pos_delta[pos_delta[:, 1] > 0]
        if np.size(pos_y_larger, 0):

            "whether the LT vehicle yield"
            pos_x_larger = pos_y_larger[pos_y_larger[:, 0] > 0]
            yield_points = np.size(pos_x_larger, 0)
            if yield_points:
                semantic_result = 'yield'

                "where the interaction finish"
                ind_coll = np.where(pos_x_larger[0, 0] == pos_delta[:, 0])
                ind = ind_coll[0] - 1
                ending_point = {'lt': track_lt[ind, :],
                                'gs': track_gs[ind, :]}

                # print('LT vehicle yielded. \n')

            else:
                semantic_result = 'rush'
                # print('LT vehicle rushed. \n')
        else:
            pos_x_smaller = pos_delta[pos_delta[:, 0] < -1]
            if np.size(pos_x_smaller, 0):
                semantic_result = 'rush'
                # print('LT vehicle rushed. \n')
            else:
                semantic_result = 'unfinished'
                # print('interaction is not finished. \n')

    else:  # case_type == 'simu'
        pos_x_smaller = pos_delta[pos_delta[:, 0] < 0]
        if np.size(pos_x_smaller, 0):

            "whether the LT vehicle yield"
            pos_y_larger = pos_x_smaller[pos_x_smaller[:, 1] > 0]
            yield_points = np.size(pos_y_larger, 0)
            if yield_points:
                semantic_result = 'yield'

                "where the interaction finish"
                ind_coll = np.where(pos_y_larger[0, 0] == pos_delta[:, 0])
                ind = ind_coll[0] - 1
                ending_point = {'lt': track_lt[ind, :],
                                'gs': track_gs[ind, :]}

                # print('LT vehicle yielded. \n')

            else:
                semantic_result = 'rush'
                # print('LT vehicle rushed. \n')
        else:
            pos_y_smaller = pos_delta[pos_delta[:, 1] < 0]
            if np.size(pos_y_smaller, 0):
                semantic_result = 'rush'
                # print('LT vehicle rushed. \n')
            else:
                semantic_result = 'unfinished'
                # print('interaction is not finished. \n')

    return semantic_result


def run_interaction(simulator, case_id, task_id, returns):
    simu_sub = copy.deepcopy(simulator)
    simu_sub.tag += '-task' + str(task_id)
    simu_sub.interact(simu_step=1)
    simu_sub.visualize_single_step(file_path='figures/convergence anlysis-' + str(case_id) + '/')

    print('simulation finished: task id - ' + str(task_id))

    returns['task' + str(task_id)] = {'lt-self': simu_sub.agent_lt.trj_solution[:, 0:2],
                                      'lt-inter': simu_sub.agent_lt.estimated_inter_agent.trj_solution[:, 0:2],
                                      'gs-self': simu_sub.agent_gs.trj_solution[:, 0:2],
                                      'gs-inter': simu_sub.agent_gs.estimated_inter_agent.trj_solution[:, 0:2]}

    return returns


def main_simulate_T():
    """
    === main for simulating unprotected left-turning ===
    1. Set initial motion state before the simulation
    2. Change controller type by manually setting controller_type_xx as:
        * 'gt' is game-theoretic planner work by solving IBR
        * 'linear-gt' is a linear game-theoretic planner work by solving IBR
        * 'opt' is the optimal controller work by solving single optimization
    """

    tag = 'test'  # tag for data saving

    '---- set initial state of the left-turn vehicle ----'
    init_position_lt = [11.7, -5]
    init_velocity_lt = [1, 2]
    init_heading_lt = math.pi / 4
    ipv_lt = 2 * math.pi / 8
    controller_type_lt = 'linear-gt'

    '---- set initial state of the go-straight vehicle ----'
    init_position_gs = [19, -2]
    init_velocity_gs = [-3, 0]
    init_heading_gs = math.pi
    ipv_gs = 0 * math.pi / 8
    controller_type_gs = 'linear-gt'

    '---- generate scenario ----'
    simu_scenario = Scenario([init_position_lt, init_position_gs],
                             [init_velocity_lt, init_velocity_gs],
                             [init_heading_lt, init_heading_gs],
                             [ipv_lt, ipv_gs],
                             [controller_type_lt, controller_type_gs])

    simu = Simulator()
    simu.sim_type = 'simu'
    simu.initialize(simu_scenario, tag)

    time1 = time.perf_counter()
    simu.interact(simu_step=20, iter_limit=5)
    time2 = time.perf_counter()
    print('time consumption: ', time2 - time1)

    simu.semantic_result = get_semantic_result(simu.agent_lt.observed_trajectory,
                                               simu.agent_gs.observed_trajectory)
    simu.visualize_final_results('./')


def main_simulate_nds():
    """
       === main for simulating unprotected left-turning scenarios in Jianhe-Xianxia dataset ===
       1. Set case_id to get initial scenarios state of a single case
       2. Set dt as 0.12 (in agent.py) before simulation
       3. Change controller type by manually setting controller_type_xx as:
           * 'gt' is the game-theoretic planner work by solving IBR process
           * 'opt' is the optimal controller work by solving single optimization
           * 'idm'
           * 'replay'
           * 'linear-gt'
       """

    model_type = 'linear-gt'

    num_failed = 0
    # for case_id in range(130):
    for case_id in {51}:

        if case_id in {39, 45, 78, 93, 99}:  # interactions finished at the beginning
            num_failed += 1
            continue
        elif case_id in {12, 13, 24, 26, 28, 31, 32, 33, 37, 38, 46, 47, 48, 52, 56, 59, 65, 66, 69, 77, 82, 83, 84, 90,
                         91,
                         92, 94, 96, 97, 98, 100}:  # no path-crossing event
            num_failed += 1
        else:
            tag = 'simu-' + model_type + str(case_id)

            print('start case:' + tag)

            simu = Simulator(case_id=case_id)
            simu.sim_type = 'nds'
            controller_type_lt = model_type
            controller_type_gs = model_type
            simu_scenario = simu.read_nds_scenario(controller_type_lt, controller_type_gs)
            if simu_scenario:
                simu.initialize(simu_scenario, tag)
                simu.agent_gs.target = 'gs_nds'
                simu.agent_gs.estimated_inter_agent.target = 'lt_nds'
                simu.agent_lt.target = 'lt_nds'
                simu.agent_lt.estimated_inter_agent.target = 'gs_nds'

                simu.agent_gs.estimated_inter_agent.ipv = simu.agent_lt.ipv
                simu.agent_lt.estimated_inter_agent.ipv = simu.agent_gs.ipv

                try:
                    time1 = time.perf_counter()
                    simu.interact(simu_step=10, make_video=False, break_when_finish=False)
                    time2 = time.perf_counter()
                    print('time consumption: ', time2 - time1)
                    simu.semantic_result = get_semantic_result(simu.agent_lt.observed_trajectory[:, 0:2],
                                                               simu.agent_gs.observed_trajectory[:, 0:2],
                                                               case_type='nds')
                    simu.visualize_final_results(file_path='figures/')
                    # simu.save_metadata(num_failed,
                    #                    file_name='outputs/simulation_meta_data20220911.xlsx',
                    #                    sheet_name=model_type + ' simulation')
                except IndexError:
                    print('# ====Failed:' + tag + '==== #')
                    num_failed += 1
                    continue
            else:
                num_failed += 1


def main_test_lattice():
    """
    === main for testing a planner with replayed trajectory ===
        * set lt agent as the vehicle under test (ipv=pi/4)
    """

    bg_type = 'linear-gt'

    num_failed = 0

    # for case_id in range(130):
    for case_id in {51}:

        if case_id in {39, 45, 78, 93, 99}:  # interactions finished at the beginning
            num_failed += 1
            continue
        elif case_id in {12, 13, 24, 26, 28, 31, 32, 33, 37, 38, 46, 47, 48, 52, 56, 59, 65, 66, 69, 77, 82, 83, 84, 90,
                         91,
                         92, 94, 96, 97, 98, 100}:  # no path-crossing event
            num_failed += 1
            continue
        else:
            tag = bg_type + '-test-lt-lattice' + str(case_id)
            # tag = bg_type + '-' + str(case_id)
            print('start case:' + tag)

            simu = Simulator(case_id=case_id)
            simu.sim_type = 'nds'
            controller_type_lt = 'lattice'
            controller_type_gs = bg_type
            simu_scenario = simu.read_nds_scenario(controller_type_lt, controller_type_gs)
            if simu_scenario:
                simu.initialize(simu_scenario, tag)
                simu.agent_gs.target = 'gs_nds'
                simu.agent_gs.estimated_inter_agent.target = 'lt_nds'
                simu.agent_lt.target = 'lt_nds'
                simu.agent_lt.estimated_inter_agent.target = 'gs_nds'
                try:

                    time1 = time.perf_counter()
                    # simu.interact(simu_step=2, make_video=False, break_when_finish=False)
                    simu.interact(simu_step=int(simu.case_len / 2 + 5), make_video=False, break_when_finish=False)
                    time2 = time.perf_counter()
                    print('time consumption: ', time2 - time1)
                    simu.semantic_result = get_semantic_result(simu.agent_lt.observed_trajectory[:, 0:2],
                                                               simu.agent_gs.observed_trajectory[:, 0:2],
                                                               case_type='nds')
                    simu.visualize_final_results(file_path='figures/')
                    # simu.visualize(file_path='figures/' + bg_type + ' test lt lattice/')
                    # simu.save_metadata(num_failed,
                    #                    file_name='outputs/test_meta_data20220912-4.xlsx',
                    #                    sheet_name=bg_type + '-test gs-lattice')
                except IndexError:
                    print('# ====Failed:' + tag + '==== #')
                    num_failed += 1
                    continue
            else:
                num_failed += 1


def main_analyze_interaction():
    """
      === main for analyzing interaction strength and convergence in nds ===
          * set all agents as selfish and solve game for them
      """

    bg_type = 'linear-gt'
    num_failed = 0

    # for case_id in range(130):
    for case_id in {51}:

        if case_id in {39, 45, 78, 93, 99}:  # interactions finished at the beginning
            num_failed += 1
            continue
        elif case_id in {12, 13, 24, 26, 28, 31, 32, 33, 37, 38, 46, 47, 48, 52, 56, 59, 65, 66, 69, 77, 82, 83, 84, 90,
                         91, 92, 94, 96, 97, 98, 100}:  # no path-crossing event
            num_failed += 1
            continue
        else:

            print('start case:' + str(case_id))

            simu = Simulator(case_id=case_id)
            simu.sim_type = 'nds'
            controller_type_lt = bg_type
            controller_type_gs = bg_type
            simu.read_nds_scenario(controller_type_lt, controller_type_gs)
            for t in range(simu.case_len - 1):

                print('time_step: ', t, '/', simu.case_len)

                simu.simu_time = t
                tag = 'convergence analysis-case' + str(case_id) + '-t' + str(t)
                simu_scenario = simu.read_nds_scenario(controller_type_lt, controller_type_gs, t=t)
                if simu_scenario:
                    simu.initialize(simu_scenario, tag)
                    simu.agent_gs.estimated_inter_agent.ipv = simu.agent_lt.ipv
                    simu.agent_lt.estimated_inter_agent.ipv = simu.agent_gs.ipv

                    simu.agent_gs.target = 'gs_nds'
                    simu.agent_gs.estimated_inter_agent.target = 'lt_nds'
                    simu.agent_lt.target = 'lt_nds'
                    simu.agent_lt.estimated_inter_agent.target = 'gs_nds'

                    simu.agent_gs.ipv = 1.5 * math.pi / 4
                    simu.agent_gs.estimated_inter_agent.ipv = 0 * math.pi / 4
                    simu.agent_lt.ipv = 0 * math.pi / 4
                    simu.agent_lt.estimated_inter_agent.ipv = 1.5 * math.pi / 4

                    # time1 = time.perf_counter()
                    simu.interact(simu_step=1, make_video=False, break_when_finish=False)
                    # time2 = time.perf_counter()
                    # print('time consumption: ', time2 - time1)

                    simu.visualize_single_step(file_path='figures/convergence anlysis-' + str(case_id) + '/')


def main_analyze_interaction_parallel():
    """
      === main for analyzing interaction strength and convergence in nds ===
          * set all agents as selfish and solve game for them
      """

    bg_type = 'linear-gt'
    num_failed = 0

    for case_id in range(130):
    # for case_id in {51}:

        if case_id in {39, 45, 78, 93, 99}:  # interactions finished at the beginning
            num_failed += 1
            continue
        elif case_id in {12, 13, 24, 26, 28, 31, 32, 33, 37, 38, 46, 47, 48, 52, 56, 59, 65, 66, 69, 77, 82, 83, 84, 90,
                         91, 92, 94, 96, 97, 98, 100}:  # no path-crossing event
            num_failed += 1
            continue
        else:

            print('start case:' + str(case_id))

            simu = Simulator(case_id=case_id)
            simu.sim_type = 'nds'
            controller_type_lt = bg_type
            controller_type_gs = bg_type
            simu.read_nds_scenario(controller_type_lt, controller_type_gs)

            trajectory_collection = []

            for t in range(simu.case_len - 1):
            # for t in range(2):

                print('time_step: ', t, '/', simu.case_len)

                simu.simu_time = t
                tag = 'conv analysis-case' + str(case_id) + '-t' + str(t)
                simu_scenario = simu.read_nds_scenario(controller_type_lt, controller_type_gs, t=t)
                traj_coll_temp = {}

                if simu_scenario:
                    simu.initialize(simu_scenario, tag)

                    simu.agent_gs.target = 'gs_nds'
                    simu.agent_gs.estimated_inter_agent.target = 'lt_nds'
                    simu.agent_lt.target = 'lt_nds'
                    simu.agent_lt.estimated_inter_agent.target = 'gs_nds'

                    # worker pairs for generating trajectory under different ipv combinations
                    tasks = []
                    manager = Manager()
                    traj_coll_temp = manager.dict()
                    task_id = 0
                    for lt_ipv in {-1, 0, 1}:
                        for gs_ipv in {-1, 0, 1}:
                            simu_temp = copy.deepcopy(simu)

                            simu_temp.agent_gs.ipv = gs_ipv * math.pi / 4
                            simu_temp.agent_gs.estimated_inter_agent.ipv = lt_ipv * math.pi / 4
                            simu_temp.agent_lt.ipv = lt_ipv * math.pi / 4
                            simu_temp.agent_lt.estimated_inter_agent.ipv = gs_ipv * math.pi / 4

                            task = Process(target=run_interaction, args=(simu_temp, case_id, task_id, traj_coll_temp))
                            task.start()
                            tasks.append(task)
                            task_id += 1

                    for task in tasks:
                        task.join()

                    # generate selfish plan
                    simu.interact(simu_step=1, interactive=False)
                    simu.tag += '-self-opt'
                    # simu.visualize_single_step(file_path='figures/convergence anlysis-' + str(case_id) + '/')
                    traj_coll_temp['non-interactive'] = {'lt': simu.agent_lt.trj_solution[:, 0:2],
                                                         'gs': simu.agent_gs.trj_solution[:, 0:2]}

                trajectory_collection.append(traj_coll_temp)
            simu.save_conv_meta(trajectory_collection, file_name='outputs/conv_meta_data20220925.xlsx')


if __name__ == '__main__':
    'simulate unprotected left-turn at a T-intersection'
    # main_simulate_T()

    'simulate with nds data from Jianhe-Xianxia intersection'
    # main_simulate_nds()

    'test lattice planner with different background vehicles'
    # main_test_lattice()

    'solve game for scenario initialized by nds cases'
    # main_analyze_interaction()
    main_analyze_interaction_parallel()
