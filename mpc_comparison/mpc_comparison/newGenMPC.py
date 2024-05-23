from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import numpy as np
from casadi import SX, vertcat, horzcat, diag, inv_minor, cross, sqrt,  cos, sin, norm_2, tanh, GenMX_zeros
from scipy.linalg import block_diag
from .workingModel import Quadrotor
from threading import Thread
from time import sleep, time
import time
from pathlib import Path
import importlib
import sys

class QuadrotorMPC2:
    def __init__(self, generate_c_code: bool, quadrotor: Quadrotor, horizon: float, num_steps: int):
        
        self.model = AcadosModel()
        self.quad = quadrotor
        self.model_name = 'holybro'
        self.horizon = horizon
        self.num_steps = num_steps


        self.ocp_solver = None
        self.generate_c_code = generate_c_code
        # self.acados_generated_files_path = Path(__file__).parent.parent.resolve() / 'acados_generated_files'
        code_export_directory = str(self.model_name) + '_mpc' + '_c_generated_code'
        # ocp.code_export_directory = code_export_directory
        if self.generate_c_code:
            self.generate_mpc()
        else:
            try:
                sys.path.append(code_export_directory)
                acados_ocp_solver_pyx = importlib.import_module('acados_ocp_solver_pyx')
                self.ocp_solver = acados_ocp_solver_pyx.AcadosOcpSolverCython(self.model_name, 'SQP', self.num_steps)

            except ImportError:
                self.generate_mpc()

    def generate_mpc(self):
        f_expl, x, u = self.quad.dynamics()
        
        # Define the Acados model 
        model = AcadosModel()
        model.f_expl_expr = f_expl
        model.x = x
        model.u = u
        model.name = self.model_name

        # Define the optimal control problem
        ocp = AcadosOcp()
        ocp.model = model

        ocp.code_export_directory = str(self.model_name) + '_mpc' + '_c_generated_code'
        nx = model.x.size()[0] # number of states
        nu = model.u.size()[0] # number of controls
        ny = nx + nu  # size of intermediate cost reference vector in least squares objective
        ny_e = nx # size of terminal reference vector

        N = self.num_steps
        Tf = self.horizon
        ocp.dims.N = N
        ocp.solver_options.tf = Tf

        Q = np.diag([50., 50., 100., 2., 2., 2., 1., 1., 20.])
        R = diag(horzcat(1., 1., 1., 1.))
        W = block_diag(Q,R)

        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.Vx = np.vstack([np.identity(nx), np.zeros((nu,nx))])
        ocp.cost.Vu = np.vstack([np.zeros((nx,nu)), np.identity(nu)])
        ocp.cost.W = W
        ocp.cost.yref = np.zeros(ny)

        ocp.cost.cost_type_e = 'LINEAR_LS'
        ocp.cost.W_e = Q
        ocp.cost.Vx_e = np.vstack([np.identity(nx)])
        ocp.cost.yref_e = np.zeros(ny_e)


        # bounds on control
        max_rate = 2.0
        max_thrust = 27.0
        min_thrust = 0.0
        ocp.constraints.lbu = np.array([min_thrust, -max_rate, -max_rate, -max_rate])
        ocp.constraints.ubu = np.array([max_thrust, max_rate, max_rate, max_rate])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])

        #initial state
        ocp.constraints.x0 = np.zeros(9)


        # solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.nlp_solver_type = 'SQP'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.tol = 1e-6
        ocp.solver_options.qp_tol = 1e-6
        ocp.solver_options.nlp_solver_max_iter = 50
        ocp.solver_options.qp_solver_iter_max = 50
        ocp.solver_options.print_level = 0




        # create ocp solver
        json_file = str(self.model_name) + '_mpc' + '_acados_ocp.json'
        AcadosOcpSolver(ocp, json_file= json_file)
        AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
        sys.path.append(ocp.code_export_directory)
        acados_ocp_solver_pyx = importlib.import_module('acados_ocp_solver_pyx')
        self.ocp_solver = acados_ocp_solver_pyx.AcadosOcpSolverCython(self.model_name, 'SQP', self.num_steps)

    def solve_mpc_control(self, x0, xd, xd_e):
        hover_input = np.array([self.quad.g*self.quad.m, 0., 0., 0.])

        N = self.num_steps
        nx = len(x0)
        nu = 4

        print(f"{x0 = }")
        print(f"yref0 = {np.array([*xd[:,0], *hover_input])}")


        if xd.shape[1] != N:
            raise ValueError("The reference trajectory should have the same length as the number of steps")


        for i in range(N):
            self.ocp_solver.set(i, 'y_ref', np.array([*xd[:,i], *hover_input]))

        self.ocp_solver.set(N, 'y_ref', xd_e)
        self.ocp_solver.set(0, 'lbx', x0)
        self.ocp_solver.set(0, 'ubx', x0)


        x_mpc = np.zeros((N+1, nx))
        u_mpc = np.zeros((N, nu))
        start_time = time.time()
        status = self.ocp_solver.solve()
        print(f"mpc_calc_time: {time.time() - start_time}")

        # extract state and control solution from solver
        for i in range(N):
            x_mpc[i,:] = self.ocp_solver.get(i, "x")
            u_mpc[i,:] = self.ocp_solver.get(i, "u")
        x_mpc[N,:] = self.ocp_solver.get(N, "x")
        return status, x_mpc, u_mpc

