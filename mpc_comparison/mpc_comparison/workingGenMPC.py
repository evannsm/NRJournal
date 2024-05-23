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
        
        # Define Acados Model
        model = AcadosModel()   
        model.f_expl_expr = f_expl
        model.x = x
        model.u = u
        model.name = self.model_name

        # Define Acados OCP
        ocp = AcadosOcp()
        ocp.model = model

        ocp.code_export_directory = str(self.model_name) + '_mpc' + '_c_generated_code'
        nx = model.x.size()[0]
        nu = model.u.size()[0]


        Tf = self.horizon
        N = self.num_steps
        ocp.dims.N = N
        ocp.solver_options.tf = Tf



        W = np.diag([50., 50., 50., 10., 10., 10., 5., 5., 5.])
        # R = diag(horzcat(5., 5., 5., 1.))
        # W = block_diag(Q)

        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'EXTERNAL'
        ocp.cost.Vx = np.diag([1.,1.,1.,1.,1.,1.,1.,1.,1.])
        ocp.cost.Vu = np.zeros((9, 4))
        ocp.cost.W = W
        ocp.cost.yref = np.zeros(9)

        # y = vertcat(x[:-1], u)
        # xref = vertcat(pxr, pyr, pzr, vxr, vyr, vzr)
        # yref = vertcat(xref,uref)

        # ocp.model.cost_expr_ext_cost = 0.5 * (y - yref).T @ W @ (y - yref)
        ocp.model.cost_expr_ext_cost_e = 0.



        # bounds on control
        max_rate = 0.8
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
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.print_level = 0
        ocp.solver_options.nlp_solver_type = 'SQP'

        # create ocp solver
        json_file = str(self.model_name) + '_mpc' + '_acados_ocp.json'
        AcadosOcpSolver(ocp, json_file= json_file)
        AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
        sys.path.append(ocp.code_export_directory)
        acados_ocp_solver_pyx = importlib.import_module('acados_ocp_solver_pyx')
        self.ocp_solver = acados_ocp_solver_pyx.AcadosOcpSolverCython(self.model_name, 'SQP', self.num_steps)

    def solve_mpc_control(self, x0, xd):
        N = self.num_steps
        nx = len(x0)
        nu = 4


        if xd.shape[1] != N:
            raise ValueError("The reference trajectory should have the same length as the number of steps")


        for i in range(N):
            # set up state and control reference vectors
            x_ref = xd[:, i]

            # u_ref = np.array((self.quad.m*self.quad.g, 0.0, 0.0, 0.0))
            # y_ref = np.hstack((x_ref, u_ref))
            y_ref = x_ref
            # y_ref = np.expand_dims(y_ref, axis=0).T

            # if i == 0:
            #     print("HERE")
            #     print(y_ref.shape)
            #     print(y_ref)



            # self.ocp_solver.set(i, 'lbx', x0)
            # self.ocp_solver.set(i, 'ubx', x0)
            self.ocp_solver.set(i, 'y_ref', y_ref)

        x_mpc = np.zeros((N+1, nx))
        u_mpc = np.zeros((N, nu))

        self.ocp_solver.set(0, 'lbx', x0)
        self.ocp_solver.set(0, 'ubx', x0)

        start_time = time.time()
        status = self.ocp_solver.solve()
        x_mpc = self.ocp_solver.get(0, 'x')
        u_mpc = self.ocp_solver.get(0, 'u')
        print(f"timel: {time.time() - start_time}")

        return status, x_mpc, u_mpc

