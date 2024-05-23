from acados_template import AcadosOcp, AcadosOcpSolver
import numpy as np
from scipy.linalg import block_diag
from .quad_model import Quadrotor
# from quad_model import Quadrotor


class MPC():
    def __init__(self, use_cython: bool, sim: bool, quadrotor: Quadrotor, T_horizon: float, N_horizon: int):
        self.quadrotor = quadrotor
        self.T_horizon = T_horizon
        self.N_horizon = N_horizon
        self.num_steps = N_horizon

        self.use_cython = use_cython

        self.ocp, self.ocp_solver = self.generate_mpc()

        self.g = 9.806
        self.m = 1.535 if sim else 1.69


    def generate_mpc(self):
        # create ocp object to formulate the optimization problem
        ocp = AcadosOcp()
        ocp.model = self.quadrotor.export_robot_model() # get model

        # set dimensions
        nx = ocp.model.x.size()[0]
        nu = ocp.model.u.size()[0]
        ny = nx + nu
        ny_e = nx
        ocp.dims.N = self.N_horizon

        # set cost
        Q_mat = np.diag([10., 10., 10.,   0., 0., 0.,   0., 0., 10.]) # [x, y, z, vx, vy, vz, roll, pitch, yaw]
        R_mat = np.diag([10., 10., 10., 10.]) # [thrust, rolldot, pitchdot, yawdot]


        ocp.cost.cost_type = "LINEAR_LS"
        # ocp.cost.cost_type_e = "LINEAR_LS"

        ocp.cost.W_e = Q_mat
        ocp.cost.W = block_diag(Q_mat, R_mat)

        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)

        Vu = np.zeros((ny, nu))
        Vu[nx : nx + nu, 0:nu] = np.eye(nu)
        ocp.cost.Vu = Vu

        ocp.cost.Vx_e = np.eye(nx)

        ocp.cost.yref = np.zeros((ny,))
        ocp.cost.yref_e = np.zeros((ny_e,))

        # set constraints
        max_rate = 0.8
        max_thrust = 27.0
        min_thrust = 0.0
        ocp.constraints.lbu = np.array([min_thrust, -max_rate, -max_rate, -max_rate])
        ocp.constraints.ubu = np.array([max_thrust, max_rate, max_rate, max_rate])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])

        X0 = np.array([0.0, 0.0, 0.0,    0.0, 0.0, 0.0,    0.0, 0.0, 0.0])  # Intitalize the states [x,y,v,th,th_d]
        ocp.constraints.x0 = X0

        # set options
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # 'GAUSS_NEWTON', 'EXACT'
        ocp.solver_options.integrator_type = "IRK"
        ocp.solver_options.nlp_solver_type = "SQP"  # SQP_RTI, SQP
        ocp.solver_options.nlp_solver_max_iter = 400
        # ocp.solver_options.levenberg_marquardt = 1e-2

        # set prediction horizon
        ocp.solver_options.tf = self.T_horizon

        if self.use_cython:
            AcadosOcpSolver.generate(ocp, json_file='acados_ocp.json')
            AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
            ocp_solver = AcadosOcpSolver.create_cython_solver('acados_ocp.json')
        else: # ctypes
            ## Note: skip generate and build assuming this is done before (in cython run)
            ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json', build=False, generate=False)

        ocp_solver.reset()

        return ocp, ocp_solver

    def solve_mpc(self, x0, x_ref):
        ocp = self.ocp
        acados_ocp_solver = self.ocp_solver
        m = self.m
        g = self.g

        N_horizon = self.N_horizon
        nx = ocp.model.x.size()[0]
        nu = ocp.model.u.size()[0]
        # print(nx, nu)
        xcurrent = x0

        # initialize solver
        for stage in range(N_horizon + 1):
            acados_ocp_solver.set(stage, "x", 0.0 * np.ones(xcurrent.shape))
        for stage in range(N_horizon):
            acados_ocp_solver.set(stage, "u", np.zeros((nu,)))

        # set initial state constraint
        acados_ocp_solver.set(0, "lbx", xcurrent)
        acados_ocp_solver.set(0, "ubx", xcurrent)

        # update yref
        for j in range(N_horizon):
            u_ref = np.array([m*g, 0, 0, 0])
            y_ref = np.hstack((x_ref, u_ref))
            # yref = np.array([0, 0, 1.5,    0, 0, 0,    0 ,0 ,0,   1.535*9.806, 0, 0, 0])
            acados_ocp_solver.set(j, "yref", y_ref)

            # if j == 0:
            #     print(u_ref)
            #     # print(x_ref.shape)
            #     print(y_ref.shape)
            #     print(yref.shape)



        yref_N = x_ref
        acados_ocp_solver.set(N_horizon, "yref", yref_N)
        # print(yref.shape)

        # solve ocp
        status = acados_ocp_solver.solve()
        u = acados_ocp_solver.get(0, "u")
        

        return u




# sim = True
# Thorizon = 2.0
# Nhorizon = 50
# use_cython = True
# mpc_solver = MPC(use_cython=use_cython, sim=sim, quadrotor=Quadrotor(sim), T_horizon=Thorizon, N_horizon=Nhorizon)
# x0 = np.zeros(9)
# xref = np.array([0, 0, -3.5, 0, 0, 0, 0, 0, 0])
# u = mpc_solver.solve_mpc(x0, xref)
# print(u)