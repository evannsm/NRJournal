import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, VehicleRatesSetpoint, VehicleCommand, VehicleStatus, VehicleOdometry, TrajectorySetpoint, RcChannels
from std_msgs.msg import Float64MultiArray

from tf_transformations import euler_from_quaternion
from math import sqrt
import math as m
import numpy as np
import time

from acados_template import AcadosModel
from acados_template import AcadosOcp, AcadosOcpSolver
from casadi import SX, vertcat, sin, cos

from scipy.linalg import block_diag
import importlib
import sys
from pathlib import Path

class OffboardControl(Node):
    """Node for controlling a vehicle in offboard mode."""
    def __init__(self) -> None:
        super().__init__('offboard_control_takeoff_and_land')

        class Quadrotor:
            def __init__(self, sim: bool):
                self.sim = sim
                self.g = 9.806 #gravity
                self.m = 1.535 if sim else 1.69 #mass

            def export_robot_model(self) -> AcadosModel:

            # set up states & controls
                
                # states
                x = SX.sym("x")
                y = SX.sym("y")
                z = SX.sym("z")
                vx = SX.sym("x_d")
                vy = SX.sym("y_d")
                vz = SX.sym("z_d")
                roll = SX.sym("roll")
                pitch = SX.sym("pitch")
                yaw = SX.sym("yaw")

                # controls
                thrust = SX.sym('thrust')
                rolldot = SX.sym('rolldot')
                pitchdot = SX.sym('pitchdot')
                yawdot = SX.sym('yawdot')

                #state vector
                x = vertcat(x, y, z, vx, vy, vz, roll, pitch, yaw)

                # control vector
                u = vertcat(thrust, rolldot, pitchdot, yawdot)


                # xdot
                x_dot = SX.sym("x_dot")
                y_dot = SX.sym("y_dot")
                z_dot = SX.sym("z_dot")
                vx_dot = SX.sym("vx_dot")
                vy_dot = SX.sym("vy_dot")
                vz_dot = SX.sym("vz_dot")
                roll_dot = SX.sym("roll_dot")
                pitch_dot = SX.sym("pitch_dot")
                yaw_dot = SX.sym("yaw_dot")
                xdot = vertcat(x_dot, y_dot, z_dot, vx_dot, vy_dot, vz_dot, roll_dot, pitch_dot, yaw_dot)

                # algebraic variables
                # z = None

                # parameters
                p = []

            # dynamics
                # define trig functions
                sr = sin(roll)
                sy = sin(yaw)
                sp = sin(pitch)
                cr = cos(roll)
                cp = cos(pitch)
                cy = cos(yaw)

                # define dynamics
                pxdot = vx
                pydot = vy
                pzdot = vz
                vxdot = -(thrust/self.m) * (sr*sy + cr*cy*sp);
                vydot = -(thrust/self.m) * (cr*sy*sp - cy*sr);
                vzdot = self.g - (thrust/self.m) * (cr*cp);
                rolldot = rolldot
                pitchdot = pitchdot
                yawdot = yawdot

                # EXPLICIT FORM
                f_expl = vertcat(pxdot, pydot, pzdot, vxdot, vydot, vzdot, rolldot, pitchdot, yawdot)

                # IMPLICIT FORM
                f_impl = xdot - f_expl

                model = AcadosModel()
                model.f_impl_expr = f_impl
                model.f_expl_expr = f_expl
                model.x = x
                model.xdot = xdot
                model.u = u
                # model.z = z
                model.p = p
                model.name = "quad"

                return model

        class MPC():
            def __init__(self, generate_c_code: bool, quadrotor: Quadrotor, T_horizon: float, N_horizon: int, code_export_directory : Path=Path('acados_generated_files')):
                self.generate_c_code = generate_c_code
                self.quadrotor = quadrotor
                self.T_horizon = T_horizon  
                self.N_horizon = N_horizon

                self.ocp_solver = None
                self.model_name = "HOLIER_BRO"

                # Where to save the generated files in self.generate_mpc(). Aslo serves as base for loading everything later
                self.acados_generated_files_path = Path(str(Path(__file__).parent.resolve()) + "/" + self.model_name + '_mpc' + '_c_generated_code')

                if self.generate_c_code:
                    print(f"You Want to generate c code!")
                    self.generate_mpc()
                    print("C code generated successfully")
                    # exit(0)
                else:
                    print("Trying to import acados cython module...\n\n")
                    try:
                        ocp = self.create_ocp_solver_description()
                        self.ocp = ocp

                        self.import_module = self.model_name + '_mpc' + '_c_generated_code' + '.acados_ocp_solver_pyx' #module is simply holier_bro_mpc_c_generated_code.acados_ocp_solver_pyx
                        self.import_module = self.import_module.replace('/', '.').replace('.pyx', '') #replace / with . and remove .pyx
                        self.pyx_module_path  = str(self.acados_generated_files_path.parent) #path is the same as the acados_generated_files_path but one level up

                        print(f"\n\n {self.acados_generated_files_path=}\n\n")
                        print(f"\n\n {self.import_module=}\n\n")
                        print(f"\n\n {self.pyx_module_path=}\n\n")

                        sys.path.append(self.pyx_module_path)  # Ensure the module path is in the python path
                        # print(f"\n\n {sys.path=}\n\n")

                        acados_ocp_solver_pyx = importlib.import_module(self.import_module) #import the module from this path (brings us basically back to self.acados_generated_files_path but needs to be done this way)
                        self.ocp_solver = acados_ocp_solver_pyx.AcadosOcpSolverCython(self.model_name, 'SQP', self.N_horizon) # load the module which is the solver
                        print("Acados cython module imported successfully")
                        # exit(0)

                    except ImportError:
                        print("Acados cython code doesn't exit. Generating cython code now...")
                        self.generate_mpc()
                        print("Cython code generated successfully")
                        # exit(0)


            def create_ocp_solver_description(self) -> AcadosOcp:
                # create ocp object to formulate the optimization problem
                ocp = AcadosOcp()
                ocp.code_export_directory = self.acados_generated_files_path




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


                return ocp
            
            def generate_mpc(self):
                self.ocp = self.create_ocp_solver_description()
                # print("ocp.code_export_directory: ", ocp.code_export_directory)


                # print(self.acados_generated_files_path.is_dir())
                json_file = str(self.model_name) + '_mpc' + '_acados_ocp.json'

                AcadosOcpSolver.generate(self.ocp, json_file=json_file)
                AcadosOcpSolver.build(self.ocp.code_export_directory, with_cython=True)
                self.ocp_solver = AcadosOcpSolver.create_cython_solver(json_file)

            def solve_mpc(self, sim: bool, x0, x_ref):
                g = 9.806
                m = 1.535 if sim else 1.69

                # prepare simulation
                ocp = self.ocp
                acados_ocp_solver = self.ocp_solver

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


        mpc_solver = MPC(False, Quadrotor(True), 2, 50)

        x0 = np.zeros(9)
        xref = np.array([0, 0, -3.5, 0, 0, 0, 0, 0, 0])

        u = mpc_solver.solve_mpc(True, x0, xref)

        print(u)
        exit(0)


# Entry point of the code -> Initializes the node and spins it
def main(args=None) -> None:
    print(f'Initializing "{__name__}" node for offboard control')
    rclpy.init(args=args)
    offboard_control = OffboardControl()
    rclpy.spin(offboard_control)
    offboard_control.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
    try:
        main()
    except Exception as e:
        print(e)