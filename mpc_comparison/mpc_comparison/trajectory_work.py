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

        # Figure out if in simulation or hardware mode to set important variables to the appropriate values
        self.sim = bool(int(input("Are you using the simulator? Write 1 for Sim and 0 for Hardware: ")))
        print(f"{'SIMULATION' if self.sim else 'HARDWARE'}")


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
            def __init__(self, generate_c_code: bool, quadrotor: Quadrotor, horizon: float, num_steps: int):
                self.generate_c_code = generate_c_code
                self.quad = quadrotor
                self.horizon = horizon  
                self.num_steps = num_steps
                self.model_name = "HOLIER_BRO"

                # Where to save the generated files in self.generate_mpc(). Aslo serves as base for loading everything later
                self.acados_generated_files_path = Path(str(Path(__file__).parent.resolve()) + "/" + self.model_name + '_mpc' + '_c_generated_code')
                self.ocp = self.create_ocp_solver_description()



                if self.generate_c_code:
                    print(f"You Want to generate c code!")
                    self.generate_mpc()
                    print("C code generated successfully")
                    # exit(0)
                else:
                    print("Trying to import acados cython module...\n\n")
                    try:
                        self.import_module = self.model_name + '_mpc' + '_c_generated_code' + '.acados_ocp_solver_pyx' #module is simply holier_bro_mpc_c_generated_code.acados_ocp_solver_pyx
                        self.import_module = self.import_module.replace('/', '.').replace('.pyx', '') #replace / with . and remove .pyx
                        self.pyx_module_path  = str(self.acados_generated_files_path.parent) #path is the same as the acados_generated_files_path but one level up

                        print(f"\n\n {self.acados_generated_files_path=}\n\n")
                        print(f"\n\n {self.import_module=}\n\n")
                        print(f"\n\n {self.pyx_module_path=}\n\n")

                        sys.path.append(self.pyx_module_path)  # Ensure the module path is in the python path
                        # print(f"\n\n {sys.path=}\n\n")

                        acados_ocp_solver_pyx = importlib.import_module(self.import_module) #import the module from this path (brings us basically back to self.acados_generated_files_path but needs to be done this way)
                        self.ocp_solver = acados_ocp_solver_pyx.AcadosOcpSolverCython(self.model_name, 'SQP', self.num_steps) # load the module which is the solver
                        print("Acados cython module imported successfully")
                        # exit(0)

                    except ImportError:
                        print("Acados cython code doesn't exit. Generating cython code now...")
                        self.generate_mpc()
                        print("Cython code generated successfully")
                        # exit(0)

            def generate_mpc(self):
                # print(self.acados_generated_files_path.is_dir())
                json_file = str(self.model_name) + '_mpc' + '_acados_ocp.json'

                AcadosOcpSolver.generate(self.ocp, json_file=json_file)
                AcadosOcpSolver.build(self.ocp.code_export_directory, with_cython=True)
                self.ocp_solver = AcadosOcpSolver.create_cython_solver(json_file)

            def create_ocp_solver_description(self) -> AcadosOcp:
                # create ocp object to formulate the optimization problem
                ocp = AcadosOcp()
                ocp.code_export_directory = self.acados_generated_files_path
                ocp.model = self.quad.export_robot_model() # get model

                # set dimensions
                nx = ocp.model.x.size()[0]
                nu = ocp.model.u.size()[0]
                ny = nx + nu
                ny_e = nx
                ocp.dims.N = self.num_steps

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
                ocp.solver_options.tf = self.horizon

                return ocp

            def solve_mpc_control(self, x0, xd):
                N = self.num_steps
                nx = len(x0)
                nu = 4


                if xd.shape[0] != N:
                    raise ValueError("The reference trajectory should have the same length as the number of steps")


                for i in range(N):
                    # set up state and control reference vectors
                    x_ref = xd[i,:]

                    u_ref = np.array((self.quad.m*self.quad.g, 0.0, 0.0, 0.0))
                    y_ref = np.hstack((x_ref, u_ref))
                    # y_ref = x_ref
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
                print(f"mpc compute time: {time.time() - start_time}")

                return status, x_mpc, u_mpc

        self.num_steps = 20
        self.horizon = 1.0
        quad = Quadrotor(self.sim)
        mpc_solver = MPC(False, quad, self.horizon, self.num_steps)
        x0 = np.zeros(9)
        xref = self.hover_ref_func(1)
        print(f"{xref.shape=}")
        status, x_mpc, u_mpc = mpc_solver.solve_mpc_control(x0, xref)
        print(u_mpc)

        self.timefromstart = 0.0
        reffunc = self.fig8_vert_ref_func_tall()
        print(f"{reffunc.shape=}")
        status, x_mpc, u_mpc = mpc_solver.solve_mpc_control(x0, reffunc)
        print(u_mpc)



        exit(0)


# ~~ The following functions are reference trajectories for tracking ~~
    def hover_ref_func(self, num): #Returns Constant Hover Reference Trajectories At A Few Different Positions for Testing ([x,y,z,yaw])
        hover_dict = {
            # 1: np.array([[0.0, 0.0, -0.5, 0.0]]).T,
            1: np.array([[0.0, 0.0, -0.5,     0.0, 0.0, 0.0,   0.0, 0.0, 0.0]]),
            2: np.array([[0.0, 1.5, -1.5,     0.0, 0.0, 0.0,   0.0, 0.0, 0.0]]),
            3: np.array([[1.5, 0.0, -1.5,     0.0, 0.0, 0.0,   0.0, 0.0, 0.0]]),
            4: np.array([[1.5, 1.5, -1.5,     0.0, 0.0, 0.0,   0.0, 0.0, 0.0]]),
            5: np.array([[0.0, 0.0, -10.0,    0.0, 0.0, 0.0,   0.0, 0.0, 0.0]]),
            6: np.array([[1.0, 1.0, -4.0,     0.0, 0.0, 0.0,   0.0, 0.0, 0.0]]),
            7: np.array([[3.0, 4.0, -5.0,     0.0, 0.0, 0.0,   0.0, 0.0, 0.0]]),
            8: np.array([[1.0, 1.0, -3.0,     0.0, 0.0, 0.0,   0.0, 0.0, 0.0]]),
        }
        if num > len(hover_dict) or num < 1:
            print(f"hover1- #{num} not found")
            exit(0)
            # return np.array([[0.0, 0.0, 0.0, self.yaw]]).T

        if not self.sim:
            if num > 4:
                print("hover modes 5+ not available for hardware")
                exit(0)
                # return np.array([[0.0, 0.0, 0.0, self.yaw]]).T
            
        print(f"hover1- #{num}")
        r = hover_dict.get(num)
        r_final = np.tile(r, (self.num_steps,1)) # np.tile(xr, (num_steps, 1))
        # print(f"{r_final=}")
        # print(f"{r_final.shape=}")
        return r_final

    def circle_vert_ref_func(self): #Returns Circle Reference Trajectory in Vertical Plane ([x,y,z,yaw])
        print("circle_vert_ref_func")

        N = self.num_steps  # Number of time steps
        start = self.timefromstart  # Start time
        stop = self.timefromstart + self.horizon  # End time
        # print(f"{N=}")
        # print(f"{start=}")
        # print(f"{stop=}")
        # Time array
        t = np.linspace(start, stop, N)
        w = 1  # Angular frequency for the circular motion

        # Trajectory components
        x = np.zeros(N)  # x is constant
        y = 0.4 * np.cos(w * t)  # y component of the circle
        z = -1 * (0.4 * np.sin(w * t) + 1.5)  # z component of the circle, shifted down by 1.5

        # Velocities and angles are zero
        vx = np.zeros(N)
        vy = np.zeros(N)
        vz = np.zeros(N)
        roll = np.zeros(N)
        pitch = np.zeros(N)
        yaw = np.zeros(N)

        # Combine all components into a (N, 9) array
        r_final = np.column_stack((x, y, z, vx, vy, vz, roll, pitch, yaw))
        return r_final
    
    def circle_horz_ref_func(self):
        print("circle_horz_ref_func")

        N = self.num_steps
        start = self.timefromstart
        stop = self.timefromstart + self.horizon
        t = np.linspace(start, stop, N)
        w = 1  # Angular frequency

        x = 0.8 * np.cos(w * t)
        y = 0.8 * np.sin(w * t)
        z = np.full(N, -1.25)  # Constant z
        vx, vy, vz = np.zeros(N), np.zeros(N), np.zeros(N)
        roll, pitch, yaw = np.zeros(N), np.zeros(N), np.zeros(N)

        r_final = np.column_stack((x, y, z, vx, vy, vz, roll, pitch, yaw))
        return r_final
    
    def fig8_horz_ref_func(self):
        print("fig8_horz_ref_func")

        N = self.num_steps
        start = self.timefromstart
        stop = self.timefromstart + self.horizon
        t = np.linspace(start, stop, N)

        x = 0.35 * np.sin(2 * t)
        y = 0.35 * np.sin(t)
        z = np.full(N, -1.25)  # Constant z
        vx, vy, vz = np.zeros(N), np.zeros(N), np.zeros(N)
        roll, pitch, yaw = np.zeros(N), np.zeros(N), np.zeros(N)

        r_final = np.column_stack((x, y, z, vx, vy, vz, roll, pitch, yaw))
        return r_final

    def fig8_vert_ref_func_short(self):
        print("fig8_vert_ref_func_short")

        N = self.num_steps
        start = self.timefromstart
        stop = self.timefromstart + self.horizon
        t = np.linspace(start, stop, N)

        x = np.zeros(N)
        y = 0.4 * np.sin(t)
        z = -1 * (0.4 * np.sin(2 * t) + 1.25)
        vx, vy, vz = np.zeros(N), np.zeros(N), np.zeros(N)
        roll, pitch, yaw = np.zeros(N), np.zeros(N), np.zeros(N)

        r_final = np.column_stack((x, y, z, vx, vy, vz, roll, pitch, yaw))
        return r_final

    def fig8_vert_ref_func_tall(self):
        print("fig8_vert_ref_func_tall")

        N = self.num_steps
        start = self.timefromstart
        stop = self.timefromstart + self.horizon
        t = np.linspace(start, stop, N)

        x = np.zeros(N)
        y = 0.4 * np.sin(2 * t)
        z = -1 * (0.4 * np.sin(t) + 1.25)
        vx, vy, vz = np.zeros(N), np.zeros(N), np.zeros(N)
        roll, pitch, yaw = np.zeros(N), np.zeros(N), np.zeros(N)

        r_final = np.column_stack((x, y, z, vx, vy, vz, roll, pitch, yaw))
        return r_final


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