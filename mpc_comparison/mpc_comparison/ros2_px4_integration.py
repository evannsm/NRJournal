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
###############################################################################################################################################
        # Figure out if in simulation or hardware mode to set important variables to the appropriate values
        self.sim = bool(int(input("Are you using the simulator? Write 1 for Sim and 0 for Hardware: ")))
        print(f"{'SIMULATION' if self.sim else 'HARDWARE'}")

###############################################################################################################################################
        # Initialize variables:        
        self.offboard_setpoint_counter = 0 #helps us count 10 cycles of sending offboard heartbeat before switching to offboard mode and arming
        self.vehicle_status = VehicleStatus() #vehicle status variable to make sure we're in offboard mode before sending setpoints

        self.T0 = time.time() # initial time of program
        self.timefromstart = time.time() - self.T0 # time from start of program initialized and updated later to keep track of current time in program

        self.g = 9.806 #gravity        
        self.time_before_land = 30.0

        # The following 3 variables are used to convert between force and throttle commands (iris gazebo simulation)
        self.motor_constant_ = 0.00000584 #iris gazebo simulation motor constant
        self.motor_velocity_armed_ = 100 #iris gazebo motor velocity when armed
        self.motor_input_scaling_ = 1000.0 #iris gazebo simulation motor input scaling

###############################################################################################################################################
        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create Publishers
        # Publishers for Setting to Offboard Mode and Arming/Diasarming/Landing/etc
        self.offboard_control_mode_publisher = self.create_publisher( #publishes offboard control heartbeat
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.vehicle_command_publisher = self.create_publisher( #publishes vehicle commands (arm, offboard, disarm, etc)
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        
        # Publishers for Sending Setpoints in Offboard Mode: 1) Body Rates and Thrust, 2) Position and Yaw 
        self.rates_setpoint_publisher = self.create_publisher( #publishes body rates and thrust setpoint
            VehicleRatesSetpoint, '/fmu/in/vehicle_rates_setpoint', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher( #publishes trajectory setpoint
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        
        # Publisher for Logging States, Inputs, and Reference Trajectories for Data Analysis
        self.state_input_ref_log_publisher_ = self.create_publisher( #publishes log of states and input
            Float64MultiArray, '/state_input_ref_log', 10)
        self.state_input_ref_log_msg = Float64MultiArray() #creates message for log of states and input
        

        # Create subscribers
        self.vehicle_odometry_subscriber = self.create_subscription( #subscribes to odometry data (position, velocity, attitude)
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.vehicle_odometry_callback, qos_profile)
        
        self.vehicle_status_subscriber = self.create_subscription( #subscribes to vehicle status (arm, offboard, disarm, etc)
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)
        
        self.offboard_mode_rc_switch_on = True if self.sim else False #Offboard mode starts on if in Sim, turn off and wait for RC if in hardware
        if not self.sim:
            self.rc_channels_subscriber = self.create_subscription( #subscribes to rc_channels topic for software "killswitch" to make sure we'd like position vs offboard vs land mode
                RcChannels, '/fmu/out/rc_channels', self.rc_channel_callback, qos_profile
            )

###############################################################################################################################################
        # Define the MPC controller
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
                Q_mat = np.diag([50., 50., 50.,   0., 0., 0.,   0., 0., 10.]) # [x, y, z, vx, vy, vz, roll, pitch, yaw]
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
        self.mpc_solver = MPC(False, quad, self.horizon, self.num_steps)

        # x0 = np.zeros(9)
        # print(f"{x0.shape=}")
        # print(f"{x0=}")
        # xref = self.hover_ref_func(1)
        # print(f"{xref.shape=}")
        # status, x_mpc, u_mpc = self.mpc_solver.solve_mpc_control(x0, xref)
        # print(u_mpc)

        # reffunc = self.fig8_vert_ref_func_tall()
        # print(f"{reffunc.shape=}")
        # status, x_mpc, u_mpc = self.mpc_solver.solve_mpc_control(x0, reffunc)
        # print(u_mpc)
        # exit(0)

###############################################################################################################################################
        #Create Function @ {1/self.offboard_timer_period}Hz (in my case should be 10Hz/0.1 period) to Publish Offboard Control Heartbeat Signal
        self.offboard_timer_period = 0.1
        self.timer = self.create_timer(self.offboard_timer_period, self.offboard_mode_timer_callback)
        # exit(0)

        # Create Function at {1/self.controller_timer_period}Hz (in my case should be 100Hz/0.01 period) to Send Control Input
        self.controller_timer_period = 0.01
        self.timer = self.create_timer(self.controller_timer_period, self.controller_timer_callback)
###############################################################################################################################################
# ~~ The remaining group of functions are outside of the contructor and all help manage vehicle operations ~~

# ~~ The following first 4 functions all call publish_vehicle_command to arm/disarm/land/ and switch to offboard mode ~~
# ~~ The 5th function publishes the vehicle command commanded by the first 4 functions ~~
# ~~ The 6th function checks our vehicle status to see if it has changed from arm/disarm/land/offboard so we can make appropriate control decisions  ~~
    def arm(self): #1. Sends arm command to vehicle via publish_vehicle_command function
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarm(self): #2. Sends disarm command to vehicle via publish_vehicle_command function
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self): #3. Sends offboard command to vehicle via publish_vehicle_command function
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def land(self): #4. Sends land command to vehicle via publish_vehicle_command function
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")

    def publish_vehicle_command(self, command, **params) -> None: #5. Called by the above 4 functions to send parameter/mode commands to the vehicle
        """Publish a vehicle command."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def vehicle_status_callback(self, vehicle_status): #6. This function helps us check if we're in offboard mode before we start sending setpoints
        """Callback function for vehicle_status topic subscriber."""
        # print('vehicle status callback')
        self.vehicle_status = vehicle_status

# ~~ For hardware only: monitors the rc radio signal for the appropriate switch position on Channel5 to switch to offboard mode ~~
    def rc_channel_callback(self, rc_channels):
        """Callback function for RC Channels to create a software 'killswitch' depending on our flight mode channel (position vs offboard vs land mode)"""
        # print('rc channel callback')
        mode_channel = 5
        flight_mode = rc_channels.channels[mode_channel-1] # +1 is offboard everything else is not offboard
        self.offboard_mode_rc_switch_on = True if flight_mode >= 0.75 else False

# ~~ The following 2 functions are used to publish offboard control heartbeat signals ~~
    def publish_offboard_control_heartbeat_signal2(self): #1)Offboard Signal2 for Returning to Origin with Position Control
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_offboard_control_heartbeat_signal1(self): #2)Offboard Signal1 for Body Rate Control
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

###############################################################################################################################################
# ~~ The remaining functions are all intimately related to the Control Algorithm ~~
    def get_throttle_command_from_force(self, collective_thrust): #Converts force to throttle command
        collective_thrust = -collective_thrust
        print(f"Conv2Throttle: collective_thrust: {collective_thrust}")
        if self.sim:
            motor_speed = sqrt(collective_thrust / (4.0 * self.motor_constant_))
            throttle_command = (motor_speed - self.motor_velocity_armed_) / self.motor_input_scaling_
            return -throttle_command
        if not self.sim:
            # print('using hardware throttle from force conversion function')
            a = 0.00705385408507030
            b = 0.0807474474438391
            c = 0.0252575818743285

            # equation form is a*x + b*sqrt(x) + c = y
            throttle_command = a*collective_thrust + b*sqrt(collective_thrust) + c
            return -throttle_command

    def get_force_from_throttle_command(self, throttle_command): #Converts throttle command to force
        throttle_command = -throttle_command
        print(f"Conv2Force: throttle_command: {throttle_command}")
        if self.sim:
            motor_speed = (throttle_command * self.motor_input_scaling_) + self.motor_velocity_armed_
            collective_thrust = 4.0 * self.motor_constant_ * motor_speed ** 2
            return -collective_thrust
        
        if not self.sim:
            # print('using hardware force from throttle conversion function')
            a = 19.2463167420814
            b = 41.8467162352942
            c = -7.19353022443441

            # equation form is a*x^2 + b*x + c = y
            collective_thrust = a*throttle_command**2 + b*throttle_command + c
            return -collective_thrust
        
    def angle_wrapper(self, angle): # Fixes yaw issues in odometry_callback below
        angle += m.pi
        angle = angle % (2 * m.pi)  # Ensure the angle is between 0 and 2π
        if angle > m.pi:            # If angle is in (π, 2π], subtract 2π to get it in (-π, 0]
            angle -= 2 * m.pi
        if angle < -m.pi:           # If angle is in (-2π, -π), add 2π to get it in (0, π]
            angle += 2 * m.pi
        return -angle
    
    def vehicle_odometry_callback(self, msg): # Odometry Callback Function Yields Position, Velocity, and Attitude Data
        """Callback function for vehicle_odometry topic subscriber."""
        # print("AT ODOM CALLBACK")
        (self.yaw, self.pitch, self.roll) = euler_from_quaternion(msg.q)
        # print(f"old_yaw: {self.yaw}")
        self.yaw = self.angle_wrapper(self.yaw)
        self.pitch = -1 * self.pitch #pitch is negative of the value in gazebo bc of frame difference

        self.p = msg.angular_velocity[0]
        self.q = msg.angular_velocity[1]
        self.r = msg.angular_velocity[2]

        self.x = msg.position[0]
        self.y = msg.position[1]
        self.z = 1 * msg.position[2] # z is negative of the value in gazebo bc of frame difference

        self.vx = msg.velocity[0]
        self.vy = msg.velocity[1]
        self.vz = 1 * msg.velocity[2] # vz is negative of the value in gazebo bc of frame difference

        # print(f"Roll: {self.roll}")
        # print(f"Pitch: {self.pitch}")
        # print(f"Yaw: {self.yaw}")
        
        self.stateVector = np.array([[self.x, self.y, self.z, self.vx, self.vy, self.vz, self.roll, self.pitch, self.yaw]]).T 
        self.nr_state = np.array([[self.x, self.y, self.z, self.yaw]]).T
        self.odom_rates = np.array([[self.p, self.q, self.r]]).T

    def publish_rates_setpoint(self, thrust: float, roll: float, pitch: float, yaw: float): #Publishes Body Rate and Thrust Setpoints
        """Publish the trajectory setpoint."""
        msg = VehicleRatesSetpoint()
        msg.roll = float(roll)
        msg.pitch = float(pitch)
        msg.yaw = float(yaw)
        msg.thrust_body[0] = 0.0
        msg.thrust_body[1] = 0.0
        msg.thrust_body[2] = 1* float(thrust)

        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.rates_setpoint_publisher.publish(msg)
        
        # print("in publish rates setpoint")
        # self.get_logger().info(f"Publishing rates setpoints [r,p,y]: {[roll, pitch, yaw]}")
        print(f"Publishing rates setpoints [thrust, r,p,y]: {[thrust, roll, pitch, yaw]}")

    def publish_position_setpoint(self, x: float, y: float, z: float): #Publishes Position and Yaw Setpoints
        """Publish the trajectory setpoint."""
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.yaw = 0.0  # (90 degree)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)
        self.get_logger().info(f"Publishing position setpoints {[x, y, z]}")


###############################################################################################################################################
# ~~ The following 2 functions are the main functions that run at 10Hz and 100Hz ~~
    def offboard_mode_timer_callback(self) -> None: # ~~Runs at 10Hz and Sets Vehicle to Offboard Mode  ~~
        """Offboard Callback Function for The 10Hz Timer."""
        # print("In offboard timer callback")

        if self.offboard_mode_rc_switch_on: #integration of RC 'killswitch' for offboard deciding whether to send heartbeat signal, engage offboard, and arm
            if self.timefromstart <= self.time_before_land:
                self.publish_offboard_control_heartbeat_signal1()
            elif self.timefromstart > self.time_before_land:
                self.publish_offboard_control_heartbeat_signal2()


            if self.offboard_setpoint_counter == 10:
                self.engage_offboard_mode()
                self.arm()
            if self.offboard_setpoint_counter < 11:
                self.offboard_setpoint_counter += 1

        else:
            print("Offboard Callback: RC Flight Mode Channel 5 Switch Not Set to Offboard (-1: position, 0: offboard, 1: land) ")
            self.offboard_setpoint_counter = 0

    def controller_timer_callback(self) -> None: # ~~This is the main function that runs at 100Hz and Administrates Calls to Every Other Function ~~
        print("Controller Callback")
        if self.offboard_mode_rc_switch_on: #integration of RC 'killswitch' for offboard deciding whether to send heartbeat signal, engage offboard, and arm
            self.timefromstart = time.time()-self.T0 #update curent time from start of program for reference trajectories and for switching between NR and landing mode
            

            print(f"--------------------------------------")
            # print(self.vehicle_status.nav_state)
            if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                # print(f"Controller callback- timefromstart: {self.timefromstart}")
                print("IN OFFBOARD MODE")

                if self.timefromstart <= self.time_before_land: # our controller for first {self.time_before_land} seconds
                    print(f"Entering Control Loop for next: {self.time_before_land-self.timefromstart} seconds")
                    self.controller()

                elif self.timefromstart > self.time_before_land: #then land at origin and disarm
                    print("BACK TO SPAWN")
                    self.publish_position_setpoint(0.0, 0.0, -0.3)
                    print(f"self.x: {self.x}, self.y: {self.y}, self.z: {self.z}")
                    if abs(self.x) < 0.1 and abs(self.y) < 0.1 and abs(self.z) <= 0.50:
                        print("Switching to Land Mode")
                        self.land()

            if self.timefromstart > self.time_before_land:
                if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LAND:
                        print("IN LAND MODE")
                        if abs(self.z) <= .18:
                            print("Disarming and Exiting Program")
                            self.disarm()
                            # # Specify the file name
                            # csv_file = 'example.csv'
                            # print(self.nr_time_el)
                            # # Open the .csv file for writing
                            # with open(csv_file, mode='w', newline='') as file:
                            #     writer = csv.writer(file)  # Create a CSV writer

                            #     # Write the data to the CSV file
                            #     for row in self.nr_time_el:
                            #         writer.writerow(row)

                            # print(f'Data has been written to {csv_file}')
                            exit(0)
            print(f"--------------------------------------")
            print("\n\n")
        else:
            print("Controller Callback: Channel 11 Switch Not Set to Offboard")


###############################################################################################################################################
# ~~ From here down are the functions that actually calculate the control input ~~
    def controller(self): # Runs Algorithm Structure
        print(f"NR States: {self.nr_state}") #prints current state

#~~~~~~~~~~~~~~~ Get Reference Trajectory ~~~~~~~~~~~~~~~
        # reffunc = self.circle_vert_ref_func()
        # reffunc = self.circle_horz_ref_func()
        reffunc = self.fig8_horz_ref_func()
        # reffunc = self.fig8_vert_ref_func_short()
        # reffunc = self.fig8_vert_ref_func_tall()
        # reffunc = self.hover_ref_func(1)
        # reffunc = self.hover_up_and_down()
        print(f"reffunc: \n{reffunc[1,:]}")


        # Calculate the MPC control input
        new_u = self.get_new_control_input(reffunc)
        new_force = -new_u[0]       
        print(f"new_force: {new_force}")

        new_throttle = self.get_throttle_command_from_force(new_force)
        new_roll_rate = new_u[1]
        new_pitch_rate = new_u[2]
        new_yaw_rate = new_u[3]



        # Build the final input vector to save as self.u0 and publish to the vehicle via publish_rates_setpoint:
        final = [new_throttle, new_roll_rate, new_pitch_rate, new_yaw_rate]

        current_input_save = np.array(final).reshape(-1, 1)
        print(f"newInput: \n{current_input_save}")
        self.u0 = current_input_save

        # Publish the final input to the vehicle
        self.publish_rates_setpoint(final[0], final[1], final[2], final[3])

        # Log the states, inputs, and reference trajectories for data analysis
        self.state_input_ref_log_msg.data = [float(self.x), float(self.y), float(self.z), float(self.yaw), float(final[0]), float(final[1]), float(final[2]), float(final[3]), float(reffunc[0][0]), float(reffunc[1][0]), float(reffunc[2][0]), float(reffunc[3][0])]
        self.state_input_ref_log_publisher_.publish(self.state_input_ref_log_msg)

    def get_new_control_input(self, reffunc):
        x0 = self.stateVector.flatten()
        # print(f"{x0=}")
        # print(f"{x0.shape=}")
        # print(f"{reffunc.shape=}")
        # exit(0)
        status, x_mpc, u_mpc = self.mpc_solver.solve_mpc_control(x0, reffunc)
        # print(f"status: {status}")
        # print(f"x_mpc: {x_mpc}")
        print(f"u_mpc: {u_mpc}")
        return u_mpc



###############################################################################################################################################
# ~~ The following functions are reference trajectories for tracking ~~
    
    def hover_up_and_down(self): #Returns Constant Hover Reference Trajectories At A Few Different Positions for Testing ([x,y,z,yaw])
        if self.timefromstart <= self.time_before_land/2:
            r_final = self.hover_ref_func(5)
        elif self.timefromstart > self.time_before_land/2:
            r_final = self.hover_ref_func(1)

        return r_final
    
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



###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
# ~~ Entry point of the code -> Initializes the node and spins it ~~
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