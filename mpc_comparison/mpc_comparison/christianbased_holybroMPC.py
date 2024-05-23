import torch
from torch import nn
import torch.nn.functional as F
 # OG SHIT!!!!!!

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, VehicleRatesSetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus, VehicleOdometry, TrajectorySetpoint, RcChannels
from std_msgs.msg import Float64, Float64MultiArray, String

from .newModel import Quadrotor
from .newGenMPC import QuadrotorMPC2

import sympy as smp
from scipy.integrate import quad
from scipy.linalg import expm
from sympy import * 
import numpy as np

from tf_transformations import euler_from_quaternion
import time

from math import sqrt
import math as m

import csv
import ctypes

import time

from casadi import *
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import numpy as np
from casadi import SX, vertcat, horzcat, diag, inv_minor, cross, sqrt,  cos, sin, norm_2, tanh
from scipy.linalg import block_diag
from threading import Thread
# from time import sleep, time

from pathlib import Path
import importlib
import sys
import os
from collections import deque


class OffboardControl(Node):
    """Node for controlling a vehicle in offboard mode."""
    def __init__(self) -> None:
        super().__init__('offboard_control_takeoff_and_land')
###############################################################################################################################################
        # Figure out if in simulation or hardware mode to set important variables to the appropriate values
        self.sim = bool(int(input("Are you using the simulator? Write 1 for Sim and 0 for Hardware: ")))
        print(f"{'SIMULATION' if self.sim else 'HARDWARE'}")

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
    
        self.offboard_mode_rc_switch_on = True if self.sim else False #Offboard mode set to on if in Sim; turn off and wait for RC switch if in hardware
        if not self.sim:
            self.rc_channels_subscriber = self.create_subscription( #subscribes to rc_channels topic for software "killswitch" to make sure we'd like position vs offboard vs land mode
                RcChannels, '/fmu/out/rc_channels', self.rc_channel_callback, qos_profile
            )

###############################################################################################################################################
        # Initialize variables:
        self.offboard_setpoint_counter = 0 #helps us count 10 cycles of sending offboard heartbeat before switching to offboard mode and arming
        self.vehicle_status = VehicleStatus() #vehicle status variable to make sure we're in offboard mode before sending setpoints

        self.T0 = time.time() # initial time of program
        self.timefromstart = time.time() - self.T0 # time from start of program initialized and updated later to keep track of current time in program

        # The following 3 variables are used to convert between force and throttle commands (iris gazebo simulation)
        self.motor_constant_ = 0.00000584 #iris gazebo simulation motor constant
        self.motor_velocity_armed_ = 100 #iris gazebo motor velocity when armed
        self.motor_input_scaling_ = 1000.0 #iris gazebo simulation motor input scaling

        self.control_queue = None # queue of mpc_N/num_steps control inputs to be published to the vehicle
        self.start_mpc = False # boolean to start mpc calculations
###############################################################################################################################################
        # Generate MPC controller
        quad = Quadrotor(sim=self.sim)
        generate_c_code = False

        # MPC Parameters
        self.mpc_tf = 3.0
        self.mpc_N = 100
        control_update_rate = self.mpc_N
        self.mpc_solver = QuadrotorMPC2(generate_c_code, quad, self.mpc_tf, self.mpc_N)


        self.time_before_land = 30.0
        print(f"time_before_land: {self.time_before_land}")

###############################################################################################################################################
        #Create Function @ {1/self.offboard_timer_period}Hz (in my case should be 10Hz/0.1 period) to Publish Offboard Control Heartbeat Signal
        self.offboard_timer_period = 0.1
        self.timer = self.create_timer(self.offboard_timer_period, self.offboard_mode_timer_callback)
        # exit(0)

        # Create Function at {1/self.controller_timer_period}Hz (in this case should be 100Hz/0.01 period) to Send Control Input
        self.controller_timer_period = 1/control_update_rate
        self.timer = self.create_timer(self.controller_timer_period, self.controller_timer_callback)

        # Create Function at {1/self.mpc_solve_time}Hz to Calculate MPC Input
        self.mpc_timer_period = 1/10
        self.timer = self.create_timer(self.mpc_timer_period, self.mpc_timer_callback)

###############################################################################################################################################
###############################################################################################################################################
        
    def rc_channel_callback(self, rc_channels): # Hardware Only: Allows us to use an RC channel as a 'killswitch' for offboard mode for safety
        """Callback function for RC Channels to create a software 'killswitch' depending on our flight mode channel (position vs offboard vs land mode)"""
        # print('rc channel callback')
        mode_channel = 5
        flight_mode = rc_channels.channels[mode_channel-1] # +1 is offboard everything else is not offboard
        self.offboard_mode_rc_switch_on = True if flight_mode >= 0.75 else False
###############################################################################################################################################

    # The following 4 functions all call publish_vehicle_command to arm/disarm/land/ and switch to offboard mode
    # The 5th function publishes the vehicle command commanded by the first 4 functions
    # The 6th function checks our vehicle status to see if it has changed from arm/disarm/land/offboard so we can make appropriate control decisions
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

    def vehicle_status_callback(self, vehicle_status): #6. This function helps us check our vehicle status to see if we're in offboard mode before we start sending setpoints
        """Callback function for vehicle_status topic subscriber."""
        # print('vehicle status callback')
        self.vehicle_status = vehicle_status
    

###############################################################################################################################################
    # The following 2 functions are used to publish offboard control heartbeat signals
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

    # The following 2 functions are used to convert between force and throttle commands
    def get_throttle_command_from_force(self, collective_thrust): #1. Converts force to throttle command
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

    def get_force_from_throttle_command(self, throttle_command): #2. Converts throttle command to force
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
        
    def angle_wrapper(self, angle): #Wraps angle between -π and π for odometry callback below
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
# ~~ The following 3 functions are the main functions: 1. Keeps Offboard Mode On Via Constant Signal; 2. Publishes Control Input; 3. Runs MPC Calculations  ~~
    def offboard_mode_timer_callback(self) -> None: # ~~ #1. Runs at 10Hz and Sets Vehicle to Offboard Mode  ~~
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


    def controller_timer_callback(self) -> None: # ~~ #2. This is the main function that publishes offboard commands and switches to landing mode when needed ~~
        # print("Controller Callback")
        if self.offboard_mode_rc_switch_on: #integration of RC 'killswitch' for offboard deciding whether to send heartbeat signal, engage offboard, and arm
            self.timefromstart = time.time()-self.T0 #update curent time from start of program for reference trajectories and for switching between NR and landing mode

            print("\n\n")
            print(f"##########################################")
            # print(self.vehicle_status.nav_state)
            if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                # print(f"Controller callback- timefromstart: {self.timefromstart}")
                print("IN OFFBOARD MODE")

                if self.timefromstart <= self.time_before_land: # our controller for first {self.time_before_land} seconds
                    print(f"Entering Control Loop for next: {self.time_before_land-self.timefromstart} seconds")
                    self.pub_ctrl()

                elif self.timefromstart > self.time_before_land: #then land at origin and disarm
                    print("BACK TO SPAWN")
                    self.start_mpc = False
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
                            exit(0)
            print(f"##########################################")
            # print("\n\n")
        else:
            print("Controller Callback: Channel 11 Switch Not Set to Offboard")

    def mpc_timer_callback(self) -> None: # ~~ #3. This is the main function that runs the MPC calculations and sends them to the deque for publishing in the controller_timer_callback ~~
        if self.start_mpc:
            print("MPC Callback")
            print(f"NR States: {self.nr_state}") #prints current state

    #~~~~~~~~~~~~~~~ Calculate reference trajectory ~~~~~~~~~~~~~~~
            reffunc = self.circle_vert_ref_func
            self.trajectory_function = reffunc
            self.reffunc = reffunc(self.timefromstart)

            t = self.timefromstart #update curent time from start of program for reference trajectories and for switching between NR and landing mode
            x0 = self.stateVector.flatten()
            trajectory = self.navigator(t)
            # print(f"{trajectory.shape = }")
            yref = trajectory[:,:-1]
            yref_e = trajectory[:,-1]

            status, x_mpc, u_mpc = self.mpc_solver.solve_mpc_control(x0, yref, yref_e)
            print(f"{u_mpc[0]}")
            # exit(0)
            
            self.control_queue = deque(u_mpc)

            # print(f"{self.control_queue = }")
            # exit(0)


# ~~ Called by "controller_timer_callback": Takes the calculated MPC control inputs, converts them into thrusts, and publishes them to the quadrotor ~~
    def pub_ctrl(self):
        if not self.start_mpc:
            self.start_mpc = True

        if self.control_queue is not None:
            new_u = self.control_queue.popleft()
            print(f"{len/home/factslabegmc/newtonraphson_final_ws/src/newton_raphson/newton_raphson/generate_data.ipynb(self.control_queue) = }")
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
            # exit(0)
            # Publish the final input to the vehicle
            self.publish_rates_setpoint(final[0], final[1], final[2], final[3])

            # Log the states, inputs, and reference trajectories for data analysis
            self.state_input_ref_log_msg.data = [float(self.x), float(self.y), float(self.z), float(self.yaw), float(final[0]), float(final[1]), float(final[2]), float(final[3]), float(self.reffunc[0]), float(self.reffunc[1]), float(self.reffunc[2]), float(self.reffunc[3])]
            self.state_input_ref_log_publisher_.publish(self.state_input_ref_log_msg)      
        else:
            print("#########################")
            print("WE'RE OUT OF INPUTS!!!!!")
            print("#########################")

# ~~ Called by "mpc_timer_callback": Gets MPC inputs and adds them to deque ~~
    def navigator(self, t):
        print(f"{self.trajectory_function(0) = }")
        t_mpc_array = np.linspace(t, self.mpc_tf + t, self.mpc_N+1)
        yref = np.array([self.trajectory_function(t_mpc) for t_mpc in t_mpc_array]).T

        return yref




###############################################################################################################################################
    
    def circle_vert_ref_func(self, t): #Returns Circle Reference Trajectory in Vertical Plane ([x,y,z,yaw])
        # print("circle_vert_ref_func")
        w=1;

        x = 0.0
        y = .4 * np.cos(w*t)
        z = -1*(.4*np.sin(w*t)+1.5)
        vx = 0.0
        vy = 0.0
        vz = 0.0
        roll = 0.0
        pitch = 0.0
        yaw = 0.0

        r_final = np.array([x, y, z, vx, vy, vz, roll, pitch, yaw])
        # r_final = np.tile(r, (1, self.num_steps))
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