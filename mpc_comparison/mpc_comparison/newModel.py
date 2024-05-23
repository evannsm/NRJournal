from casadi import SX, vertcat, horzcat, diag, inv_minor, cross, sqrt, cos, sin
import numpy as np
class Quadrotor:

    def __init__(self, sim: bool):
        self.sim = sim
        self.g = 9.806 #gravity
        self.m = 1.535 if sim else 1.69 #mass

    def dynamics(self):
        #states
        px = SX.sym('px')
        py = SX.sym('py')
        pz = SX.sym('pz')
        vx = SX.sym('vx')
        vy = SX.sym('vy')
        vz = SX.sym('vz')
        roll = SX.sym('roll')
        pitch = SX.sym('pitch')
        yaw = SX.sym('yaw')

        #state vector
        x = vertcat(px, py, pz, vx, vy, vz, roll, pitch, yaw)
        
        #control inputs
        thrust = SX.sym('thrust')
        rolldot = SX.sym('rolldot')
        pitchdot = SX.sym('pitchdot')
        yawdot = SX.sym('yawdot')

        #control vector
        u = vertcat(thrust, rolldot, pitchdot, yawdot)

        #dynamics
        # define trig functions
        sr = sin(roll)
        sy = sin(yaw)
        sp = sin(pitch)
        cr = cos(roll)
        cp = cos(pitch)
        cy = cos(yaw)

        # # Define rotation matrix from quadrotor body to inertial reference frames
        # Rotm = vertcat(
        #     horzcat(cp*cy, sr*sp*cy - cr*sy, cr*sp*cy + sr*sy),
        #     horzcat(cp*sy, sr*sp*sy + cr*cy, cr*sp*sy - sr*cy),
        #     horzcat(-sp, sr*cp, cr*cp)
        # )
        # # velocity dynamics
        # f_vec = vertcat(0., 0., thrust)
        # vdot = vertcat(0.,0.,-self.g) + Rotm @ f_vec / self.mass

        #define dynamics
        pxdot = vx
        pydot = vy
        pzdot = vz
        vxdot = -(thrust/self.m) * (sr*sy + cr*cy*sp); # note that this is the explicit verion of the above vdot equation (vdot[0])
        vydot = -(thrust/self.m) * (cr*sy*sp - cy*sr); # note that this is the explicit verion of the above vdot equation (vdot[1])
        vzdot = self.g - (thrust/self.m) * (cr*cp); # note that this is the explicit verion of the above vdot equation (vdot[2])
        rolldot = rolldot
        pitchdot = pitchdot
        yawdot = yawdot

        # vector containing the quadrotor's explicit dynamics with 9 states and 4 control inputs of body rates and thrust
        f_expl = vertcat(pxdot, pydot, pzdot, vxdot, vydot, vzdot, rolldot, pitchdot, yawdot)

        return (f_expl, x, u)

