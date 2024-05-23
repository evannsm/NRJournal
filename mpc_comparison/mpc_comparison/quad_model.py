from acados_template import AcadosModel
from casadi import SX, vertcat, cos, sin

# Define the quadrotor model and MPC controller
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