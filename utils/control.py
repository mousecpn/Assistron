import roboticstoolbox as rtb
from spatialmath import SE3,base
import spatialmath as sm
import qpsolvers as qp
import numpy as np
import sys
try:
    from sdfsc.sdfsc.colchecker import colchecker
except:
    pass
import torch
import time

def arrived(panda, cur_joint, target_pose, threshold=0.001):
    n = 7
    panda.q = cur_joint
    Te = panda.fkine(cur_joint)

    Tep = target_pose.as_matrix()
    Tep = sm.SE3(Tep)


    # Transform from the end-effector to desired pose
    eTep = Te.inv() * Tep

    # Spatial error
    e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() ])) # * np.pi / 180

    # Calulate the required end-effector spatial velocity for the robot
    # to approach the goal. Gain is set to 1.0
    # v, arrived = rtb.p_servo(Te, Tep, 5, 0.001)
    v, arrived = rtb.p_servo(Te, Tep, 1, threshold)
    return e, arrived




class NEO_SS:
    def __init__(self,):
        self.checker=colchecker(use_selfcol=True)


    def velocity_based_control(self, panda, cur_joint, tar_vel, ang_vel, Lambda=0.1, Gain=1, pcl=None, onbase=True, params={'di':0.4,'ds':0.05,'xi':1.0}):
        # The pose of the Panda's end-effector
        n = 7
        panda.q = cur_joint
        Te = panda.fkine(cur_joint)
        if onbase:
            tar_vel = Te.inv().A[:3,:3] @ tar_vel # convert absolute to relative 
            ang_vel = Te.inv().A[:3,:3] @ ang_vel

        # Spatial error
        e = np.sum(np.abs(np.r_[tar_vel, ang_vel * np.pi /180]))

        # Calulate the required end-effector spatial velocity for the robot
        # to approach the goal. Gain is set to 1.0
        v = np.r_[tar_vel, ang_vel] * Gain

        # Gain term (lambda) for control minimisation
        Y = Lambda

        # v += rand(v.shape[0]) * v * 0.5 # * np.array([1,1,1,0,0,0])

        # Quadratic component of objective function
        Q = np.eye(n + 6)

        # Joint velocity component of Q
        Q[:n, :n] *= Y

        # Slack component of Q
        Q[n:, n:] = (1 / e) * np.eye(6)

        # The equality contraints
        Aeq = np.c_[panda.jacobe(panda.q), np.eye(6)]
        beq = v.reshape((6,))

        # The inequality constraints for joint limit avoidance
        Ain = np.zeros((n + 6, n + 6))
        bin = np.zeros(n + 6)

        # The minimum angle (in radians) in which the joint is allowed to approach
        # to its limit
        ps = 0.05

        # The influence angle (in radians) in which the velocity damper
        # becomes active
        pi = 0.9

        # Form the joint limit velocity damper
        Ain[:n, :n], bin[:n] = panda.joint_velocity_damper(ps, pi, n)

        c_Ain, c_bin, di = self.SDFSC_link_collision_damper(pcl, panda.q[:n], t_step=0.05, di=params['di'], ds=params['ds'], xi=params['xi'])
        if c_Ain is not None and c_bin is not None:
            c_Ain = np.c_[c_Ain, np.zeros((c_Ain.shape[0], 6))]
            Ain = np.r_[Ain, c_Ain]
            bin = np.r_[bin, c_bin]

        # Linear component of objective function: the manipulability Jacobian
        c = np.r_[-panda.jacobm(panda.q).reshape((n,)), np.zeros(6)]
        # c = c * 0.0

        # The lower and upper bounds on the joint velocity and slack variable
        lb = -np.r_[panda.qdlim[:n], 10 * np.ones(6)]
        ub = np.r_[panda.qdlim[:n], 10 * np.ones(6)]

        # Solve for the joint velocities dq
        qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver='daqp')

        # Apply the joint velocities to the Panda
        joint_velocity = qd[:n]

        return joint_velocity


    def calculate_velocity_ss(self, panda, cur_joint, target_pose, pcl=None, Lambda=0.1, Gain=1, threshold=0.001, params={'di':0.4,'ds':0.05,'xi':1.0}):
        # The pose of the Panda's end-effector
        n = 7
        panda.q = cur_joint
        t1 = time.time()
        Te = panda.fkine(cur_joint)

        try:
            Tep = target_pose.as_matrix()
        except:
            Tep = target_pose
        Tep = sm.SE3(Tep)
        t2 = time.time()

        # Transform from the end-effector to desired pose
        eTep = Te.inv() * Tep

        # Spatial error
        e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi / 180])) #  

        # Calulate the required end-effector spatial velocity for the robot
        # to approach the goal. Gain is set to 1.0
        # v, arrived = rtb.p_servo(Te, Tep, 5, 0.001)
        v, arrived = rtb.p_servo(Te, Tep, Gain, threshold)
        # print('v:', v)

        # Gain term (lambda) for control minimisation
        Y = Lambda

        # v += rand(v.shape[0]) * v * 0.5 # * np.array([1,1,1,0,0,0])

        # Quadratic component of objective function
        Q = np.eye(n + 6)

        # Joint velocity component of Q
        Q[:n, :n] *= Y

        # Slack component of Q
        Q[n:, n:] = (1 / e) * np.eye(6)

        # The equality contraints
        Aeq = np.c_[panda.jacobe(panda.q), np.eye(6)]
        beq = v.reshape((6,))

        # The inequality constraints for joint limit avoidance
        Ain = np.zeros((n + 6, n + 6))
        bin = np.zeros(n + 6)

        # The minimum angle (in radians) in which the joint is allowed to approach
        # to its limit
        ps = 0.05

        # The influence angle (in radians) in which the velocity damper
        # becomes active
        pi = 0.9

        # Form the joint limit velocity damper
        Ain[:n, :n], bin[:n] = panda.joint_velocity_damper(ps, pi, n)

        c_Ain, c_bin,di = self.SDFSC_link_collision_damper(pcl,panda.q[:n],t_step=0.05,di=params['di'],ds=params['ds'],xi=params['xi'])
        if c_Ain is not None and c_bin is not None:
            c_Ain = np.c_[c_Ain, np.zeros((c_Ain.shape[0], 6))]
            Ain = np.r_[Ain, c_Ain]
            bin = np.r_[bin, c_bin]

        # Linear component of objective function: the manipulability Jacobian
        c = np.r_[-panda.jacobm(panda.q).reshape((n,)), np.zeros(6)]
        c = c * 0.0

        # The lower and upper bounds on the joint velocity and slack variable
        lb = -np.r_[panda.qdlim[:n], 10 * np.ones(6)]
        ub = np.r_[panda.qdlim[:n], 10 * np.ones(6)]

        # Solve for the joint velocities dq
        qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver='daqp')

        # Apply the joint velocities to the Panda
        joint_velocity = qd[:n]

        return joint_velocity, arrived

    def SDFSC_link_collision_damper(
            self,
            all_points,
            q,
            t_step: float=0.05,
            di: float = 0.4,
            ds: float = 0.05,
            xi: float =1.0,
        ):
        n=len(q)
        if all_points is None:
            return None,None,None
        # print(all_points.shape)
        self.checker.get_points(all_points)
        col_func=self.checker.get_scores
        q_tensor=torch.tensor(q, dtype=torch.float32, device='cuda',requires_grad=True)

        d=col_func(q_tensor).cpu().item()
        jac = -torch.autograd.functional.jacobian(
                lambda x: col_func(x), 
                q_tensor,
                create_graph=False, strict=False,
                vectorize=True, strategy='reverse-mode'
            )*1e-1
        # print("cost time jac:",(jac_end-jac_start))
        c_Ain=np.zeros((n,7))
        c_bin=np.zeros(n,)
        c_Ain_=jac.cpu().numpy().reshape(1,n)
        c_bin_=np.array([xi*(d-ds)/(di-ds)])#-0.2*t_step
        for i in range(n):
            c_Ain[i,:(i+1)]=c_Ain_[0,:(i+1)]
            c_bin[i]=c_bin_[0]
        return c_Ain,c_bin,d


def simple_calculate_velocity(panda, cur_joint, target_pose, Gain=1, threshold=0.001):
    Te = panda.fkine(cur_joint)
    Tep = target_pose.as_matrix()
    v, arrived = rtb.p_servo(Te, Tep, Gain, threshold)
    if arrived:
        return np.zeros(7,), True
    jacobian = panda.jacobe(cur_joint)
    jacobian_pinv = np.linalg.pinv(jacobian)
    dq_task = jacobian_pinv @ v
    return dq_task.reshape(-1), False

def simple_velocity_based_control(panda, cur_joint, tar_vel, ang_vel, Gain=1, onbase=True, j_vel=None):
    n = 7
    panda.q = cur_joint
    Te = panda.fkine(cur_joint)
    if onbase:
        tar_vel = Te.inv().A[:3,:3] @ tar_vel # convert absolute to relative 
        ang_vel = Te.inv().A[:3,:3] @ ang_vel

    v = np.r_[tar_vel, ang_vel] * Gain

    jacobian = panda.jacobe(cur_joint)
    jacobian_pinv = np.linalg.pinv(jacobian)
    dq_task = jacobian_pinv @ v
    if j_vel is not None:
        k_null = 0.1
        I = np.eye(jacobian.shape[1])  # 7x7 单位矩阵
        null_space_projection = I - (jacobian_pinv @ jacobian)
        dq_0 = -j_vel.reshape(7,1) # 阻尼效果
            
        dq_null = null_space_projection @ dq_0
        
        # 4. 合并总速度
        dq_task = dq_task + k_null * dq_null.reshape(-1)

    return dq_task.reshape(-1)


def calculate_velocity(panda, cur_joint, target_pose, obstacles=None, Lambda=0.1, Gain=1, threshold=0.001, initvals=None):
    # The pose of the Panda's end-effector
    n = 7
    panda.q = cur_joint
    Te = panda.fkine(cur_joint)
    
    # t1 = time.time()
    try:
        Tep = target_pose.as_matrix()
    except:
        Tep = target_pose
    Tep = sm.SE3(Tep)


    # Transform from the end-effector to desired pose
    eTep = Te.inv() * Tep
    # t2 = time.time()
    # Spatial error
    e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy()])) #  * np.pi / 180

    # Calulate the required end-effector spatial velocity for the robot
    # to approach the goal. Gain is set to 1.0
    # v, arrived = rtb.p_servo(Te, Tep, 5, 0.001)
    v, arrived = rtb.p_servo(Te, Tep, Gain, threshold)
    # print('v:', v)
    
    # Gain term (lambda) for control minimisation
    Y = Lambda

    # v += rand(v.shape[0]) * v * 0.5 # * np.array([1,1,1,0,0,0])

    # Quadratic component of objective function
    Q = np.eye(n + 6)

    # Joint velocity component of Q
    Q[:n, :n] *= Y

    # Slack component of Q
    Q[n:, n:] = (1 / e) * np.eye(6)
    # t3 = time.time()
    # The equality contraints
    Aeq = np.c_[panda.jacobe(panda.q), np.eye(6)]
    # t4 = time.time()

    beq = v.reshape((6,))

    # The inequality constraints for joint limit avoidance
    Ain = np.zeros((n + 6, n + 6))
    bin = np.zeros(n + 6)

    # The minimum angle (in radians) in which the joint is allowed to approach
    # to its limit
    ps = 0.05

    # The influence angle (in radians) in which the velocity damper
    # becomes active
    pi = 0.9

    # Form the joint limit velocity damper
    Ain[:n, :n], bin[:n] = panda.joint_velocity_damper(ps, pi, n)
    
    if obstacles is not None:
        for collision in obstacles:
            # Form the velocity damper inequality contraint for each collision
            # object on the robot to the collision in the scene
            c_Ain, c_bin = panda.link_collision_damper(
                collision,
                panda.q[:n],
                0.3,
                0.05,
                1.0,
                start=panda.link_dict["panda_link1"],
                end=panda.link_dict["panda_hand"],
            )

            # If there are any parts of the robot within the influence distance
            # to the collision in the scene
            if c_Ain is not None and c_bin is not None:
                c_Ain = np.c_[c_Ain[:,:n], np.zeros((c_Ain.shape[0], 6))]

                # Stack the inequality constraints
                Ain = np.r_[Ain, c_Ain]
                bin = np.r_[bin, c_bin]

    # Linear component of objective function: the manipulability Jacobian
    c = np.r_[-panda.jacobm(panda.q).reshape((n,)), np.zeros(6)]

    # The lower and upper bounds on the joint velocity and slack variable
    lb = -np.r_[panda.qdlim[:n], 10 * np.ones(6)]
    ub = np.r_[panda.qdlim[:n], 10 * np.ones(6)]
    # Solve for the joint velocities dq
    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver='daqp', initvals=initvals)
    # t5 = time.time()
    # Apply the joint velocities to the Panda
    joint_velocity = qd[:n]

    # print(f"Timing: fkine+inv+error={t2-t1:.6f}s, p_servo={t3-t2:.6f}s, set_constraints={t4-t3:.6f}s, qp_solve={t5-t4:.6f}s")

    return joint_velocity, arrived



def velocity_based_control(panda, cur_joint, tar_vel, ang_vel, Lambda=0.1, Gain=1, obstacles=None, onbase=True, T_tcp_robotiq=None):
    # The pose of the Panda's end-effector
    # T_tcp_robotiq: optional 4x4 numpy array (T from TCP to robotiq_tcp).
    #   None  -> velocities are expressed in / resolved to fr3_hand_tcp frame (original behaviour)
    #   array -> velocities are expressed in robotiq_tcp frame; they are
    #            converted to TCP body frame via rigid-body velocity transform before IK
    n = 7
    panda.q = cur_joint
    Te = panda.fkine(cur_joint)

    if T_tcp_robotiq is not None:
        # --- robotiq_tcp mode ---
        R_tr = T_tcp_robotiq[:3, :3]   # rotation: TCP -> robotiq
        t_tr = T_tcp_robotiq[:3,  3]   # translation of robotiq origin in TCP frame

        if onbase:
            # World -> robotiq body frame
            Te_rob = Te.A @ T_tcp_robotiq
            R_w2r = Te_rob[:3, :3].T
            tar_vel = R_w2r @ tar_vel
            ang_vel = R_w2r @ ang_vel

        # Rigid-body velocity transform: robotiq body -> TCP body
        # omega_tcp = R_tr @ omega_rob
        # v_tcp     = R_tr @ v_rob  -  omega_tcp x t_tr   (lever-arm correction)
        omega_tcp = R_tr @ ang_vel
        tar_vel   = R_tr @ tar_vel - np.cross(omega_tcp, t_tr)
        ang_vel   = omega_tcp
    else:
        # --- TCP frame mode (original behaviour) ---
        if onbase:
            tar_vel = Te.inv().A[:3,:3] @ tar_vel # convert absolute to relative
            ang_vel = Te.inv().A[:3,:3] @ ang_vel

    # Spatial error
    e = np.sum(np.abs(np.r_[tar_vel, ang_vel* np.pi /180]))

    # Calulate the required end-effector spatial velocity for the robot
    # to approach the goal. Gain is set to 1.0
    v = np.r_[tar_vel, ang_vel] * Gain

    # Gain term (lambda) for control minimisation
    Y = Lambda

    # Quadratic component of objective function
    Q = np.eye(n + 6)

    # Joint velocity component of Q
    Q[:n, :n] *= Y

    # Slack component of Q
    Q[n:, n:] = (1 / (e + 1e-8)) * np.eye(6)

    # The equality contraints
    Aeq = np.c_[panda.jacobe(panda.q), np.eye(6)]
    beq = v.reshape((6,))

    # The inequality constraints for joint limit avoidance
    Ain = np.zeros((n + 6, n + 6))
    bin = np.zeros(n + 6)

    # The minimum angle (in radians) in which the joint is allowed to approach
    # to its limit
    ps = 0.05

    # The influence angle (in radians) in which the velocity damper
    # becomes active
    pi = 0.9

    # Form the joint limit velocity damper
    Ain[:n, :n], bin[:n] = panda.joint_velocity_damper(ps, pi, n)

    if obstacles is not None:
        for collision in obstacles:
            # Form the velocity damper inequality contraint for each collision
            # object on the robot to the collision in the scene
            c_Ain, c_bin = panda.link_collision_damper(
                collision,
                panda.q[:n],
                0.3,
                0.05,
                1.0,
                start=panda.link_dict["panda_link1"],
                end=panda.link_dict["panda_hand"],
            )
            

            # If there are any parts of the robot within the influence distance
            # to the collision in the scene
            if c_Ain is not None and c_bin is not None:
                c_Ain = np.c_[c_Ain[:,:n], np.zeros((c_Ain.shape[0], 6))]

                # Stack the inequality constraints
                Ain = np.r_[Ain, c_Ain]
                bin = np.r_[bin, c_bin]

    # Linear component of objective function: the manipulability Jacobian
    c = np.r_[-panda.jacobm(panda.q).reshape((n,)), np.zeros(6)]

    # The lower and upper bounds on the joint velocity and slack variable
    lb = -np.r_[panda.qdlim[:n], 10 * np.ones(6)]
    ub = np.r_[panda.qdlim[:n], 10 * np.ones(6)]
    # print(c_Ain)
    # Solve for the joint velocities dq
    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver='daqp')

    # Apply the joint velocities to the Panda
    joint_velocity = qd[:n]

    return joint_velocity

