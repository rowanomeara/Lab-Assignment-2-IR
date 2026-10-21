import sys
import os
import numpy as np
from math import pi
from spatialmath import SE3
from machinevisiontoolbox import CentralCamera

# Ensure repo root is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from Jonas.Assignment2 import DoosanM0609

def test_pipeline():
    robot = DoosanM0609()
    robot.base = SE3(-0.6, -1.0, 0.97)
    q_home = np.array([0.0, -pi/6, -pi/2, 0.0, pi/3, 0.0])
    robot.q = q_home.copy()

    # Target roll position in world
    p_roll = np.array([-0.15, -1.27, 0.99])
    
    # 1. Approach near display case
    q_seed = np.array([pi/4, -pi/6, -pi/2, 0.0, pi/3, 0.0])
    T_coarse = SE3(p_roll[0] + 0.03, p_roll[1] - 0.02, p_roll[2] + 0.16) * SE3.Rx(-pi)
    ik_coarse = robot.ikine_LM(T_coarse, q0=q_seed)
    assert ik_coarse.success, "Coarse IK failed"
    robot.q = ik_coarse.q.copy()
    print("Step 1: Arrived at coarse approach pose above display case.")

    # 2. Visual Servoing Test
    cam = CentralCamera(f=0.08, rho=10e-5, imagesize=[1024, 1024], pp=[512, 512])
    pts_world = p_roll[:, None] + np.array([
        [ 0.035, -0.035,  0.035, -0.035],
        [ 0.025,  0.025, -0.025, -0.025],
        [ 0.015,  0.015,  0.015,  0.015]
    ])
    T_desired = SE3(p_roll[0], p_roll[1], p_roll[2] + 0.15) * SE3.Rx(-pi)
    p_star = cam.project_point(T_desired.inv() * pts_world)

    dt = 0.05
    converged = False
    for step in range(35):
        T_cam = robot.fkine(robot.q)
        pts_cam = T_cam.inv() * pts_world
        uv = cam.project_point(pts_cam)
        e = (p_star - uv).flatten('F')
        err = np.linalg.norm(e)
        if err < 4.0:
            print(f"Step 2: IBVS Visual Servoing converged in {step} steps! Final pixel error: {err:.3f} px")
            converged = True
            break
        depth = np.mean(pts_cam[2, :])
        J_img = cam.visjac_p(uv, depth)[:, :3]
        v_cam = 4.0 * np.linalg.pinv(J_img) @ e
        v_world = np.hstack((T_cam.R @ v_cam, [0, 0, 0]))
        J_robot = robot.jacob0(robot.q)
        dq = np.linalg.pinv(J_robot) @ v_world
        robot.q += dq * dt

    assert converged, "Visual servoing did not converge"

    # 3. RMRC Straight Line Descent
    T_grasp = SE3(p_roll[0], p_roll[1], p_roll[2] + 0.04) * SE3.Rx(-pi)
    steps = 20
    for k in range(steps):
        T_curr = robot.fkine(robot.q)
        v_lin = (T_grasp.t - T_curr.t) / max((steps - k) * dt, dt)
        xdot = np.hstack((v_lin, np.zeros(3)))
        J = robot.jacob0(robot.q)
        m = np.sqrt(np.linalg.det(J @ J.T))
        damping = 0.05 if m < 0.02 else 0.0
        J_dls = np.linalg.inv(J.T @ J + (damping**2) * np.eye(6)) @ J.T
        dq = J_dls @ xdot
        robot.q += dq * dt

    dist_to_grasp = np.linalg.norm(robot.fkine(robot.q).t - T_grasp.t)
    print(f"Step 3: RMRC Descent completed! Distance to target: {dist_to_grasp*1000:.2f} mm")
    assert dist_to_grasp < 0.005, "RMRC descent did not reach target"

    # 4. RMRC Straight Line Lift
    T_lift = SE3(p_roll[0], p_roll[1], p_roll[2] + 0.18) * SE3.Rx(-pi)
    for k in range(steps):
        T_curr = robot.fkine(robot.q)
        v_lin = (T_lift.t - T_curr.t) / max((steps - k) * dt, dt)
        xdot = np.hstack((v_lin, np.zeros(3)))
        J = robot.jacob0(robot.q)
        dq = np.linalg.pinv(J) @ xdot
        robot.q += dq * dt

    dist_to_lift = np.linalg.norm(robot.fkine(robot.q).t - T_lift.t)
    print(f"Step 4: RMRC Lift completed! Distance to target: {dist_to_lift*1000:.2f} mm")
    assert dist_to_lift < 0.005, "RMRC lift did not reach target"

    # 5. Collision Avoidance Check
    p_obs = np.array([-0.38, -0.92, 1.05])
    r_obs = 0.07
    h_obs = 0.12

    in_box = lambda p: (abs(p[0] - p_obs[0]) <= r_obs and
                        abs(p[1] - p_obs[1]) <= r_obs and
                        abs(p[2] - p_obs[2]) <= h_obs)

    p_plate = np.array([-0.6, -0.6, 0.985])
    T_hover_plate = SE3(p_plate[0], p_plate[1], p_plate[2] + 0.15) * SE3.Rx(-pi)

    # Check nominal straight path for collision
    nominal_path = [robot.fkine(robot.q).t + alpha * (T_hover_plate.t - robot.fkine(robot.q).t) for alpha in np.linspace(0, 1, 30)]
    collision = any(in_box(p) for p in nominal_path)
    print(f"Step 5: Collision detection check along nominal path: {collision}")
    assert collision, "Collision should have been detected!"

    # Compute Detour
    z_detour = p_obs[2] + h_obs + 0.10
    T_detour = SE3(p_obs[0], p_obs[1], z_detour) * SE3.Rx(-pi)
    print(f"        Active Detour planned at elevation Z = {z_detour:.3f} m")

    print("\nALL CLOSED-LOOP TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_pipeline()
