import numpy as np
from math import pi
import swift
from spatialgeometry import Mesh
from spatialmath import SE3
from ir_support.robots import DobotMagician
from ir_support_extra_parts.parts import part_mesh 
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Import
from Rowan.abb_irb120 import ABB_IRB120
from Jonas.Assignment2 import DoosanM0609
from Lachlan.LachlanMesh.staubli_tx60.staubliTX60 import StaubliTX60

class AutomatedCafe:
    def __init__(self):
        """
        Initialize the cafe simulation environment.
        """
        self.env = swift.Swift()
        self.env.launch(realtime=True)
        
        self.barista = ABB_IRB120()
        
        self.baker = DoosanM0609()
        
        self.staubli = StaubliTX60()
        
        self._setup_workcell()
        
    def _setup_workcell(self):
        """
        Add all robots, tables, and props into the Swift environment.
        """
        
        #Rowan's Barista Robot
        self.barista.base = SE3(-0.25, 0.2, 0.97)
        self.barista.base_link_mesh.T = SE3(-0.25, 0.2, 0.97)
        self.env.add(self.barista.base_link_mesh)
        self.env.add(self.barista)
        
        #Jonas's Baker Robot
        self.baker.base = SE3(-0.25, -1.8, 0.97)
        self.baker.base_link_mesh.T = SE3(-0.25, -1.8, 0.97)
        self.env.add(self.baker.base_link_mesh)
        self.env.add(self.baker)
        
        #Lachlan's Staubli Robot
        self.staubli.base = SE3(-0.92, -0.45, 0.97)
        self.staubli.base_link_mesh.T = SE3(-0.92, -0.45, 0.97)
        self.env.add(self.staubli.base_link_mesh)
        self.env.add(self.staubli)
        
        #Dobot 
        dobot_base = SE3(-0.85, 0.0, 0.97)
        self.dobot = DobotMagician(base=dobot_base)
        self.dobot.base = dobot_base
        
        # Making the jug level:
        q2_ready = 25 * pi / 180
        q3_ready = 20 * pi / 180
        q4_level = -(q2_ready + q3_ready)
        self.dobot_q_ready = np.array([0.0, q2_ready, q3_ready, q4_level, 0.0])
        self.dobot.q = self.dobot_q_ready
        
        import os
        from spatialgeometry import Box
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        #Counter Top
        counter_path = os.path.join(base_path, "Workcell meshes", "Counter", "counter.stl")
        self.counter = Mesh(counter_path, color=[0.8, 0.8, 0.8, 1.0])
        self.counter.T = SE3(0.5, 0, 0) 
        self.env.add(self.counter)
        
        # Plank
        tabletop = Box(scale=[1.8, 2.2, 0.05], color=[0.35, 0.25, 0.15, 1.0])
        tabletop.T = SE3(-0.6, -0.2, 0.95)
        self.env.add(tabletop)

        #Coffee Machine
        machine_path = os.path.join(base_path, "Workcell meshes", "Coffee Machine", "machine.stl")
        self.coffee_machine = Mesh(machine_path, color=[0.2, 0.2, 0.2, 1.0])
        self.coffee_machine.T = SE3(0.45, 0.45, 0.97)*SE3.Rx(-pi/2)*SE3.Rz(pi) 
        self.env.add(self.coffee_machine)
        
        # Coffee Cup
        cup_path = os.path.join(base_path, "Workcell meshes", "Coffee Cup", "Coffee+Cup.stl")
        self.cup = Mesh(cup_path, color=[0.9, 0.9, 0.9, 1.0], scale=[0.04, 0.04, 0.04])
        self.cup.T = SE3(0, 0.6, 0.97) 
        self.env.add(self.cup)
        
        # Milk Jug
        jug_path = os.path.join(base_path, "Workcell meshes", "Milk Pouring Jug.stl")
        self.jug = Mesh(jug_path, color=[0.7, 0.7, 0.75, 1.0], scale=[0.001, 0.001, 0.001])
        self.jug_offset = SE3.Ry(pi) * SE3(-0.168, -0.091, -0.081)
        self.jug.T = self.dobot.fkine(self.dobot.q) * self.jug_offset
        self.env.add(self.jug)
        self.dobot.add_to_env(self.env)
        
        # Service Bell 
        bell_path = os.path.join(base_path, "Workcell meshes", "Service Bell", "Table_Bell.stl")
        self.bell = Mesh(bell_path, color=[0.8, 0.6, 0.1, 1.0])
        self.bell.T = SE3(0.6, -1, 0.97)
        self.env.add(self.bell)
        
        # Cash Register
        register_path = os.path.join(base_path, "Workcell meshes", "Cash Register", "registermachine.stl")
        self.register = Mesh(register_path, color=[0.3, 0.3, 0.3, 1.0])
        self.register.T = SE3(-0.7, -0.35, 0.02) * SE3.Rz(-pi/4)
        self.env.add(self.register)
        
        #Tray
        self.tray = part_mesh("Tray")
        self.tray.T = SE3(-0.55, -0.5, 0.97)
        self.env.add(self.tray)
        
    def barista_pick_up_cup(self):
        from roboticstoolbox import jtraj
        
        target_pose = SE3(self.cup.T) * SE3(0.0, -0.11, 0.06) * SE3.Rx(-pi/2)
        hover_pose = SE3(self.cup.T) * SE3(0.0, -0.10, 0.23) * SE3.Rx(-pi/2)
        
        ik_hover  = self.barista.ikine_LM(hover_pose,  q0=self.barista.q)
        ik_target = self.barista.ikine_LM(target_pose, q0=ik_hover.q)
        ik_raise  = self.barista.ikine_LM(hover_pose,  q0=ik_target.q)   
        
        traj_hover  = jtraj(self.barista.q, ik_hover.q, 10)
        traj_target = jtraj(ik_hover.q,     ik_target.q, 5)
        traj_raise  = jtraj(ik_target.q,    ik_raise.q, 5)
        
        #Move to Hover
        for q_step in traj_hover.q:
            self.barista.q = q_step
            self.env.step(0.05)
            
        # Lower to Cup
        for q_step in traj_target.q:
            self.barista.q = q_step
            self.env.step(0.05)
            
        self.cup_attached = True
        self.cup_offset = self.barista.fkine(self.barista.q).inv() * SE3(self.cup.T)
        
        for q_step in traj_raise.q:
            self.barista.q = q_step
            if self.cup_attached:
                self.cup.T = self.barista.fkine(self.barista.q) * self.cup_offset
            self.env.step(0.05)
            
        
    def barista_place_cup(self):
        from roboticstoolbox import jtraj
        
        current_pose = self.barista.fkine(self.barista.q)
        target_pose = SE3(0.27, -0.28, -0.07) * current_pose
        
        ik_target = self.barista.ikine_LM(target_pose, q0=self.barista.q)
        
        traj_target = jtraj(self.barista.q,  ik_target.q, 10)
            
        for q_step in traj_target.q:
            self.barista.q = q_step
            if self.cup_attached:
                self.cup.T = self.barista.fkine(self.barista.q) * self.cup_offset
            self.env.step(0.05)
        
        traj_raise  = jtraj(self.barista.q,  self.barista.q + [0, -50*pi/180, 80*pi/180, 0, 0, 0], 10)
            
        self.cup_attached = False
        
        for q_step in traj_raise.q:
            self.barista.q = q_step
            self.env.step(0.05)
            
    def barista_get_coffee(self):
        from roboticstoolbox import jtraj
        from spatialgeometry import Cylinder
        
        coffee_color = [0.15, 0.08, 0.03, 1.0]
        self.coffee = Cylinder(radius=0.06, length=0.02, color=coffee_color)
        self.coffee.T = SE3(self.cup.T) * SE3(0, 0, 0.08)
        self.env.add(self.coffee)
        
        target_pose = SE3(self.cup.T) * SE3(0.0, -0.11, 0.06) * SE3.Rx(-pi/2)
        
        ik_target = self.barista.ikine_LM(target_pose, q0=self.barista.q)
        
        traj_target = jtraj(self.barista.q,  ik_target.q, 20)
            
        for q_step in traj_target.q:
            self.barista.q = q_step
            self.env.step(0.05)
        
        traj_raise  = jtraj(self.barista.q,  self.barista.q + [0, -50*pi/180, 80*pi/180, 0, 0, -25*pi/180], 20)
            
        self.cup_attached = True
        
        for q_step in traj_raise.q:
            self.barista.q = q_step
            if self.cup_attached:
                self.cup.T = self.barista.fkine(self.barista.q) * self.cup_offset
                self.coffee.T = SE3(self.cup.T) * SE3(0, 0, 0.08)
            self.env.step(0.05)
            
        traj_rot  = jtraj(self.barista.q,  self.barista.q + [-90*pi/180, 0, 0, 0, 0, 0], 30)
        
        for q_step in traj_rot.q:
            self.barista.q = q_step
            if self.cup_attached:
                self.cup.T = self.barista.fkine(self.barista.q) * self.cup_offset
                self.coffee.T = SE3(self.cup.T) * SE3(0, 0, 0.08)
            self.env.step(0.05)
            
    def barista_present_cup_to_dobot(self):
        from roboticstoolbox import jtraj
        
        handover_cup_pose = self.dobot.base * SE3(0.28, 0.0, 0.16)
        
        ee_target = handover_cup_pose * self.cup_offset.inv()
        
        ik_present = self.barista.ikine_LM(ee_target, q0=self.barista.q)
        
        traj_present = jtraj(self.barista.q, ik_present.q, 25)
        for q_step in traj_present.q:
            self.barista.q = q_step
            if self.cup_attached:
                self.cup.T = self.barista.fkine(self.barista.q) * self.cup_offset
                self.coffee.T = SE3(self.cup.T) * SE3(0, 0, 0.08)
            self.env.step(0.05)

    def dobot_pour_milk(self):
        from roboticstoolbox import jtraj
        from spatialgeometry import Cylinder

        def move(q_target, steps):
            for q in jtraj(self.dobot.q, q_target, steps).q:
                self.dobot.q = q
                self.jug.T = self.dobot.fkine(self.dobot.q) * self.jug_offset
                self.env.step(0.05)

        T_cup_rim = (SE3(self.cup.T) * SE3(-0.05, 0.0, 0.125)).t
        T_ee_target = SE3(T_cup_rim - [0.156, 0.0, 0.080]) * SE3.Rx(pi)

        sol_reach = self.dobot.ikine_LM(T_ee_target, q0=self.dobot.q, mask=[1, 1, 1, 0, 0, 0])
        q_reach = sol_reach.q.copy()
        q_reach[3] = -(q_reach[1] + q_reach[2])

        move(q_reach, 20)

        q_pour = q_reach.copy()
        q_pour[3] += 25 * pi / 180
        move(q_pour, 15)

        self.milk = Cylinder(radius=0.058, length=0.015, color=[0.55, 0.36, 0.20, 1.0])
        self.milk.T = SE3(self.cup.T) * SE3(0, 0, 0.088)
        self.env.add(self.milk)

        move(q_reach, 15)
        move(self.dobot_q_ready, 20)
        
    def barista_serve_coffee(self):
        from roboticstoolbox import jtraj

        target_cup = SE3(self.tray.T) * SE3(0, 0, 0.03)
        ee_target = target_cup * SE3.Rz(65 * pi / 180) * self.cup_offset.inv()
        
        ik_serve = self.barista.ikine_LM(ee_target, q0=self.barista.q)

        traj_serve = jtraj(self.barista.q, ik_serve.q, 25)
        for q_step in traj_serve.q:
            self.barista.q = q_step
            self.cup.T = self.barista.fkine(self.barista.q) * self.cup_offset
            self.coffee.T = SE3(self.cup.T) * SE3(0, 0, 0.08)
            self.milk.T = SE3(self.cup.T) * SE3(0, 0, 0.088)
            self.env.step(0.05)

        self.cup_attached = False

        traj_home = jtraj(self.barista.q, np.zeros(6), 20)
        for q_step in traj_home.q:
            self.barista.q = q_step
            self.env.step(0.05)

    def staubli_serve_tray(self):
        from roboticstoolbox import jtraj

        p_cup = SE3(self.cup.T).t
        p_tray = SE3(self.tray.T).t

        T_hover_cup = SE3(p_cup[0], p_cup[1], p_cup[2] + 0.22) * SE3.Ry(pi)
        T_grasp_cup = SE3(p_cup[0], p_cup[1], p_cup[2] + 0.11) * SE3.Ry(pi)

        ik_hover_cup = self.staubli.ikine_LM(T_hover_cup, q0=[0.3, 0.4, 0.6, 0, 0.5, 0])
        ik_grasp_cup = self.staubli.ikine_LM(T_grasp_cup, q0=ik_hover_cup.q)

        traj_hover_cup = jtraj(self.staubli.q, ik_hover_cup.q, 25)
        traj_grasp_cup = jtraj(ik_hover_cup.q, ik_grasp_cup.q, 15)
        traj_lift_cup  = jtraj(ik_grasp_cup.q, ik_hover_cup.q, 15)

        for q_step in traj_hover_cup.q:
            self.staubli.q = q_step
            self.env.step(0.05)

        for q_step in traj_grasp_cup.q:
            self.staubli.q = q_step
            self.env.step(0.05)

        self.cup_offset = self.staubli.fkine(self.staubli.q).inv() * SE3(self.cup.T)

        for q_step in traj_lift_cup.q:
            self.staubli.q = q_step
            self.cup.T = self.staubli.fkine(self.staubli.q) * self.cup_offset
            self.coffee.T = SE3(self.cup.T) * SE3(0, 0, 0.08)
            self.milk.T = SE3(self.cup.T) * SE3(0, 0, 0.088)
            self.env.step(0.05)

        p_place = p_tray + [0.04, 0.0, 0.03]
        T_place_hover = SE3(p_place[0], p_place[1], p_place[2] + 0.25) * SE3.Ry(pi)
        T_place_cup   = SE3(p_place[0], p_place[1], p_place[2] + 0.11) * SE3.Ry(pi)

        ik_place_hover = self.staubli.ikine_LM(T_place_hover, q0=self.staubli.q)
        ik_place_cup   = self.staubli.ikine_LM(T_place_cup,   q0=ik_place_hover.q)

        traj_place_hover   = jtraj(self.staubli.q,     ik_place_hover.q, 20)
        traj_place_cup     = jtraj(ik_place_hover.q,   ik_place_cup.q,   15)
        traj_place_retract = jtraj(ik_place_cup.q,     ik_place_hover.q, 15)

        for q_step in traj_place_hover.q:
            self.staubli.q = q_step
            self.cup.T = self.staubli.fkine(self.staubli.q) * self.cup_offset
            self.coffee.T = SE3(self.cup.T) * SE3(0, 0, 0.08)
            self.milk.T = SE3(self.cup.T) * SE3(0, 0, 0.088)
            self.env.step(0.05)

        for q_step in traj_place_cup.q:
            self.staubli.q = q_step
            self.cup.T = self.staubli.fkine(self.staubli.q) * self.cup_offset
            self.coffee.T = SE3(self.cup.T) * SE3(0, 0, 0.08)
            self.milk.T = SE3(self.cup.T) * SE3(0, 0, 0.088)
            self.env.step(0.05)

        for q_step in traj_place_retract.q:
            self.staubli.q = q_step
            self.env.step(0.05)

        p_handle = p_tray + [-0.14, 0.0, 0.04]
        T_tray_hover = SE3(p_handle[0], p_handle[1], p_handle[2] + 0.18) * SE3.Ry(pi)
        T_tray_grasp = SE3(p_handle[0], p_handle[1], p_handle[2]) * SE3.Ry(pi)
        T_tray_lift  = SE3(p_handle[0], p_handle[1], p_handle[2] + 0.28) * SE3.Ry(pi)

        ik_tray_hover = self.staubli.ikine_LM(T_tray_hover, q0=self.staubli.q)
        ik_tray_grasp = self.staubli.ikine_LM(T_tray_grasp, q0=ik_tray_hover.q)
        ik_tray_lift  = self.staubli.ikine_LM(T_tray_lift,  q0=ik_tray_grasp.q)

        traj_tray_hover = jtraj(self.staubli.q,   ik_tray_hover.q, 15)
        traj_tray_grasp = jtraj(ik_tray_hover.q,  ik_tray_grasp.q, 15)
        traj_tray_lift  = jtraj(ik_tray_grasp.q,  ik_tray_lift.q,  20)

        for q_step in traj_tray_hover.q:
            self.staubli.q = q_step
            self.env.step(0.05)

        for q_step in traj_tray_grasp.q:
            self.staubli.q = q_step
            self.env.step(0.05)

        self.tray_offset = self.staubli.fkine(self.staubli.q).inv() * SE3(self.tray.T)
        self.cup_offset  = self.staubli.fkine(self.staubli.q).inv() * SE3(self.cup.T)

        for q_step in traj_tray_lift.q:
            self.staubli.q = q_step
            self.tray.T = self.staubli.fkine(self.staubli.q) * self.tray_offset
            self.cup.T = self.staubli.fkine(self.staubli.q) * self.cup_offset
            self.coffee.T = SE3(self.cup.T) * SE3(0, 0, 0.08)
            self.milk.T = SE3(self.cup.T) * SE3(0, 0, 0.088)
            self.env.step(0.05)

        q_rot = self.staubli.q + [pi, 0, 0, 0, 0, 0]
        traj_rot = jtraj(self.staubli.q, q_rot, 40)

        for q_step in traj_rot.q:
            self.staubli.q = q_step
            self.tray.T = self.staubli.fkine(self.staubli.q) * self.tray_offset
            self.cup.T = self.staubli.fkine(self.staubli.q) * self.cup_offset
            self.coffee.T = SE3(self.cup.T) * SE3(0, 0, 0.08)
            self.milk.T = SE3(self.cup.T) * SE3(0, 0, 0.088)
            self.env.step(0.05)

    def run(self):
        """
        Main execution loop for the cafe's logic and trajectories.
        """
        self.barista_pick_up_cup()
        
        self.barista_place_cup()
        
        self.barista_get_coffee()

        self.barista_present_cup_to_dobot()

        self.dobot_pour_milk()
        
        self.barista_serve_coffee()

        self.staubli_serve_tray()
        
        while True:
            self.env.step(0.05)

if __name__ == "__main__":

    cafe = AutomatedCafe()
    cafe.run()

