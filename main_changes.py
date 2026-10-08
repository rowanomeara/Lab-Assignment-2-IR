import numpy as np
from math import pi
import swift
from spatialgeometry import Mesh, Box, Cylinder
from spatialmath import SE3
from roboticstoolbox import jtraj
import os

# Import
from Rowan.abb_irb120 import ABB_IRB120
from Jonas.Assignment2 import DoosanM0609


class AutomatedCafe:
    def __init__(self):
        """
        Initialize the cafe simulation environment.
        """
        self.env = swift.Swift()
        self.env.launch(realtime=True)
        self.barista = ABB_IRB120()
        self.baker = DoosanM0609()

        self._setup_workcell()

    def _setup_workcell(self):
        """
        Add all robots, tables, and props into the Swift environment.
        """
        # Barista Robot
        self.barista.base = SE3(-0.25, 0.2, 0.97)
        self.barista.base_link_mesh.T = SE3(-0.25, 0.2, 0.97)
        self.env.add(self.barista.base_link_mesh)
        self.env.add(self.barista)

        # Baker Robot (Doosan M0609)
        self.baker.base = SE3(-0.25, -0.6, 0.97)
        self.baker.base_link_mesh.T = SE3(-0.25, -0.6, 0.97)
        self.env.add(self.baker.base_link_mesh)
        self.env.add(self.baker)

        base_path = os.path.dirname(os.path.abspath(__file__))

        # Counter Top
        counter_path = os.path.join(base_path, "Workcell meshes", "Counter", "counter.stl")
        self.counter = Mesh(counter_path, color=[0.8, 0.8, 0.8, 1.0])
        self.counter.T = SE3(0.5, 0, 0)
        self.env.add(self.counter)

        # Plank Tabletop
        tabletop = Box(scale=[1.2, 2.2, 0.05], color=[0.35, 0.25, 0.15, 1.0])
        tabletop.T = SE3(-0.3, -0.2, 0.95)
        self.env.add(tabletop)

        # Coffee Machine
        machine_path = os.path.join(base_path, "Workcell meshes", "Coffee Machine", "machine.stl")
        self.coffee_machine = Mesh(machine_path, color=[0.2, 0.2, 0.2, 1.0])
        self.coffee_machine.T = SE3(0.45, 0.45, 0.97) * SE3.Rx(-pi/2) * SE3.Rz(pi)
        self.env.add(self.coffee_machine)

        # Coffee Cup
        cup_path = os.path.join(base_path, "Workcell meshes", "Coffee Cup", "Coffee+Cup.stl")
        self.cup = Mesh(cup_path, color=[0.9, 0.9, 0.9, 1.0], scale=[0.04, 0.04, 0.04])
        self.cup.T = SE3(0, 0.6, 0.97)
        self.env.add(self.cup)

        # Milk Jug
        jug_path = os.path.join(base_path, "Workcell meshes", "Milk Pouring Jug.stl")
        self.jug = Mesh(jug_path, color=[0.7, 0.7, 0.75, 1.0])
        self.jug.T = SE3(0.4, 0.5, 0.97)
        self.env.add(self.jug)

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

        # Serving Tray
        tray_path = os.path.join(base_path, "Workcell meshes", "Serving Tray", "serving_tray.stl")
        self.tray = Mesh(tray_path, color=[0.35, 0.25, 0.18, 1.0])
        self.tray.T = SE3(0.15, -0.40, 0.975)
        self.env.add(self.tray)

        # Plate on Serving Tray
        self.plate = Cylinder(radius=0.10, length=0.015, color=[0.95, 0.95, 0.95, 1.0])
        self.plate.T = SE3(0.15, -0.40, 0.987)
        self.env.add(self.plate)

        # Baked Roll on Tabletop Prep Area
        roll_path = os.path.join(base_path, "Workcell meshes", "Baked Roll", "roll.stl")
        self.roll = Mesh(roll_path, color=[0.82, 0.55, 0.28, 1.0])
        self.roll.T = SE3(0.15, -0.85, 0.975) * SE3.Rx(-pi/2)
        self.env.add(self.roll)
        self.roll_attached = False

    def barista_pick_up_cup(self):
        target_pose = SE3(self.cup.T) * SE3(0.0, -0.11, 0.06) * SE3.Rx(-pi/2)
        hover_pose = SE3(self.cup.T) * SE3(0.0, -0.10, 0.23) * SE3.Rx(-pi/2)
        ik_hover = self.barista.ikine_LM(hover_pose, q0=self.barista.q)
        ik_target = self.barista.ikine_LM(target_pose, q0=ik_hover.q)
        ik_raise = self.barista.ikine_LM(hover_pose, q0=ik_target.q)

        traj_hover = jtraj(self.barista.q, ik_hover.q, 30)
        traj_target = jtraj(ik_hover.q, ik_target.q, 20)
        traj_raise = jtraj(ik_target.q, ik_raise.q, 20)

        for q_step in traj_hover.q:
            self.barista.q = q_step
            self.env.step(0.05)

        for q_step in traj_target.q:
            self.barista.q = q_step
            self.env.step(0.05)

        print("Cup grasped by Barista!")
        self.cup_attached = True
        self.cup_offset = self.barista.fkine(self.barista.q).inv() * SE3(self.cup.T)

        for q_step in traj_raise.q:
            self.barista.q = q_step
            if self.cup_attached:
                self.cup.T = self.barista.fkine(self.barista.q) * self.cup_offset
            self.env.step(0.05)

    def barista_place_cup(self):
        current_pose = self.barista.fkine(self.barista.q)
        target_pose = SE3(0.27, -0.28, -0.07) * current_pose
        ik_target = self.barista.ikine_LM(target_pose, q0=self.barista.q)
        traj_target = jtraj(self.barista.q, ik_target.q, 30)

        for q_step in traj_target.q:
            self.barista.q = q_step
            if self.cup_attached:
                self.cup.T = self.barista.fkine(self.barista.q) * self.cup_offset
            self.env.step(0.05)

        traj_raise = jtraj(self.barista.q, self.barista.q + [0, -50*pi/180, 80*pi/180, 0, 0, 0], 30)
        print("Cup under coffee machine :)")
        self.cup_attached = False

        for q_step in traj_raise.q:
            self.barista.q = q_step
            self.env.step(0.05)

    def barista_get_coffee(self):
        coffee_color = [0.35, 0.18, 0.05, 1.0]
        self.coffee = Cylinder(radius=0.06, length=0.02, color=coffee_color)
        self.coffee.T = SE3(self.cup.T) * SE3(0, 0, 0.08)
        self.env.add(self.coffee)

        target_pose = SE3(self.cup.T) * SE3(0.0, -0.11, 0.06) * SE3.Rx(-pi/2)
        ik_target = self.barista.ikine_LM(target_pose, q0=self.barista.q)
        traj_target = jtraj(self.barista.q, ik_target.q, 30)

        for q_step in traj_target.q:
            self.barista.q = q_step
            self.env.step(0.05)

        traj_raise = jtraj(self.barista.q, self.barista.q + [0, -50*pi/180, 80*pi/180, 0, 0, -25*pi/180], 30)
        print("Cup under coffee machine :)")
        self.cup_attached = True

        for q_step in traj_raise.q:
            self.barista.q = q_step
            if self.cup_attached:
                self.cup.T = self.barista.fkine(self.barista.q) * self.cup_offset
                self.coffee.T = SE3(self.cup.T) * SE3(0, 0, 0.08)
            self.env.step(0.05)

        traj_rot = jtraj(self.barista.q, self.barista.q + [-90*pi/180, 0, 0, 0, 0, 0], 30)

        for q_step in traj_rot.q:
            self.barista.q = q_step
            if self.cup_attached:
                self.cup.T = self.barista.fkine(self.barista.q) * self.cup_offset
                self.coffee.T = SE3(self.cup.T) * SE3(0, 0, 0.08)
            self.env.step(0.05)

    def baker_pick_and_place_roll(self):
        """
        Baker robot picks up the baked roll and places it onto the plate on the serving tray.
        """
        p_roll = np.array([0.15, -0.85, 0.975])
        p_plate = np.array([0.15, -0.40, 1.002])

        T_hover_roll = SE3(p_roll[0], p_roll[1], p_roll[2] + 0.15) * SE3.Rx(pi)
        T_grasp_roll = SE3(p_roll[0], p_roll[1], p_roll[2] + 0.04) * SE3.Rx(pi)
        T_hover_plate = SE3(p_plate[0], p_plate[1], p_plate[2] + 0.15) * SE3.Rx(pi)
        T_drop_plate = SE3(p_plate[0], p_plate[1], p_plate[2] + 0.04) * SE3.Rx(pi)

        ik_hover_roll = self.baker.ikine_LM(T_hover_roll, q0=self.baker.q)
        ik_grasp_roll = self.baker.ikine_LM(T_grasp_roll, q0=ik_hover_roll.q)
        ik_lift_roll = self.baker.ikine_LM(T_hover_roll, q0=ik_grasp_roll.q)
        ik_hover_plate = self.baker.ikine_LM(T_hover_plate, q0=ik_lift_roll.q)
        ik_drop_plate = self.baker.ikine_LM(T_drop_plate, q0=ik_hover_plate.q)
        ik_retract = self.baker.ikine_LM(T_hover_plate, q0=ik_drop_plate.q)

        # 1. Approach above roll
        traj1 = jtraj(self.baker.q, ik_hover_roll.q, 25)
        for q in traj1.q:
            self.baker.q = q
            self.env.step(0.05)

        # 2. Lower to roll
        traj2 = jtraj(ik_hover_roll.q, ik_grasp_roll.q, 15)
        for q in traj2.q:
            self.baker.q = q
            self.env.step(0.05)

        print("Baked roll grasped by Baker!")
        self.roll_attached = True
        self.roll_offset = self.baker.fkine(self.baker.q).inv() * SE3(self.roll.T)

        # 3. Lift roll
        traj3 = jtraj(ik_grasp_roll.q, ik_lift_roll.q, 15)
        for q in traj3.q:
            self.baker.q = q
            self.roll.T = self.baker.fkine(self.baker.q) * self.roll_offset
            self.env.step(0.05)

        # 4. Transit over plate on serving tray
        traj4 = jtraj(ik_lift_roll.q, ik_hover_plate.q, 25)
        for q in traj4.q:
            self.baker.q = q
            self.roll.T = self.baker.fkine(self.baker.q) * self.roll_offset
            self.env.step(0.05)

        # 5. Lower roll onto plate
        traj5 = jtraj(ik_hover_plate.q, ik_drop_plate.q, 15)
        for q in traj5.q:
            self.baker.q = q
            self.roll.T = self.baker.fkine(self.baker.q) * self.roll_offset
            self.env.step(0.05)

        print("Baked roll placed on plate atop serving tray :)")
        self.roll_attached = False

        # 6. Retract from plate
        traj6 = jtraj(ik_drop_plate.q, ik_retract.q, 15)
        for q in traj6.q:
            self.baker.q = q
            self.env.step(0.05)

        # 7. Return to home configuration
        traj_home = jtraj(self.baker.q, np.zeros(6), 25)
        for q in traj_home.q:
            self.baker.q = q
            self.env.step(0.05)

    def run(self):
        """
        Main execution loop for the cafe's logic and trajectories.
        """
        print("Cafe Simulation Running... Close browser to exit.")
        self.barista_pick_up_cup()
        self.barista_place_cup()
        self.barista_get_coffee()

        self.baker_pick_and_place_roll()

        while True:
            self.env.step(0.05)


if __name__ == "__main__":
    cafe = AutomatedCafe()
    cafe.run()
