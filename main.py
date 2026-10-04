import numpy as np
from math import pi
import swift
from spatialgeometry import Mesh
from spatialmath import SE3
import os

# Import
from Rowan.abb_irb120 import ABB_IRB120

class AutomatedCafe:
    def __init__(self):
        """
        Initialize the cafe simulation environment.
        """
        self.env = swift.Swift()
        self.env.launch(realtime=True)
        
        self.barista = ABB_IRB120()

        #self.baker = Jonas?a
        #self.dispatcher = Laclan's tx60
        
        self._setup_workcell()
        
    def _setup_workcell(self):
        """
        Add all robots, tables, and props into the Swift environment.
        """
        
        #Barista Robot
        self.barista.base = SE3(-0.4, 0.2, 0.97)
        self.barista.base_link_mesh.T = SE3(-0.4, 0.2, 0.97)
        self.env.add(self.barista.base_link_mesh)
        self.env.add(self.barista)

        import os
        from spatialgeometry import Box
        base_path = os.path.dirname(os.path.abspath(__file__))

        #Counter Top
        counter_path = os.path.join(base_path, "Workcell meshes", "Counter", "counter.stl")
        self.counter = Mesh(counter_path, color=[0.8, 0.8, 0.8, 1.0])
        self.counter.T = SE3(0.5, 0, 0) 
        self.env.add(self.counter)
        
        # Plank
        tabletop = Box(scale=[1.2, 2.2, 0.05], color=[0.35, 0.25, 0.15, 1.0])
        tabletop.T = SE3(-0.3, -0.2, 0.95)#*SE3.Rz(pi)
        self.env.add(tabletop)

        #Coffee Machine
        machine_path = os.path.join(base_path, "Workcell meshes", "Coffee Machine", "machine.stl")
        self.coffee_machine = Mesh(machine_path, color=[0.2, 0.2, 0.2, 1.0])
        self.coffee_machine.T = SE3(0.45, 0.45, 0.97)*SE3.Rx(-pi/2)*SE3.Rz(pi) 
        self.env.add(self.coffee_machine)
        
        # Coffee Cup
        cup_path = os.path.join(base_path, "Workcell meshes", "Coffee Cup", "Coffee+Cup.stl")
        self.cup = Mesh(cup_path, color=[0.9, 0.9, 0.9, 1.0], scale=[0.05, 0.05, 0.05])
        self.cup.T = SE3(0, 0.4, 0.97) 
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


    def run(self):
        """
        Main execution loop for the cafe's logic and trajectories.
        """
        print("Cafe Simulation Running... Close browser to exit.")
        
        while True:
            self.env.step(0.05)

if __name__ == "__main__":

    cafe = AutomatedCafe()
    cafe.run()
