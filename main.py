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

        #self.baker = Jonas?
        #self.dispatcher = Lachlan?
        
        self._setup_workcell()
        
    def _setup_workcell(self):
        """
        Add all robots, tables, and props into the Swift environment.
        """
        
        #Barista Robot
        self.barista.base = SE3(0, 0, 0.97)
        self.barista.base_link_mesh.T = SE3(0, 0, 0.97)
        self.env.add(self.barista.base_link_mesh)
        self.env.add(self.barista)

        import os
        base_path = os.path.dirname(os.path.abspath(__file__))

        #Counter Top
        counter_path = os.path.join(base_path, "Workcell meshes", "Counter", "counter.stl")
        self.counter = Mesh(counter_path, color=[0.8, 0.8, 0.8, 1.0])
        self.counter.T = SE3(0.5, 0, 0) 
        self.env.add(self.counter)

        #Coffee Machine
        machine_path = os.path.join(base_path, "Workcell meshes", "Coffee Machine", "machine.stl")
        self.coffee_machine = Mesh(machine_path, color=[0.2, 0.2, 0.2, 1.0])
        self.coffee_machine.T = SE3(0.3, 0.6, 0.97)*SE3.Rx(pi/2) 
        self.env.add(self.coffee_machine)


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
