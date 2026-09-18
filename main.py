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
        
        #self.barista = ABB_IRB120()
        #self.baker = Jonas?
        #self.dispatcher = Lachlan?
        
        self._setup_workcell()
        
    def _setup_workcell(self):
        """
        Add all robots, tables, and props into the Swift environment.
        """

        if self.barista.base_link_mesh is not None:
            self.env.add(self.barista.base_link_mesh)
            
        self.env.add(self.barista)
        
        # Placeholder: Eventually we will add tables, coffee machines, etc. here.
        # table = SE3(0, 0, 0)
        # self.env.add(table)

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
