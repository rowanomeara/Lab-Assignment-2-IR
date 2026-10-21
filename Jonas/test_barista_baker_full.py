import sys
import os
import numpy as np

# Add repository root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import swift
# Mock swift launch and step without breaking isinstance(env, swift.Swift)
orig_launch = swift.Swift.launch
orig_step = swift.Swift.step
orig_add = swift.Swift.add

swift.Swift.launch = lambda self, *args, **kwargs: None
swift.Swift.step = lambda self, *args, **kwargs: None
swift.Swift.add = lambda self, *args, **kwargs: None

# Now import the class from main_barista+baker
from importlib import import_module
mod = import_module("main_barista+baker")
AutomatedCafe = mod.AutomatedCafe

def test_full_cafe_pipeline():
    print("Initializing AutomatedCafe with headless Swift...")
    cafe = AutomatedCafe()
    
    print("Testing Barista pick up cup...")
    cafe.barista_pick_up_cup()
    
    print("Testing Barista place cup...")
    cafe.barista_place_cup()
    
    print("Testing Barista get coffee...")
    cafe.barista_get_coffee()
    
    print("Testing Barista present cup to Dobot...")
    cafe.barista_present_cup_to_dobot()
    
    print("Testing Dobot pour milk...")
    cafe.dobot_pour_milk()
    
    print("Testing Barista serve coffee (with collision check & detour)...")
    cafe.barista_serve_coffee()
    
    print("Testing Baker pick and place roll (with IBVS, RMRC, & collision detour)...")
    cafe.baker_pick_and_place_roll()
    
    print("\nSUCCESS: All tasks in main_barista+baker executed successfully without error!")

if __name__ == "__main__":
    test_full_cafe_pipeline()
