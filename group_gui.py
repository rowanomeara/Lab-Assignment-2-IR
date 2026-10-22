"""
Unified Teach Pendant GUI for Multi-Robot Automated Cafe
Controls Barista (ABB IRB 120), Baker (Doosan M0609), and Service (Staubli TX60).
"""
import os
import sys
import threading
import tkinter as tk
import numpy as np
from spatialmath import SE3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from main_group import AutomatedCafe

ROBOT_OPTIONS = ["Barista (ABB)", "Baker (Doosan)", "Service (Staubli)"]

class GroupGUI:
    def __init__(self, cafe):
        self.cafe = cafe
        self.root = tk.Tk()
        self.root.title("Cafe Teach Pendant")
        self.root.geometry("360x430")
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(500, lambda: self.root.attributes("-topmost", False))
        self.root.focus_force()

        if not hasattr(self.cafe, "is_paused"):
            self.cafe.is_paused = False

        # Physical Xbox controller for E-Stop (B button)
        try:
            import pygame
            pygame.init()
            self.joy = pygame.joystick.Joystick(0) if pygame.joystick.get_count() > 0 else None
        except Exception:
            self.joy = None

        # 1. Status banner
        self.status = tk.Label(self.root, text="READY", bg="#2e7d32", fg="white", font=("Arial", 11, "bold"))
        self.status.pack(fill=tk.X, padx=5, pady=4)

        # 2. Robot selector dropdown
        rf = tk.Frame(self.root)
        rf.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(rf, text="Active Robot:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=3)
        self.robot_var = tk.StringVar(value=ROBOT_OPTIONS[0])
        self.robot_menu = tk.OptionMenu(rf, self.robot_var, *ROBOT_OPTIONS)
        self.robot_menu.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 3. Telemetry readout
        self.telem = tk.Label(self.root, text="", font=("Courier", 9), justify=tk.LEFT)
        self.telem.pack(padx=5, pady=2)

        # 4. Joint jog section (+/- 5 degrees)
        jf = tk.LabelFrame(self.root, text="Joint Jog (±5°)")
        jf.pack(fill=tk.X, padx=5, pady=2)
        for i in range(6):
            tk.Label(jf, text=f"J{i+1}").grid(row=i // 3, column=(i % 3) * 3, padx=2)
            tk.Button(jf, text="-", width=3, command=lambda j=i: self.jog_joint(j, -1)).grid(row=i // 3, column=(i % 3) * 3 + 1)
            tk.Button(jf, text="+", width=3, command=lambda j=i: self.jog_joint(j, +1)).grid(row=i // 3, column=(i % 3) * 3 + 2)

        # 5. Cartesian jog section (+/- 10 mm)
        cf = tk.LabelFrame(self.root, text="Cartesian Jog (±10mm)")
        cf.pack(fill=tk.X, padx=5, pady=2)
        for i, ax in enumerate(["X", "Y", "Z"]):
            tk.Label(cf, text=ax).grid(row=0, column=i * 3, padx=5)
            tk.Button(cf, text="-", width=4, command=lambda a=i: self.jog_cart(a, -1)).grid(row=0, column=i * 3 + 1)
            tk.Button(cf, text="+", width=4, command=lambda a=i: self.jog_cart(a, +1)).grid(row=0, column=i * 3 + 2)

        # 6. Safety & Run controls
        sf = tk.Frame(self.root)
        sf.pack(fill=tk.X, padx=5, pady=5)
        tk.Button(sf, text="🛑 EMERGENCY STOP", font=("Arial", 10, "bold"), command=self.estop).pack(fill=tk.X, pady=2)
        tk.Button(sf, text="Reset", command=self.reset).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        tk.Button(sf, text="Curtain", command=self.curtain).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        self.run_btn = tk.Button(sf, text="▶ Run", font=("Arial", 10, "bold"), command=self.run_seq)
        self.run_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        self._poll()

    @property
    def active_robot(self):
        """Returns the currently selected 6-DOF robot arm."""
        name = self.robot_var.get()
        if "Baker" in name:
            return self.cafe.baker
        elif "Service" in name:
            return self.cafe.staubli
        return self.cafe.barista

    def jog_joint(self, j, direction):
        if self.cafe.estop_active or self.cafe.sensor_breached or getattr(self.cafe, "is_running", False) or getattr(self.cafe, "is_paused", False):
            return
        bot = self.active_robot
        q = bot.q.copy()
        q[j] += np.deg2rad(5 * direction)
        q[j] = np.clip(q[j], bot.qlim[0, j], bot.qlim[1, j])
        bot.q = q
        self.cafe.env.step(0.02)

    def jog_cart(self, axis_idx, direction):
        if self.cafe.estop_active or self.cafe.sensor_breached or getattr(self.cafe, "is_running", False) or getattr(self.cafe, "is_paused", False):
            return
        bot = self.active_robot
        delta = [0, 0, 0]
        delta[axis_idx] = 0.01 * direction
        T_target = SE3(*delta) * bot.fkine(bot.q)
        ik = bot.ikine_LM(T_target, q0=bot.q)
        if ik.success:
            bot.q = ik.q
            self.cafe.env.step(0.02)

    def estop(self):
        self.cafe.estop_active = True
        self.cafe.is_paused = True
        self._update_status()

    def reset(self):
        if self.cafe.curtain_breached and getattr(self.cafe, "is_running", False):
            return
        self.cafe.estop_active = False
        self.cafe.sensor_breached = False
        if not getattr(self.cafe, "is_running", False):
            self.cafe.is_paused = False
            self.cafe.reset_simulation()
        self._update_status()

    def curtain(self):
        self.cafe.sensor_breached = not self.cafe.sensor_breached
        if self.cafe.sensor_breached:
            self.cafe.is_paused = True
        self._update_status()

    def run_seq(self):
        if self.cafe.estop_active or self.cafe.curtain_breached:
            return

        if getattr(self.cafe, "is_running", False):
            if getattr(self.cafe, "is_paused", False):
                self.cafe.is_paused = False
                self._update_status()
            return

        self.cafe.is_running = True
        self.cafe.is_paused = False
        self._update_status()

        def runner():
            try:
                self.cafe.run_sequence()
            finally:
                self.cafe.is_running = False
                self.cafe.is_paused = False

        threading.Thread(target=runner, daemon=True).start()

    def _update_status(self):
        if self.cafe.estop_active:
            self.status.config(text="E-STOPPED", bg="#d32f2f")
        elif self.cafe.curtain_breached:
            self.status.config(text="CURTAIN BREACHED", bg="#f57c00")
        elif getattr(self.cafe, "is_running", False) and getattr(self.cafe, "is_paused", False):
            self.status.config(text="ACKNOWLEDGED / PAUSED", bg="#f57c00")
        elif getattr(self.cafe, "is_running", False):
            self.status.config(text="RUNNING", bg="#1976d2")
        else:
            self.status.config(text="READY", bg="#2e7d32")

        if getattr(self.cafe, "is_running", False) and getattr(self.cafe, "is_paused", False):
            self.run_btn.config(text="▶ Resume")
        else:
            self.run_btn.config(text="▶ Run")

    def _poll(self):
        if not self.joy:
            try:
                import pygame
                if pygame.joystick.get_count() > 0:
                    self.joy = pygame.joystick.Joystick(0)
                    self.joy.init()
            except Exception:
                pass

        if self.joy:
            import pygame
            pygame.event.pump()
            if self.joy.get_button(1):
                self.estop()

        try:
            self._update_status()
            bot = self.active_robot
            q_deg = np.round(np.rad2deg(bot.q), 1)
            pos = np.round(bot.fkine(bot.q).t, 3)
            self.telem.config(text=f"q: {list(q_deg)}\nPos: X:{pos[0]:+.3f} Y:{pos[1]:+.3f} Z:{pos[2]:+.3f}")
        except Exception:
            pass
        self.root.after(100, self._poll)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    cafe = AutomatedCafe()
    gui = GroupGUI(cafe)
    gui.run()
