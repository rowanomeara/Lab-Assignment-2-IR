"""
Teach Pendant GUI for Barista
"""
import threading
import tkinter as tk
import numpy as np
from spatialmath import SE3

class BaristaGUI:
    def __init__(self, cafe):
        self.cafe = cafe
        self.root = tk.Tk()
        self.root.title("IRB 120 Teach Pendant")
        self.root.geometry("360x430")
        self.root.lift()
        self.root.focus_force()

        if not hasattr(self.cafe, "is_paused"):
            self.cafe.is_paused = False

        # Physical Xbox controller for E-Stop
        try:
            import pygame
            pygame.init()
            self.joy = pygame.joystick.Joystick(0) if pygame.joystick.get_count() > 0 else None
        except Exception:
            self.joy = None

        # Top status banner showing current state
        self.status = tk.Label(self.root, text="READY", bg="#2e7d32", fg="white", font=("Arial", 11, "bold"))
        self.status.pack(fill=tk.X, padx=5, pady=5)

        # Telemetry readout for joint angles and XYZ position
        self.telem = tk.Label(self.root, text="", font=("Courier", 9), justify=tk.LEFT)
        self.telem.pack(padx=5, pady=2)

        # Joint jogging section (+/- 5 degrees)
        jf = tk.LabelFrame(self.root, text="Joint Jog (±5°)")
        jf.pack(fill=tk.X, padx=5, pady=2)
        for i in range(6):
            tk.Label(jf, text=f"J{i+1}").grid(row=i // 3, column=(i % 3) * 3, padx=2)
            tk.Button(jf, text="-", width=3, command=lambda j=i: self.jog_joint(j, -1)).grid(row=i // 3, column=(i % 3) * 3 + 1)
            tk.Button(jf, text="+", width=3, command=lambda j=i: self.jog_joint(j, +1)).grid(row=i // 3, column=(i % 3) * 3 + 2)

        # Cartesian jogging section (+/- 10 mm)
        cf = tk.LabelFrame(self.root, text="Cartesian Jog (±10mm)")
        cf.pack(fill=tk.X, padx=5, pady=2)
        for i, ax in enumerate(["X", "Y", "Z"]):
            tk.Label(cf, text=ax).grid(row=0, column=i * 3, padx=5)
            tk.Button(cf, text="-", width=4, command=lambda a=i: self.jog_cart(a, -1)).grid(row=0, column=i * 3 + 1)
            tk.Button(cf, text="+", width=4, command=lambda a=i: self.jog_cart(a, +1)).grid(row=0, column=i * 3 + 2)

        # Safety buttons and auto-run trigger
        sf = tk.Frame(self.root)
        sf.pack(fill=tk.X, padx=5, pady=5)
        tk.Button(sf, text="🛑 E-STOP", bg="#d32f2f", fg="white", font=("Arial", 10, "bold"), command=self.estop).pack(fill=tk.X, pady=2)
        tk.Button(sf, text="Reset", command=self.reset).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        tk.Button(sf, text="Curtain", command=self.curtain).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        self.run_btn = tk.Button(sf, text="▶ Run", bg="#1976d2", fg="white", command=self.run_seq)
        self.run_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        self._poll()

    def jog_joint(self, j, direction):
        # Don't move if e-stopped, curtain breached, running auto sequence, or paused
        if self.cafe.estop_active or self.cafe.sensor_breached or getattr(self.cafe, "is_running", False) or getattr(self.cafe, "is_paused", False):
            return
        q = self.cafe.barista.q.copy()
        q[j] += np.deg2rad(5 * direction)
        # Stay within joint limits
        q[j] = np.clip(q[j], self.cafe.barista.qlim[0, j], self.cafe.barista.qlim[1, j])
        self.cafe.barista.q = q
        self.cafe.env.step(0.02)

    def jog_cart(self, axis_idx, direction):
        # Don't move if e-stopped, curtain breached, running auto sequence, or paused
        if self.cafe.estop_active or self.cafe.sensor_breached or getattr(self.cafe, "is_running", False) or getattr(self.cafe, "is_paused", False):
            return
        delta = [0, 0, 0]
        delta[axis_idx] = 0.01 * direction
        # Apply translation and solve IK from current joint pose
        T_target = SE3(*delta) * self.cafe.barista.fkine(self.cafe.barista.q)
        ik = self.cafe.barista.ikine_LM(T_target, q0=self.cafe.barista.q)
        if ik.success:
            self.cafe.barista.q = ik.q
            self.cafe.env.step(0.02)

    def estop(self):
        # Freeze the robot right away
        self.cafe.estop_active = True
        self.cafe.is_paused = True
        self._update_status()

    def reset(self):
        # Only allow reset if the light curtain is clear
        if self.cafe.sensor_breached:
            return
        self.cafe.estop_active = False
        if not getattr(self.cafe, "is_running", False):
            self.cafe.is_paused = False
            self.cafe.reset_simulation()
        self._update_status()

    def curtain(self):
        # Toggle simulated safety light curtain breach
        self.cafe.sensor_breached = not self.cafe.sensor_breached
        self._update_status()

    def run_seq(self):
        # Cannot run or resume if e-stopped or light curtain breached
        if self.cafe.estop_active or self.cafe.sensor_breached:
            return

        # If already running but paused (armed/acknowledged state), resume motion
        if getattr(self.cafe, "is_running", False):
            if getattr(self.cafe, "is_paused", False):
                self.cafe.is_paused = False
                self._update_status()
            return

        # Otherwise start the automated sequence from idle
        self.cafe.is_running = True
        self.cafe.is_paused = False
        self._update_status()

        def thread_task():
            try:
                self.cafe.run_sequence()
            finally:
                self.cafe.is_running = False
                self.cafe.is_paused = False

        threading.Thread(target=thread_task, daemon=True).start()

    def _update_status(self):
        if self.cafe.estop_active:
            self.status.config(text="E-STOPPED", bg="#d32f2f")
        elif self.cafe.sensor_breached:
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
        # Check physical Xbox controller E-Stop (B button)
        if self.joy:
            import pygame
            pygame.event.pump()
            if self.joy.get_button(1):
                self.estop()

        # Update telemetry labels and status banner every 100ms
        try:
            self._update_status()
            q_deg = np.round(np.rad2deg(self.cafe.barista.q), 1)
            pos = np.round(self.cafe.barista.fkine(self.cafe.barista.q).t, 3)
            self.telem.config(text=f"q: {list(q_deg)}\nPos: X:{pos[0]:+.3f} Y:{pos[1]:+.3f} Z:{pos[2]:+.3f}")
        except Exception:
            pass
        self.root.after(100, self._poll)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    from main_barista import AutomatedCafe
    cafe = AutomatedCafe()
    gui = BaristaGUI(cafe)
    gui.run()