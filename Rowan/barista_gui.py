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
        tk.Button(sf, text="▶ Run", bg="#1976d2", fg="white", command=self.run_seq).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        self._poll()

    def jog_joint(self, j, direction):
        # Don't move if e-stopped or running auto sequence
        if self.cafe.estop_active or getattr(self.cafe, "is_running", False):
            return
        q = self.cafe.barista.q.copy()
        q[j] += np.deg2rad(5 * direction)
        # Stay within joint limits
        q[j] = np.clip(q[j], self.cafe.barista.qlim[0, j], self.cafe.barista.qlim[1, j])
        self.cafe.barista.q = q
        self.cafe.env.step(0.02)

    def jog_cart(self, axis_idx, direction):
        # Don't move if e-stopped or running auto sequence
        if self.cafe.estop_active or getattr(self.cafe, "is_running", False):
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
        self.cafe.update_safety_visuals()
        self.status.config(text="E-STOPPED", bg="#d32f2f")

    def reset(self):
        # Only allow reset if the light curtain is clear
        if self.cafe.sensor_breached:
            return
        self.cafe.estop_active = False
        self.cafe.update_safety_visuals()
        running = getattr(self.cafe, "is_running", False)
        self.status.config(text="RUNNING" if running else "READY", bg="#1976d2" if running else "#2e7d32")

    def curtain(self):
        # Toggle simulated safety light curtain breach
        self.cafe.sensor_breached = not self.cafe.sensor_breached
        self.cafe.update_safety_visuals()
        self.status.config(
            text="CURTAIN BREACHED" if self.cafe.sensor_breached else "READY",
            bg="#f57c00" if self.cafe.sensor_breached else "#2e7d32"
        )

    def run_seq(self):
        # Run the full coffee routine on a background thread so GUI doesn't freeze
        if getattr(self.cafe, "is_running", False):
            return
        self.cafe.is_running = True
        self.status.config(text="RUNNING", bg="#1976d2")

        def thread_task():
            try:
                self.cafe.run_sequence()
            finally:
                self.cafe.is_running = False
                if not self.cafe.estop_active:
                    self.status.config(text="READY", bg="#2e7d32")

        threading.Thread(target=thread_task, daemon=True).start()

    def _poll(self):
        # Update telemetry labels every 100ms
        try:
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