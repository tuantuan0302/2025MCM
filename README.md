# 2025 MCM  
**Smoke-Chaff Deployment Strategy Optimization & Simulation System**

---

**Battlefield Modeling** – 3D space initialization  
One-click generation of missile, UAV, decoy and real-target coordinates.

**Trajectory Prediction** – Missile flight path calculation  
Constant-velocity straight-line model outputs missile position at any instant.

**Chaff Modeling** – Cartridge trajectory & cloud descent  
Gravity included; cloud sinks at 3 m/s, 10 m radius effective within 20 s.

**Strategy Optimization** – Differential-evolution global search  
Automatically returns speed, release point and detonation point that maximize obscuration time.

**Multi-Bomb Scheduling** – 1 s minimum interval  
Single-UAV triple-release sequence satisfies safety spacing without human tuning.

**Cooperative Planning** – Multi-UAV parallel optimization  
Three UAVs planned simultaneously to maximize total obscuration time.

**Real-Time Evaluation** – Dynamic effective-window slicing  
0.1 s step size computes exact obscuration window of each chaff cloud on the missile.

**3D Visualization** – Trajectory + cloud + event points  
One-click 3-D trajectory plot and X-Z projection, auto-saved as PNG.


