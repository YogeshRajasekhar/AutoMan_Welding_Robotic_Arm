# 🔧 Robotic Arc Welding with ABB KR6 in MuJoCo

This repository contains a complete simulation and control framework for **robotic arc welding** using an **ABB KR6 industrial manipulator**, implemented in the **MuJoCo physics engine**. The goal of this project is to simulate the robotic welding process—including precise toolpath tracking, welding torch dynamics, and control logic—prior to real-world deployment, improving safety, accuracy, and process development time.

---

## 📖 Project Overview

Robotic arc welding is one of the most demanding applications in industrial automation due to the precision, coordination, and path fidelity required. In this project, the **ABB KR6 R900 sixx** robot model is recreated in MuJoCo using accurate CAD and kinematic data, and controlled via Python scripts to follow realistic welding paths. The simulation framework can be used for:

- Path planning and evaluation
- Kinematic and dynamic analysis
- Torch orientation control
- Weld seam tracking
- Sensor-based feedback control (future)
- Offline programming validation

---

## 🔍 Key Features

- ✅ MuJoCo XML model of ABB KR6 with accurate DH parameters and joint limits
- ✅ Simulated torch tip tracking along weld seams
- ✅ Visualization tools for toolpaths and robot postures
- ✅ Modular CAD-to-MuJoCo model integration
- ✅ Forward and inverse kinematics utilities
- 🧪 Support for planned force or impedance control
- 📦 Easily extendable to other industrial tasks (grinding, painting, inspection)

---
