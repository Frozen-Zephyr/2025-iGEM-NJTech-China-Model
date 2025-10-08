# Intelligent Biomanufacturing System for Bio-based Polyamide PA54

## Overview
This project focuses on the **green synthesis and intelligent process monitoring of bio-based polyamide PA54**, integrating enzyme engineering, metabolic regulation, and automated detection.  
In the enzyme engineering module, key substrate-interacting residues were identified through molecular docking, and a **generative deep learning model based on MPNN-GNN feature extraction** was developed to predict the effects of mutations on lysine decarboxylase (CadA) catalytic performance. Site-directed mutagenesis and activity screening yielded improved enzyme variants.  
At the metabolic level, an **environment-responsive enzyme release system** was designed using A-IDP encapsulation of CadA, which can be activated on demand via protease cleavage. A **kinetic mathematical model based on ordinary differential equations (ODEs)** was constructed to describe the entire process from signal sensing to product formation, enabling prediction of enzyme release timing and production rate.  
For process control, an **ESP32-based automated sampling and detection platform** was developed to achieve periodic sampling, on-line dilution, and enzyme-electrode concentration detection. The data are transmitted wirelessly to a local server and visualized through a web interface, realizing unmanned and digital monitoring.  
The overall workflow achieves a closed-loop optimization from enzyme design to fermentation control, providing an integrated framework for efficient synthesis and intelligent manufacturing of bio-based materials.

## System Components
The project consists of the following five core modules:  
1. **Generative Model of Lysine Decarboxylase Based on MPNN-GNN Feature Extraction** — Predicts mutational effects and identifies high-activity enzyme variants.  
2. **ODEs-based Kinetic Model** — Describes and optimizes enzyme-controlled release dynamics and product formation.  
3. **Rational Design and Optimization of a Cadaverine and Succinate Co-production System Using a Genome-Scale Metabolic Model** — Simulates and optimizes the co-production network at the genome scale.  
4. **Hardware Control Code** — ESP32 firmware for automated sampling, dilution, and electrochemical detection.  
5. **Web Frontend & Backend Code** — Provides real-time data visualization, storage, and remote process monitoring via Flask + Vue.js.

## Usage
Each module can be executed independently or integrated into a unified pipeline for system-level analysis.  
- **Model modules (1–3)** require Python 3.10 and dependencies such as `PyTorch`, `DGL`, `SciPy`, and `cobra`.  
- **Hardware control** runs on the ESP32 platform, with serial communication scripts for server synchronization.  
- **Web interface** is built with `Flask` (backend) and `Vue.js` (frontend) for real-time monitoring and visualization.  

This system bridges molecular design, metabolic control, and intelligent bioprocessing to enable next-generation sustainable biomanufacturing.
