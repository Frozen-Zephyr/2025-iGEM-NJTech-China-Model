# Intelligent Biomanufacturing System for Bio-based Polyamide PA54

## Overview
Bio-based polyamide PA54, derived from lysine and succinate, is a sustainable polymer with broad market potential. However, during continuous fermentation, the dynamic fluctuations of key metabolites such as lysine and cadaverine significantly affect both yield and process stability. Conventional manual sampling and measurement methods are labor-intensive, hindering continuous monitoring and precise control, thereby limiting process optimization and efficient product formation.

In this project, we designed a dry-lab platform to support experimental work, integrating enzyme engineering, metabolic regulation, and ODE-based modeling, along with an automated detection device. In the dry-lab studies, molecular simulations and deep learning predictions were used to optimize the catalytic performance of lysine decarboxylase (CadA). In the wet-lab experiments, these designs were applied to actual fermentation systems to validate the effect of engineered enzymes on product formation. To facilitate the experimental workflow, an ESP32-based automated sampling and concentration analysis system was developed, enabling periodic sampling, online dilution, real-time data upload, and remote monitoring and visualization via a web interface.

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
