# Harrow-Hassidim-Lloyd (HHL) Quantum Algorithm in Qiskit

![Domain](https://img.shields.io/badge/Domain-Quantum%20Computing-blueviolet?style=for-the-badge)
![Qiskit](https://img.shields.io/badge/Qiskit-6929C4?style=for-the-badge&logo=qiskit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

---

### 📌 Overview

This repository provides an implementation of the **Harrow-Hassidim-Lloyd (HHL) algorithm** using **Qiskit**. The HHL algorithm is a fundamental quantum computing algorithm designed to solve linear systems of equations of the form:

$$A\mathbf{x} = \mathbf{b}$$

under specific conditions (matrix $A$ being Hermitian and sparse), achieving an exponential speedup $O(\log(N))$ over classical solvers.

---

### ⚡ Quantum Circuit Workflow

*High-level pipeline of the HHL quantum algorithm execution:*

```mermaid
graph LR
    A["State Prep |b⟩"] --> B["Quantum Phase Estimation (QPE)"]
    B --> C["Controlled Ancilla Rotation R(λ⁻¹)"]
    C --> D["Inverse QPE (QPE†)"]
    D --> E["Ancilla Measurement"]
    E -->|Flag = 1| F["Output State |x⟩"]
