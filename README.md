# Graph Algorithms & Network Mining Project  

---

## Problem Statement

This project analyzes the robustness of two types of networks—**Random Networks** and **Scale-Free Networks**—under **targeted attacks** and **random failures**.

### Tasks
1. **Generate Networks**
   - **R:** Random Network `G(n, p)` using the Erdős-Rényi (ER) model with *n = 1000+* and *p = 1/2*.  
   - **S:** Scale-Free Network using the Barabási-Albert (BA) algorithm with *1000+ nodes*.
2. **Analyze Network Robustness**
   - Perform **targeted attacks** (removing highest-degree nodes).  
   - Perform **random failures** (removing random nodes).  
   - Compare percolation thresholds, analyze behavior, and visualize outcomes.

---

## Implementation Overview

### 1. Random Network (R)
- **Model:** Erdős-Rényi (ER)  
- **Number of nodes:** 1500  
- **Edge probability (p):** 0.1  

#### Attack Results
- **Targeted Attack Threshold:** 0.950 (95% of nodes removed)  
- **Random Failure Threshold:** 0.990 (99% of nodes removed)

**Observation:**  
The Random Network is **1.04× more robust to random failures** than to targeted attacks.

---

### 2. Scale-Free Network (S)
- **Model:** Barabási-Albert (BA)  
- **Number of nodes:** 1500  
- **Preferential attachment (m):** 3  

#### Attack Results
- **Targeted Attack Threshold:** 0.310 (31% of nodes removed)  
- **Random Failure Threshold:** 0.930 (93% of nodes removed)

**Observation:**  
The Scale-Free Network is **3× more robust to random failures** than to targeted attacks.

---

## 🔍 Analysis

### Percolation Threshold
The **percolation threshold** is the fraction of nodes that must be removed for the **giant component** (largest connected cluster) to disappear.

#### Methodology
- Both networks begin with **1500 nodes**.  
- Nodes are removed in batches of **15 per step** for **100 steps**.  
- The **size of the largest connected component** is recorded at each step.  
- When the component size approaches zero, the network is considered fragmented.

#### Comparative Insights
| Network | Attack Type | Threshold | Robustness | Observation |
|----------|--------------|------------|-------------|--------------|
| **R** | Random | 0.99 | Very High | Uniform connectivity ensures resilience. |
| **R** | Targeted | 0.95 | High | Connections are evenly distributed; no single node is critical. |
| **S** | Random | 0.93 | Moderate | Still robust to random loss due to many low-degree nodes. |
| **S** | Targeted | 0.31 | Low | Vulnerable due to hub nodes acting as key connectors. |

---

## Inference

- The **Random Network (R)** shows homogeneous connectivity, leading to steady degradation under both attack types.  
- The **Scale-Free Network (S)** exhibits heterogeneity, with few highly connected hubs.  
  - These hubs are critical for network integrity.  
  - Once removed (in targeted attacks), the network quickly disintegrates.

---

## Conclusion

- **Random Networks** are more robust to both targeted and random node removals than Scale-Free Networks.  
- **Scale-Free Networks** demonstrate **high vulnerability to targeted attacks** due to the presence of a small number of high-degree hub nodes.  
- **Overall**, the experiment confirms classical network theory predictions:
  - **Homogeneous networks (ER)** degrade uniformly.  
  - **Heterogeneous networks (BA)** collapse rapidly when hubs are removed.

---
