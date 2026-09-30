# genpark-generalized-advantage-estimation-gae-skill

Generalized Advantage Estimation (GAE) engine calculating bias-variance balanced TD advantage estimates.

## Architecture

```mermaid
flowchart RL
    StepT["Time T: Delta_T = r_T - V_T"] --> GAE_T["A_T = Delta_T"]
    StepPrev["Time T-1: Delta_T-1"] --> GAE_Prev["A_T-1 = Delta_T-1 + (gamma * lambda) * A_T"]
    GAE_Prev --> TargetReturns["Target Returns R_t = A_t + V_t"]
```

## Features
- **Configurable Gamma and Lambda**: Balance sample efficiency and bias.
- **Zero Dependencies**: 100% Python Standard Library.
