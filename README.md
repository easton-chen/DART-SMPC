This repo is source code of the experiment section of the paper "Reliable Proactive Adaptation via Prediction Fusion and Extended Stochastic Model Predictive Control".

This code is based on python 3, and the do_mpc requires the following Python packages and their dependencies:numpy, CasADi, matplotlib. More information could be found on https://www.do-mpc.com/en/latest/index.html.

The basic usage is by running the sript file "run.sh"
>run.sh EXP_TYPE ENV_TYPE PRED_TYPE PLAN_TYPE

The function of each file is as follows:

DS.py: A basic implementation of D-S evidence theory.

DartSim.py: Simulation of DARTSim, the managed system.

Environment: Generate environment from file.

ManagingSystem.py: The managing system.

Predictor.py: An implementation of the forward-looking sensor in DARTSim as the predictor.

main.py: The overall procedure for running the experiment.

smpc.py: An implementation of SMPC based on do_mpc package.