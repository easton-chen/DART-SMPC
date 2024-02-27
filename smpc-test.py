import do_mpc

import numpy as np
import pandas as pd
from casadi import *
import time
import random

class MPCController:
    def __init__(self, horizon, robust_horizon, n, f):
        self.horizon = horizon
        self.robust_horizon = robust_horizon
        self.n = n
        self.log_file = f

        self.defineModel()
        self.setupSMPC()
        self.setupSim()
        
    def defineModel(self):
        self.model = do_mpc.model.Model('discrete')

        self.x_1_a = self.model.set_variable(var_type='_x', var_name='x_1_a', shape=(1,1))
        self.x_2_f = self.model.set_variable(var_type='_x', var_name='x_2_f', shape=(1,1))
        self.x_3_e = self.model.set_variable(var_type='_x', var_name='x_3_e', shape=(1,1))
        
    
        self.u_1_a = self.model.set_variable(var_type='_u', var_name='u_1_a')
        self.u_2_f = self.model.set_variable(var_type='_u', var_name='u_2_f')
        self.u_3_e = self.model.set_variable(var_type='_u', var_name='u_3_e')

    
        self.c_1_target = self.model.set_variable('_p', 'c_1_target') 
        self.c_2_threat = self.model.set_variable('_p', 'c_2_threat')
        
        
        x_1_next = self.x_1_a + self.u_1_a
        x_2_next = self.x_2_f + self.u_2_f
        x_3_next = self.u_3_e
        
        self.model.set_rhs('x_1_a', x_1_next)
        self.model.set_rhs('x_2_f', x_2_next)
        self.model.set_rhs('x_3_e', x_3_next)
        
        self.model.setup()

    def defineModelN(self):
        self.model = do_mpc.model.Model('discrete')
        self.x = []
        self.u = []
        for i in range(self.n):
            xvarname = 'x_' + str(i+1)
            self.x.append(self.model.set_variable(var_type='_x', var_name=xvarname, shape=(1,1)))
            uvarname = 'u_' + str(i+1)
            self.u.append(self.model.set_variable(var_type='_u', var_name=uvarname, shape=(1,1)))
    
        self.c_1_target = self.model.set_variable('_p', 'c_1_target') 
        self.c_2_threat = self.model.set_variable('_p', 'c_2_threat')
        
        for i in range(self.n):
            xvarname = 'x_' + str(i+1)
            self.model.set_rhs(xvarname, self.x[i] + self.u[i])
        
        self.model.setup()

    def setupSMPC(self):
        starttime = time.time()
        self.mpc = do_mpc.controller.MPC(self.model)
        self.mpc.settings.supress_ipopt_output()
        setup_mpc = {
            'n_horizon': self.horizon,
            't_step': 1,
            'n_robust': self.robust_horizon,
            'store_full_solution': True,
        }
        self.mpc.set_param(**setup_mpc)

        
        rS = 5
        f_factor = 1.3
        ecm_factor = 2
        ecm_cost = 0
        mterm = 20 * (0 - ((rS - self.x_1_a) / rS * ((1 - self.x_2_f) + self.x_2_f / f_factor) * ((1 - self.x_3_e) + self.x_3_e / ecm_factor) * self.c_1_target)) + \
            15 * (((rS - self.x_1_a) / rS * ((1 - self.x_2_f) + self.x_2_f / f_factor) * ((1 - self.x_3_e) + self.x_3_e / ecm_factor) * self.c_2_threat) - 1) + \
            0.5 * (self.x_3_e)
        lterm = 20 * (0 - ((rS - self.x_1_a) / rS * ((1 - self.x_2_f) + self.x_2_f / f_factor) * ((1 - self.x_3_e) + self.x_3_e / ecm_factor) * self.c_1_target)) + \
            15 * (((rS - self.x_1_a) / rS * ((1 - self.x_2_f) + self.x_2_f / f_factor) * ((1 - self.x_3_e) + self.x_3_e / ecm_factor) * self.c_2_threat) - 1) + \
            0.5 * (self.x_3_e + self.u_1_a**2 + self.u_2_f**2) 
        self.mpc.set_rterm(
            u_1_a=0,
            u_2_f=0,
            u_3_e=0
        )
        self.mpc.set_objective(mterm=mterm, lterm=lterm)

        
        self.mpc.bounds['lower','_x', 'x_1_a'] = 0
        self.mpc.bounds['lower','_x', 'x_2_f'] = 0
        self.mpc.bounds['lower','_x', 'x_3_e'] = 0

        self.mpc.bounds['upper','_x', 'x_1_a'] = 5
        self.mpc.bounds['upper','_x', 'x_2_f'] = 1
        self.mpc.bounds['upper','_x', 'x_3_e'] = 1

        self.mpc.bounds['lower','_u', 'u_1_a'] = -2
        self.mpc.bounds['lower','_u', 'u_2_f'] = -0.5
        self.mpc.bounds['lower','_u', 'u_3_e'] = 0

        self.mpc.bounds['upper','_u', 'u_1_a'] = 2
        self.mpc.bounds['upper','_u', 'u_2_f'] = 0.5
        self.mpc.bounds['upper','_u', 'u_3_e'] = 1

        if(self.n == -1):
            target_values = np.array([0,1])
            threat_values = np.array([0,1])
        else:   
            
            target_values = np.array(range(self.n))
            threat_values = np.array([0])

        self.mpc.set_uncertainty_values(
            c_1_target = target_values,
            c_2_threat = threat_values
        )
        
        self.mpc.setup()
        
        self.mpc.x0 = np.array([0,0,0]).reshape(-1,1)
       
        #self.mpc.x0 = x0
        endtime = time.time()
        print("set up time: " + str(endtime - starttime),file=self.log_file)

    def setupSMPCN(self):
        starttime = time.time()
        self.mpc = do_mpc.controller.MPC(self.model)
        self.mpc.settings.supress_ipopt_output()
        setup_mpc = {
            'n_horizon': self.horizon,
            't_step': 1,
            'n_robust': self.robust_horizon,
            'store_full_solution': True,
        }
        self.mpc.set_param(**setup_mpc)

        coeff = []
        for i in range(self.n):
            coeff.append(random.random())
        mterm = 0
        lterm = 0
        for i in range(self.n):
            mterm += coeff[i] * self.x[i] * (self.c_1_target + self.c_1_target)
            lterm += coeff[i] * self.x[i] * (self.c_1_target + self.c_1_target)
            
    
        self.mpc.set_objective(mterm=mterm, lterm=lterm)

        for i in range(self.n):
            xvarname = 'x_' + str(i+1)
            uvarname = 'u_' + str(i+1)
            self.mpc.bounds['lower','_x',xvarname] = 0
            self.mpc.bounds['upper','_x',xvarname] = 2
            self.mpc.bounds['lower','_u',uvarname] = -0.5
            self.mpc.bounds['upper','_u',uvarname] = 0.5

        target_values = np.array([0,1])
        threat_values = np.array([0,1])

        self.mpc.set_uncertainty_values(
            c_1_target = target_values,
            c_2_threat = threat_values
        )
        
        self.mpc.setup()
       
        #self.mpc.x0 = x0
        endtime = time.time()
        print("set up time: " + str(endtime - starttime),file=self.log_file)
    
    def setupSim(self):
        self.simulator = do_mpc.simulator.Simulator(self.model)
        # Instead of supplying a dict with the splat operator (**), as with the optimizer.set_param(),
        # we can also use keywords (and call the method multiple times, if necessary):
        self.simulator.set_param(t_step = 1)
        p_template = self.simulator.get_p_template()
        def p_fun(t_now):
            if(random.random() > 0.5):
                p_template['c_1_target'] = 1
            else:
                p_template['c_1_target'] = 0
            if(random.random() > 0.5):
                p_template['c_2_threat'] = 1
            else:
                p_template['c_2_threat'] = 0
            return p_template
        self.simulator.set_p_fun(p_fun)
        self.simulator.setup()
            
if __name__ == "__main__":
    exp = "-1"
    if(exp == "horizon"):
        horizon_range = 11
        for horizon in range(1,horizon_range):
            for robust_horizon in range(1,min(horizon+1,5)):
                print(horizon, robust_horizon)
                log_file = './Results/time/' + str(horizon) + "-" + str(robust_horizon) + ".log"
                with open(log_file, 'w') as f:
                    mpc_controller = MPCController(horizon, robust_horizon, -1, f)
                    x0 = np.array([0,0,0]).reshape(-1,1)
                    mpc_controller.simulator.x0 = x0
                    mpc_controller.mpc.x0 = x0
                    mpc_controller.mpc.set_initial_guess()
                    for t in range(20):
                        starttime = time.time()
                        u0 = mpc_controller.mpc.make_step(x0)
                        endtime = time.time()
                        print("planning time: " + str(endtime - starttime),file=f)
                        mpc_controller.simulator.make_step(u0)

    if(exp == "u"):
        for i in range(10):
            print(i)
            log_file = './Results/time/' + "u-" + str(i+1) + ".log"
            with open(log_file, 'w') as f:
                mpc_controller = MPCController(5, 3, i+1, f)
                x0 = np.array([0]*(i+1)).reshape(-1,1)
                mpc_controller.simulator.x0 = x0
                mpc_controller.mpc.x0 = x0
                mpc_controller.mpc.set_initial_guess()
                for t in range(20):
                    starttime = time.time()
                    u0 = mpc_controller.mpc.make_step(x0)
                    endtime = time.time()
                    print("planning time: " + str(endtime - starttime),file=f)
                    mpc_controller.simulator.make_step(u0)

    if(exp == "cv"):
        for i in range(3):
            for j in range(1,10):
                log_file = './Results/time/' + "cv-" + str(i+1) + "-" + str(j+1) + ".log"
                with open(log_file, 'w') as f:
                    mpc_controller = MPCController(5, i+1, j+1, f)
                    x0 = np.array([0,0,0]).reshape(-1,1)
                    mpc_controller.simulator.x0 = x0
                    mpc_controller.mpc.x0 = x0
                    mpc_controller.mpc.set_initial_guess()
                    for t in range(20):
                        starttime = time.time()
                        u0 = mpc_controller.mpc.make_step(x0)
                        endtime = time.time()
                        print("planning time: " + str(endtime - starttime),file=f)
                        mpc_controller.simulator.make_step(u0)
    
    if(exp == "1"):
        for i in range(20):
            log_file = './Results/time/' + "time-" + str(i) + ".log"
            with open(log_file, 'w') as f:
                mpc_controller = MPCController(5, 3, -1, f)
                x0 = np.array([0,0,0]).reshape(-1,1)
                mpc_controller.simulator.x0 = x0
                mpc_controller.mpc.x0 = x0
                mpc_controller.mpc.set_initial_guess()
                for t in range(20):
                    starttime = time.time()
                    u0 = mpc_controller.mpc.make_step(x0)
                    endtime = time.time()
                    print("planning time: " + str(endtime - starttime),file=f)
                    mpc_controller.simulator.make_step(u0)

    dir = "./Results/time/"
    time_list = []
    
    for i in range(20):
        filename = dir + "time-" + str(i) + ".log"
        plantime = 0
        setup_time = 0
        num = 0
        #print(filename)
        with open(filename) as f:
            res = f.readlines()
            for line in res:
                if(line.find("set up time:") != -1):
                    setup_time = float(line.strip().split(" ")[3])
                if(line.find("planning time:") != -1):
                    plantime += float(line.strip().split(" ")[2])
                    num += 1 
            plantime = plantime / num + setup_time
            time_list.append(plantime)

    print(np.mean(time_list))
    print(np.std(time_list))
    print(max(time_list))