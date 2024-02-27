import do_mpc

import numpy as np
import pandas as pd
from casadi import *
import time

class MPCController:
    def __init__(self, horizon, plan_type="smpc", model_type="lat", target_revenue = 20, threat_revenue = 15):
        self.horizon = horizon
        self.plan_type = plan_type
        self.model_type = model_type
        self.weights_list = [0.33, 0.33, 0.33]
        self.lbounds_list = [0, 0, 0]
        self.ubounds_list = [5, 1, 1]
        self.reference_list = [1, 0, 0]
        self.target_pred_list = [0] * (horizon + 1)
        self.threat_pred_list = [0] * (horizon + 1)

        self.target_revenue = target_revenue
        self.threat_revenue = threat_revenue
        
        self.defineModel()

        if(self.plan_type == "mpc"):
            self.setupMPC(self.weights_list, self.reference_list, self.lbounds_list, self.ubounds_list)
        

        
    def defineModel(self):
        self.model = do_mpc.model.Model('discrete')
        if(self.model_type == "nolat"):
            # x, which is performance indicators
            '''
            self.x_1_probtar = self.model.set_variable(var_type='_x', var_name='x_1_probtar', shape=(1,1))
            self.x_2_probthr = self.model.set_variable(var_type='_x', var_name='x_2_probthr', shape=(1,1))
            self.x_3_cost = self.model.set_variable(var_type='_x', var_name='x_3_cost', shape=(1,1))
            '''
            self.x_1_a = self.model.set_variable(var_type='_x', var_name='x_1_a', shape=(1,1))
            self.x_2_f = self.model.set_variable(var_type='_x', var_name='x_2_f', shape=(1,1))
            self.x_3_e = self.model.set_variable(var_type='_x', var_name='x_3_e', shape=(1,1))
            
        else:
            self.x_1_a = self.model.set_variable(var_type='_x', var_name='x_1_a', shape=(1,1))
            self.x_2_f = self.model.set_variable(var_type='_x', var_name='x_2_f', shape=(1,1))
            self.x_3_e = self.model.set_variable(var_type='_x', var_name='x_3_e', shape=(1,1))
            self.x_3_e_1 = self.model.set_variable(var_type='_x', var_name='x_3_e_1', shape=(1,1))
        
        if(self.model_type == "nolat"):
            # u, which is control parameters
            self.u_1_a = self.model.set_variable(var_type='_u', var_name='u_1_a')
            self.u_2_f = self.model.set_variable(var_type='_u', var_name='u_2_f')
            self.u_3_e = self.model.set_variable(var_type='_u', var_name='u_3_e')
        else:
            self.u_1_a = self.model.set_variable(var_type='_u', var_name='u_1_a')
            self.u_2_f = self.model.set_variable(var_type='_u', var_name='u_2_f')
            self.u_3_e = self.model.set_variable(var_type='_u', var_name='u_3_e')

        if(self.plan_type == "smpc"):
            # time varying parameter, which is context variables
            self.c_1_target = self.model.set_variable('_p', 'c_1_target') 
            self.c_2_threat = self.model.set_variable('_p', 'c_2_threat')
        else:
            # time varying parameter, which is context variables
            self.c_1_target = self.model.set_variable('_tvp', 'c_1_target') 
            self.c_2_threat = self.model.set_variable('_tvp', 'c_2_threat')
        
        rS = 5
        f_factor = 1.3
        ecm_factor = 2
        ecm_cost = 0.1
        if(self.model_type == "nolat"):
            '''
            x_1_next = (rS - self.u_1_a) / rS * ((1 - self.u_2_f) + self.u_2_f / f_factor) * ((1 - self.u_3_e) + self.u_3_e / ecm_factor) * self.c_1_target
            x_2_next = (rS - self.u_1_a) / rS * ((1 - self.u_2_f) + self.u_2_f / f_factor) * ((1 - self.u_3_e) + self.u_3_e / ecm_factor) * self.c_2_threat
            x_3_next = ecm_cost * self.u_3_e
            self.model.set_rhs('x_1_probtar', x_1_next)
            self.model.set_rhs('x_2_probthr', x_2_next)
            self.model.set_rhs('x_3_cost', x_3_next)
            '''
            x_1_next = self.u_1_a
            x_2_next = self.u_2_f
            x_3_next = self.u_3_e
            self.model.set_rhs('x_1_a', x_1_next)
            self.model.set_rhs('x_2_f', x_2_next)
            self.model.set_rhs('x_3_e', x_3_next)
            
        else:
            x_1_next = self.x_1_a + self.u_1_a
            x_2_next = self.x_2_f + self.u_2_f
            x_3_next = self.x_3_e_1
            x_3_1_next = self.u_3_e
            
            #x_3_next = self.u_3_e
            
            self.model.set_rhs('x_1_a', x_1_next)
            self.model.set_rhs('x_2_f', x_2_next)
            self.model.set_rhs('x_3_e', x_3_next)
            self.model.set_rhs('x_3_e_1', x_3_1_next)
        
        self.model.setup()

    def setupMPC(self, weights_list, reference_list, lbounds_list, ubounds_list):
        starttime = time.time()
        self.mpc = do_mpc.controller.MPC(self.model)
        self.mpc.settings.supress_ipopt_output()
        setup_mpc = {
            'n_horizon': self.horizon,
            't_step': 1,
            'n_robust': 1,
            'store_full_solution': True,
        }
        self.mpc.set_param(**setup_mpc)
        
        tar_scale = self.target_revenue
        thr_scale = self.threat_revenue
        if(self.model_type == "nolat"):
            # define objective function
            '''
            tar_scale = 10
            thr_scale = 5
            cost_scale = 1
            mterm = (reference_list[0] - self.x_1_probtar) * tar_scale + (self.x_2_probthr - reference_list[1]) * thr_scale + (self.x_3_cost - reference_list[2]) * cost_scale
            lterm = (reference_list[0] - self.x_1_probtar) * tar_scale + (self.x_2_probthr - reference_list[1]) * thr_scale + (self.x_3_cost - reference_list[2]) * cost_scale
            self.mpc.set_rterm(
                u_1_a=0,
                u_2_f=0,
                u_3_e=0
            )
            '''
            rS = 5
            f_factor = 1.3
            ecm_factor = 2
            ecm_cost = 0
            mterm = tar_scale * (reference_list[0] - ((rS - self.x_1_a) / rS * ((1 - self.x_2_f) + self.x_2_f / f_factor) * ((1 - self.x_3_e) + self.x_3_e / ecm_factor) * self.c_1_target)) + \
                thr_scale * (((rS - self.x_1_a) / rS * ((1 - self.x_2_f) + self.x_2_f / f_factor) * ((1 - self.x_3_e) + self.x_3_e / ecm_factor) * self.c_2_threat) - reference_list[1]) + \
                0.5 * (self.x_3_e - reference_list[2])
            lterm = tar_scale * (reference_list[0] - ((rS - self.x_1_a) / rS * ((1 - self.x_2_f) + self.x_2_f / f_factor) * ((1 - self.x_3_e) + self.x_3_e / ecm_factor) * self.c_1_target)) + \
                thr_scale * (((rS - self.x_1_a) / rS * ((1 - self.x_2_f) + self.x_2_f / f_factor) * ((1 - self.x_3_e) + self.x_3_e / ecm_factor) * self.c_2_threat) - reference_list[1]) + \
                0.5 * (self.x_3_e * ecm_cost - reference_list[2])
            self.mpc.set_rterm(
                u_1_a=0,
                u_2_f=0,
                u_3_e=0
            )
            
        else:
            rS = 5
            f_factor = 1.3
            ecm_factor = 2
            ecm_cost = 0
            mterm = tar_scale * (reference_list[0] - ((rS - self.x_1_a) / rS * ((1 - self.x_2_f) + self.x_2_f / f_factor) * ((1 - self.x_3_e) + self.x_3_e / ecm_factor) * self.c_1_target)) + \
                thr_scale * (((rS - self.x_1_a) / rS * ((1 - self.x_2_f) + self.x_2_f / f_factor) * ((1 - self.x_3_e) + self.x_3_e / ecm_factor) * self.c_2_threat) - reference_list[1]) + \
                0.5 * (self.x_3_e - reference_list[2])
            lterm = tar_scale * (reference_list[0] - ((rS - self.x_1_a) / rS * ((1 - self.x_2_f) + self.x_2_f / f_factor) * ((1 - self.x_3_e) + self.x_3_e / ecm_factor) * self.c_1_target)) + \
                thr_scale * (((rS - self.x_1_a) / rS * ((1 - self.x_2_f) + self.x_2_f / f_factor) * ((1 - self.x_3_e) + self.x_3_e / ecm_factor) * self.c_2_threat) - reference_list[1]) + \
                0.5 * (self.x_3_e - reference_list[2] + self.u_1_a**2 + self.u_2_f**2)
            self.mpc.set_rterm(
                u_1_a=0,
                u_2_f=0,
                u_3_e=0
            )
        self.mpc.set_objective(mterm=mterm, lterm=lterm)

        

        if(self.model_type == "nolat"):
            # define bounds
            self.mpc.bounds['lower','_u', 'u_1_a'] = lbounds_list[0]
            self.mpc.bounds['lower','_u', 'u_2_f'] = lbounds_list[1]
            self.mpc.bounds['lower','_u', 'u_3_e'] = lbounds_list[2]

            self.mpc.bounds['upper','_u', 'u_1_a'] = ubounds_list[0]
            self.mpc.bounds['upper','_u', 'u_2_f'] = ubounds_list[1]
            self.mpc.bounds['upper','_u', 'u_3_e'] = ubounds_list[2]
        else:
            self.mpc.bounds['lower','_x', 'x_1_a'] = lbounds_list[0]
            self.mpc.bounds['lower','_x', 'x_2_f'] = lbounds_list[1]
            self.mpc.bounds['lower','_x', 'x_3_e'] = lbounds_list[2]

            self.mpc.bounds['upper','_x', 'x_1_a'] = ubounds_list[0]
            self.mpc.bounds['upper','_x', 'x_2_f'] = ubounds_list[1]
            self.mpc.bounds['upper','_x', 'x_3_e'] = ubounds_list[2]

            self.mpc.bounds['lower','_u', 'u_1_a'] = -2
            self.mpc.bounds['lower','_u', 'u_2_f'] = -0.5
            self.mpc.bounds['lower','_u', 'u_3_e'] = 0

            self.mpc.bounds['upper','_u', 'u_1_a'] = 2
            self.mpc.bounds['upper','_u', 'u_2_f'] = 0.5
            self.mpc.bounds['upper','_u', 'u_3_e'] = 1

        
        tvp_prediction = self.mpc.get_tvp_template()
        def tvp_fun(t_now):
            pvalue_list = []
            #pvalue_list.append([req_history[int(t_now)],res_history[int(t_now)]])
            for t in range(self.horizon):
                pvalue_list.append([self.target_pred_list[int(t)], self.threat_pred_list[int(t)]])
            pvalue_list.append([self.target_pred_list[-1], self.threat_pred_list[-1]])    
            tvp_prediction['_tvp'] = pvalue_list
            return tvp_prediction

        self.mpc.set_tvp_fun(tvp_fun)
        
        self.mpc.setup()
        
        if(self.model_type == "nolat"):
            self.mpc.x0 = np.array([0,0,0]).reshape(-1,1)
        else:
            #self.mpc.x0 = np.array([0,0,0]).reshape(-1,1)
            self.mpc.x0 = np.array([0,0,0,0]).reshape(-1,1)
        self.mpc.set_initial_guess()
        endtime = time.time()
        print("set up time: " + str(endtime - starttime))

    def setupSMPC(self, weights_list, reference_list, lbounds_list, ubounds_list, cv_probs, x0):
        starttime = time.time()
        self.mpc = do_mpc.controller.MPC(self.model)
        #self.mpc.settings.supress_ipopt_output()
        setup_mpc = {
            'n_horizon': 5,
            't_step': 1,
            'n_robust': 1,
            'store_full_solution': True,
        }
        self.mpc.set_param(**setup_mpc)

        tar_scale = self.target_revenue
        thr_scale = self.threat_revenue
        if(self.model_type == "nolat"):
            # define objective function
            '''
            tar_scale = 10
            thr_scale = 5
            cost_scale = 0.5
            mterm = (reference_list[0] - self.x_1_probtar) * tar_scale + (self.x_2_probthr - reference_list[1]) * thr_scale + (self.x_3_cost - reference_list[2]) * cost_scale
            lterm = (reference_list[0] - self.x_1_probtar) * tar_scale + (self.x_2_probthr - reference_list[1]) * thr_scale + (self.x_3_cost - reference_list[2]) * cost_scale
            self.mpc.set_rterm(
                u_1_a=0,
                u_2_f=0,
                u_3_e=0
            )
            '''
            rS = 5
            f_factor = 1.3
            ecm_factor = 2
            ecm_cost = 0
            mterm = tar_scale * (0 - ((rS - self.x_1_a) / rS * ((1 - self.x_2_f) + self.x_2_f / f_factor) * ((1 - self.x_3_e) + self.x_3_e / ecm_factor) * self.c_1_target)) + \
                thr_scale * (((rS - self.x_1_a) / rS * ((1 - self.x_2_f) + self.x_2_f / f_factor) * ((1 - self.x_3_e) + self.x_3_e / ecm_factor) * self.c_2_threat) - reference_list[1]) + \
                0.5 * (self.x_3_e)
            lterm = tar_scale * (0 - ((rS - self.x_1_a) / rS * ((1 - self.x_2_f) + self.x_2_f / f_factor) * ((1 - self.x_3_e) + self.x_3_e / ecm_factor) * self.c_1_target)) + \
                thr_scale * (((rS - self.x_1_a) / rS * ((1 - self.x_2_f) + self.x_2_f / f_factor) * ((1 - self.x_3_e) + self.x_3_e / ecm_factor) * self.c_2_threat) - reference_list[1]) + \
                0.5 * (self.x_3_e + self.u_1_a**2 / 2 + self.u_2_f**2) 
            self.mpc.set_rterm(
                u_1_a=0,
                u_2_f=0,
                u_3_e=0
            )
            
        else:
            rS = 5
            f_factor = 1.3
            ecm_factor = 2
            ecm_cost = 0
            mterm = tar_scale * (0 - ((rS - self.x_1_a) / rS * ((1 - self.x_2_f) + self.x_2_f / f_factor) * ((1 - self.x_3_e) + self.x_3_e / ecm_factor) * self.c_1_target)) + \
                thr_scale * (((rS - self.x_1_a) / rS * ((1 - self.x_2_f) + self.x_2_f / f_factor) * ((1 - self.x_3_e) + self.x_3_e / ecm_factor) * self.c_2_threat) - reference_list[1]) + \
                0.5 * (self.x_3_e)
            lterm = tar_scale * (0 - ((rS - self.x_1_a) / rS * ((1 - self.x_2_f) + self.x_2_f / f_factor) * ((1 - self.x_3_e) + self.x_3_e / ecm_factor) * self.c_1_target)) + \
                thr_scale * (((rS - self.x_1_a) / rS * ((1 - self.x_2_f) + self.x_2_f / f_factor) * ((1 - self.x_3_e) + self.x_3_e / ecm_factor) * self.c_2_threat) - reference_list[1]) + \
                0.5 * (self.x_3_e + self.u_1_a**2 / 2 + self.u_2_f**2) 
            self.mpc.set_rterm(
                u_1_a=0,
                u_2_f=0,
                u_3_e=0
            )
        self.mpc.set_objective(mterm=mterm, lterm=lterm)

        if(self.model_type == "nolat"):
            # define bounds
            self.mpc.bounds['lower','_u', 'u_1_a'] = lbounds_list[0]
            self.mpc.bounds['lower','_u', 'u_2_f'] = lbounds_list[1]
            self.mpc.bounds['lower','_u', 'u_3_e'] = lbounds_list[2]

            self.mpc.bounds['upper','_u', 'u_1_a'] = ubounds_list[0]
            self.mpc.bounds['upper','_u', 'u_2_f'] = ubounds_list[1]
            self.mpc.bounds['upper','_u', 'u_3_e'] = ubounds_list[2]
        else:
            self.mpc.bounds['lower','_x', 'x_1_a'] = lbounds_list[0]
            self.mpc.bounds['lower','_x', 'x_2_f'] = lbounds_list[1]
            self.mpc.bounds['lower','_x', 'x_3_e'] = lbounds_list[2]

            self.mpc.bounds['upper','_x', 'x_1_a'] = ubounds_list[0]
            self.mpc.bounds['upper','_x', 'x_2_f'] = ubounds_list[1]
            self.mpc.bounds['upper','_x', 'x_3_e'] = ubounds_list[2]

            self.mpc.bounds['lower','_u', 'u_1_a'] = -2
            self.mpc.bounds['lower','_u', 'u_2_f'] = -0.5
            self.mpc.bounds['lower','_u', 'u_3_e'] = 0

            self.mpc.bounds['upper','_u', 'u_1_a'] = 2
            self.mpc.bounds['upper','_u', 'u_2_f'] = 0.5
            self.mpc.bounds['upper','_u', 'u_3_e'] = 1

        target_values = np.array([0,1])
        threat_values = np.array([0,1])

        self.mpc.set_uncertainty_values(
            c_1_target = target_values,
            c_2_threat = threat_values
        )
        
        #self.mpc.setup()
        self.mpc._prepare_nlp_s(cv_probs)
        self.mpc.create_nlp()

        '''
        if(self.model_type == "nolat"):
            self.mpc.x0 = np.array([0,0,0]).reshape(-1,1)
        else:
            self.mpc.x0 = np.array([0,0,0,0]).reshape(-1,1)
        '''
        self.mpc.x0 = x0
        self.mpc.set_initial_guess()
        endtime = time.time()
        print("set up time: " + str(endtime - starttime))

    # target_prob_list: prob(target=1)
    # target_pred_list: target=1
    def adapt(self, x0, target_prob_list, threat_prob_list): 
        self.target_pred_list = []
        self.threat_pred_list = []
        for i in range(len(target_prob_list)):
            if(target_prob_list[i] > 0.5):  
                self.target_pred_list.append(target_prob_list[i])
            else:
                self.target_pred_list.append(target_prob_list[i])
            if(threat_prob_list[i] > 0.5):  
                self.threat_pred_list.append(threat_prob_list[i])
            else:
                self.threat_pred_list.append(threat_prob_list[i])
        if(self.plan_type == "smpc"):
            target_prob_dist = []
            threat_prob_dist = []
            for i in range(self.horizon):
                target_prob_dist.append([1 - target_prob_list[i], target_prob_list[i]])
                threat_prob_dist.append([1 - threat_prob_list[i], threat_prob_list[i]])
            cv_probs = [target_prob_dist, threat_prob_dist]
            
            self.setupSMPC(self.weights_list, self.reference_list, self.lbounds_list, self.ubounds_list, cv_probs, x0)
        starttime = time.time()
        u0 = self.mpc.make_step(x0)
        endtime = time.time()
        print("planning time: " + str(endtime - starttime))
        return u0
    
    
if __name__ == "__main__":
    mpc_controller = MPCController(5, "smpc", "lat")
    