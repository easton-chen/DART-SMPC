from smpc import MPCController
from Predictor import Predictor

class ManagingSystem:
    def __init__(self, horizon, env, plan_type, model_type, log_file, target_revenue=20, threat_revenue=15):
        self.env = env
        self.horizon = horizon
        self.plan_type = plan_type
        self.model_type = model_type
        self.predictor = Predictor(horizon, 0.1)
        self.controller = MPCController(horizon, plan_type, model_type, target_revenue, threat_revenue)
        self.log_file = log_file
        
    def step(self, t, x0, fused_target_prob_list, fused_threat_prob_list):
        # 1. predict
        #target_pred_list = [[0.2,0.8]] * self.horizon
        #threat_pred_list = [[0.5,0.5]] * self.horizon
        #target_pred_list,threat_pred_list = self.getNaivePred(t)
        

        # 2. plan
        u0 = self.controller.adapt(x0, fused_target_prob_list, fused_threat_prob_list)
        return u0
    
    def getNaivePred(self,t):
        target_list = self.env.target[t:t+self.horizon]
        threat_list = self.env.threat[t:t+self.horizon]
        while(len(target_list) < self.horizon):
            target_list.append(target_list[-1])
            threat_list.append(threat_list[-1])
        target_pred_list = []
        threat_pred_list = []
        for i in range(self.horizon):
            target_pred_list.append([target_list[i],1-target_list[i]])
            threat_pred_list.append([threat_list[i],1-threat_list[i]])

        return target_pred_list,threat_pred_list
    
    