from DartSim import Dart
from ManagingSystem import ManagingSystem
from Environment import Environment
from Predictor import Predictor

import sys
import numpy as np

# base controller parameter
horizon = 5
pred_type = "fuse"
plan_type = "smpc"
model_type = "lat"

# system init 
init_a = 0
init_f = 0
init_ecm = 0

class ExpInstance:
    def __init__(self, pred_type, plan_type, model_type, horizon, env, log_file, target_revenue, threat_revenue) -> None:
        self.pred_type = pred_type
        self.plan_type = plan_type
        self.model_type = model_type
        self.dart = Dart(init_a, init_f, init_ecm, model_type, target_revenue, threat_revenue)
        self.managing_system = ManagingSystem(horizon, env, plan_type, model_type, log_file, target_revenue, threat_revenue)
        self.revenue_list = []
        self.principal_list = []
        self.interest_list = []
        self.log_file = log_file

    def printType(self):
        print(str(self.pred_type) + " " + str(self.plan_type))

# global exp setting
arg_len = len(sys.argv)
exp_type = "test"
if(arg_len > 1):
    exp_type = sys.argv[1]

env_type = "random"
if(arg_len > 2):
    env_type = sys.argv[2]

env_case = 0
if(arg_len > 3):
    env_case = sys.argv[3]

# environment init
env = Environment(env_type, env_case)
env.generateEnv()
time_limit = env.length

predictor = Predictor(horizon, 0.15)

if(exp_type == "test"):
    log_file = './Results/DART-' + str(exp_type) + ".log"

    env = Environment("fix")
    time_limit = env.length
    # start exp
    exp_instance = ExpInstance(pred_type, plan_type, model_type, horizon, env, log_file)
    #exp_instance = ExpInstance(pred_type, "mpc", "nolat", horizon, env, log_file)

    with open(log_file, 'w') as f:
        for t in range(time_limit):
            print("\ntime: " + str(t), file=f)
            target = env.target[t]
            threat = env.threat[t]
            print("Environment: target: " + str(target) + " " + "threat: " + str(threat) ,file=f)
            
            target_list = env.target[t:t+horizon]
            threat_list = env.threat[t:t+horizon]
            target_prob_list, threat_prob_list = predictor.getEnvPred2(target_list, threat_list)
            predictor.storePrediction(target_prob_list, threat_prob_list)
            fused_target_prob_list, fused_threat_prob_list = predictor.DSPredictionFusion()
            print("target prediction: " + str(fused_target_prob_list[0]), file=f)
            print("threat prediction: " + str(fused_threat_prob_list[0]), file=f)

            if(exp_instance.model_type == "nolat"):
                x0 = np.array([exp_instance.dart.altitude, exp_instance.dart.formation, exp_instance.dart.ECM]).reshape(-1,1)
            else:
                x0 = np.array([exp_instance.dart.altitude, exp_instance.dart.formation, exp_instance.dart.ECM, exp_instance.dart.ECM_lat]).reshape(-1,1)
            u0 = exp_instance.managing_system.step(t, x0, fused_target_prob_list, fused_threat_prob_list)
            print("control parameter:" + str(u0[0][0]) + " " + str(u0[1][0]) + " " + str(u0[2][0]),file=f)
            exp_instance.dart.adjust(u0)

            exp_instance.dart.showState(f)

            revenue = exp_instance.dart.getReward(target, threat)
            principal = exp_instance.dart.getPrincipal(u0)
            interest = exp_instance.dart.getInterest()
            print("Revenue: " + str(revenue) + " Principal: " + str(principal) + " Interest: " + str(interest),file=f)
            exp_instance.revenue_list.append(revenue)
            exp_instance.principal_list.append(principal)
            exp_instance.interest_list.append(interest)

        total_revenue = 0
        total_principal = 0
        total_interest = 0
        for r in exp_instance.revenue_list:
            total_revenue += r
        for p in exp_instance.principal_list:
            total_principal += p
        for i in exp_instance.interest_list:
            total_interest += i

        print("total revenue = " + str(total_revenue) + ", total principal = " + str(total_principal) + ", total interest = " + str(total_interest),file=f)
        
elif(exp_type == "timing"):
    log_file = './Results/DART-' + str(exp_type) + "-" + str(env_case) + ".log"

    exp_instance_1 = ExpInstance(pred_type, plan_type, "lat", horizon, env, log_file)
    exp_instance_2 = ExpInstance(pred_type, plan_type, "nolat", horizon, env, log_file)
    exp_instance_list = [exp_instance_1, exp_instance_2]

    with open(log_file, 'w') as f:
        for t in range(time_limit):
            print("\ntime: " + str(t), file=f)
            target = env.target[t]
            threat = env.threat[t]
            print("Environment: target: " + str(target) + " " + "threat: " + str(threat) ,file=f)

            target_list = env.target[t:t+horizon]
            threat_list = env.threat[t:t+horizon]
            target_prob_list, threat_prob_list = predictor.getEnvPred2(target_list, threat_list)
            predictor.storePrediction(target_prob_list, threat_prob_list)
            fused_target_prob_list, fused_threat_prob_list = predictor.DSPredictionFusion()
            print("target prediction: " + str(fused_target_prob_list[0]), file=f)
            print("threat prediction: " + str(fused_threat_prob_list[0]), file=f)

            for exp_instance in exp_instance_list:
                print("###EXP###" + exp_instance.model_type, file=f)
                if(exp_instance.model_type == "nolat"):
                    x0 = np.array([exp_instance.dart.altitude, exp_instance.dart.formation, exp_instance.dart.ECM]).reshape(-1,1)
                else:
                    x0 = np.array([exp_instance.dart.altitude, exp_instance.dart.formation, exp_instance.dart.ECM]).reshape(-1,1)
                    x0 = np.array([exp_instance.dart.altitude, exp_instance.dart.formation, exp_instance.dart.ECM, exp_instance.dart.ECM_lat]).reshape(-1,1)
                u0 = exp_instance.managing_system.step(t, x0, fused_target_prob_list, fused_threat_prob_list)
                print("control parameter:" + str(round(u0[0][0],2)) + " " + str(round(u0[1][0],2)) + " " + str(round(u0[2][0],2)),file=f)
                exp_instance.dart.adjust(u0)

                exp_instance.dart.showState(f)

                revenue = exp_instance.dart.getReward(target, threat)
                principal = exp_instance.dart.getPrincipal(u0)
                interest = exp_instance.dart.getInterest()
                print("Revenue: " + str(revenue) + " Principal: " + str(principal) + " Interest: " + str(interest),file=f)
                exp_instance.revenue_list.append(revenue)
                exp_instance.principal_list.append(principal)
                exp_instance.interest_list.append(interest)

        for exp_instance in exp_instance_list:
            total_revenue = 0
            total_principal = 0
            total_interest = 0
            for r in exp_instance.revenue_list:
                total_revenue += r
            for p in exp_instance.principal_list:
                total_principal += p
            for i in exp_instance.interest_list:
                total_interest += i

            print("total revenue = " + str(total_revenue) + ", total principal = " + str(total_principal) + ", total interest = " + str(total_interest),file=f)

elif(exp_type == "eff"):
    log_file = './Results/DART-' + str(exp_type) + "-" + str(env_case) + ".log"

    exp_instance_1 = ExpInstance(pred_type, plan_type, model_type, horizon, env, log_file)
    exp_instance_2 = ExpInstance(pred_type, "mpc", model_type, horizon, env, log_file)
    exp_instance_3 = ExpInstance(pred_type, plan_type, "nolat", horizon, env, log_file)
    exp_instance_4 = ExpInstance(pred_type, "mpc", "nolat", horizon, env, log_file)

    exp_instance_list = [exp_instance_1, exp_instance_2, exp_instance_3, exp_instance_4]

    with open(log_file, 'w') as f:
        for t in range(time_limit):
            print("\ntime: " + str(t), file=f)
            target = env.target[t]
            threat = env.threat[t]
            print("Environment: target: " + str(target) + " " + "threat: " + str(threat) ,file=f)

            target_list = env.target[t:t+horizon]
            threat_list = env.threat[t:t+horizon]
            target_prob_list, threat_prob_list = predictor.getEnvPred2(target_list, threat_list)
            predictor.storePrediction(target_prob_list, threat_prob_list)
            fused_target_prob_list, fused_threat_prob_list = predictor.DSPredictionFusion()
            print("target prediction: " + str(fused_target_prob_list[0]), file=f)
            print("threat prediction: " + str(fused_threat_prob_list[0]), file=f)

            for exp_instance in exp_instance_list:
                print("###EXP###" + exp_instance.plan_type, file=f)
                if(exp_instance.model_type == "nolat"):
                    x0 = np.array([exp_instance.dart.altitude, exp_instance.dart.formation, exp_instance.dart.ECM]).reshape(-1,1)
                else:
                    #x0 = np.array([exp_instance.dart.altitude, exp_instance.dart.formation, exp_instance.dart.ECM]).reshape(-1,1)
                    x0 = np.array([exp_instance.dart.altitude, exp_instance.dart.formation, exp_instance.dart.ECM, exp_instance.dart.ECM_lat]).reshape(-1,1)
                u0 = exp_instance.managing_system.step(t, x0, fused_target_prob_list, fused_threat_prob_list)
                print("control parameter:" + str(round(u0[0][0],2)) + " " + str(round(u0[1][0],2)) + " " + str(round(u0[2][0],2)),file=f)
                exp_instance.dart.adjust(u0)

                exp_instance.dart.showState(f)

                revenue = exp_instance.dart.getReward(target, threat)
                principal = exp_instance.dart.getPrincipal(u0)
                interest = exp_instance.dart.getInterest()
                print("Revenue: " + str(revenue) + " Principal: " + str(principal) + " Interest: " + str(interest),file=f)
                exp_instance.revenue_list.append(revenue)
                exp_instance.principal_list.append(principal)
                exp_instance.interest_list.append(interest)

        for exp_instance in exp_instance_list:
            total_revenue = 0
            total_principal = 0
            total_interest = 0
            for r in exp_instance.revenue_list:
                total_revenue += r
            for p in exp_instance.principal_list:
                total_principal += p
            for i in exp_instance.interest_list:
                total_interest += i

            print("total revenue = " + str(total_revenue) + ", total principal = " + str(total_principal) + ", total interest = " + str(total_interest),file=f)

elif(exp_type == "pred"):
    log_file = './Results/DART-' + str(exp_type) + "-" + str(env_case) + ".log"

    exp_instance_1 = ExpInstance("fuse", plan_type, model_type, horizon, env, log_file)
    exp_instance_2 = ExpInstance("latest", plan_type, model_type, horizon, env, log_file)

    exp_instance_list = [exp_instance_1, exp_instance_2]

    with open(log_file, 'w') as f:
        for t in range(time_limit):
            print("\ntime: " + str(t), file=f)
            target = env.target[t]
            threat = env.threat[t]
            print("Environment: target: " + str(target) + " " + "threat: " + str(threat) ,file=f)

            target_list = env.target[t:t+horizon]
            threat_list = env.threat[t:t+horizon]
            target_prob_list, threat_prob_list = predictor.getEnvPred2(target_list, threat_list)
            predictor.storePrediction(target_prob_list, threat_prob_list)
            fused_target_prob_list, fused_threat_prob_list = predictor.DSPredictionFusion()
            print("fused target prediction: " + str(fused_target_prob_list[0]), file=f)
            print("fused threat prediction: " + str(fused_threat_prob_list[0]), file=f)
            print("latest target prediction: " + str(target_prob_list[0]), file=f)
            print("latest threat prediction: " + str(threat_prob_list[0]), file=f)

            for exp_instance in exp_instance_list:
                print("###EXP###" + exp_instance.plan_type, file=f)               
                x0 = np.array([exp_instance.dart.altitude, exp_instance.dart.formation, exp_instance.dart.ECM, exp_instance.dart.ECM_lat]).reshape(-1,1)
                if(exp_instance.pred_type == "fuse"):
                    u0 = exp_instance.managing_system.step(t, x0, fused_target_prob_list, fused_threat_prob_list)
                else:
                    u0 = exp_instance.managing_system.step(t, x0, target_prob_list, threat_prob_list)
                print("control parameter:" + str(round(u0[0][0],2)) + " " + str(round(u0[1][0],2)) + " " + str(round(u0[2][0],2)),file=f)
                exp_instance.dart.adjust(u0)

                exp_instance.dart.showState(f)

                revenue = exp_instance.dart.getReward(target, threat)
                principal = exp_instance.dart.getPrincipal(u0)
                interest = exp_instance.dart.getInterest()
                print("Revenue: " + str(revenue) + " Principal: " + str(principal) + " Interest: " + str(interest),file=f)
                exp_instance.revenue_list.append(revenue)
                exp_instance.principal_list.append(principal)
                exp_instance.interest_list.append(interest)

        for exp_instance in exp_instance_list:
            total_revenue = 0
            total_principal = 0
            total_interest = 0
            for r in exp_instance.revenue_list:
                total_revenue += r
            for p in exp_instance.principal_list:
                total_principal += p
            for i in exp_instance.interest_list:
                total_interest += i

            print("total revenue = " + str(total_revenue) + ", total principal = " + str(total_principal) + ", total interest = " + str(total_interest),file=f)

elif(exp_type == "setting-u"):
    log_file = './Results/setting-u/DART-' + str(env_case) + ".log"
    with open(log_file, 'w') as f:
        for target_revenue in [5,10,15,20,25,30]:
            for threat_revenue in [5,10,15,20,25,30]:
                exp_instance_1 = ExpInstance(pred_type, "smpc", "lat", horizon, env, log_file, target_revenue, threat_revenue)
                exp_instance_2 = ExpInstance(pred_type, "mpc", "nolat", horizon, env, log_file, target_revenue, threat_revenue)
                exp_instance_list = [exp_instance_1, exp_instance_2]
        
                for t in range(time_limit):
                    #print("\ntime: " + str(t), file=f)
                    target = env.target[t]
                    threat = env.threat[t]
                    #print("Environment: target: " + str(target) + " " + "threat: " + str(threat) ,file=f)

                    target_list = env.target[t:t+horizon]
                    threat_list = env.threat[t:t+horizon]
                    target_prob_list, threat_prob_list = predictor.getEnvPred2(target_list, threat_list)
                    predictor.storePrediction(target_prob_list, threat_prob_list)
                    fused_target_prob_list, fused_threat_prob_list = predictor.DSPredictionFusion()
                    #print("target prediction: " + str(fused_target_prob_list[0]), file=f)
                    #print("threat prediction: " + str(fused_threat_prob_list[0]), file=f)

                    for exp_instance in exp_instance_list:
                        #print("###EXP###" + exp_instance.model_type, file=f)
                        if(exp_instance.model_type == "nolat"):
                            x0 = np.array([exp_instance.dart.altitude, exp_instance.dart.formation, exp_instance.dart.ECM]).reshape(-1,1)
                        else:
                            #x0 = np.array([exp_instance.dart.altitude, exp_instance.dart.formation, exp_instance.dart.ECM]).reshape(-1,1)
                            x0 = np.array([exp_instance.dart.altitude, exp_instance.dart.formation, exp_instance.dart.ECM, exp_instance.dart.ECM_lat]).reshape(-1,1)
                        u0 = exp_instance.managing_system.step(t, x0, fused_target_prob_list, fused_threat_prob_list)
                        #print("control parameter:" + str(round(u0[0][0],2)) + " " + str(round(u0[1][0],2)) + " " + str(round(u0[2][0],2)),file=f)
                        exp_instance.dart.adjust(u0)

                        #exp_instance.dart.showState(f)

                        revenue = exp_instance.dart.getReward(target, threat)
                        principal = exp_instance.dart.getPrincipal(u0)
                        interest = exp_instance.dart.getInterest()
                        #print("Revenue: " + str(revenue) + " Principal: " + str(principal) + " Interest: " + str(interest),file=f)
                        exp_instance.revenue_list.append(revenue)
                        exp_instance.principal_list.append(principal)
                        exp_instance.interest_list.append(interest)

                for exp_instance in exp_instance_list:
                    total_revenue = 0
                    total_principal = 0
                    total_interest = 0
                    for r in exp_instance.revenue_list:
                        total_revenue += r
                    for p in exp_instance.principal_list:
                        total_principal += p
                    for i in exp_instance.interest_list:
                        total_interest += i

                    print("total revenue = " + str(total_revenue) + ", total principal = " + str(total_principal) + ", total interest = " + str(total_interest)
                            + " " + str(target_revenue) + " " + str(threat_revenue) ,file=f)
