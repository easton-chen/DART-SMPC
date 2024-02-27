import sys
class Dart:
    def __init__(self, init_a, init_f, init_ecm, model_type, target_revenue=20,threat_revenue=15):
        self.altitude = init_a
        self.formation = init_f
        self.ECM = init_ecm
        self.ECM_lat = init_ecm
        self.max_altitude = 10
        self.target_range = 5
        self.threat_range = 5
        self.model_type = model_type
        self.target_revenue = target_revenue
        self.threat_revenue = threat_revenue

    def adjust(self, u0):
        if(self.model_type == "nolat"):
            if(self.altitude > u0[0][0]):
                self.d_u_a = round(max(u0[0][0] - self.altitude, -2),2)
            else:
                self.d_u_a = round(min(u0[0][0] - self.altitude, 2),2)
            if(self.formation > u0[1][0]):
                self.d_u_f = round(max(u0[1][0] - self.formation, -0.5),2)
            else:
                self.d_u_f = round(min(u0[1][0] - self.formation, 0.5),2)
        else:
            self.d_u_a = round(u0[0][0],2)
            self.d_u_f = round(u0[1][0],2)
        
        self.altitude += self.d_u_a
        self.formation += self.d_u_f
        self.ECM = self.ECM_lat
        self.ECM_lat = round(u0[2][0])
        #self.ECM = round(u0[2][0])
        self.altitude = max(0, self.altitude)
        self.formation = max(0, self.formation)
        self.ECM = max(0, self.ECM)

    def showState(self, outfile=sys.stdout):
        print("Altitude: " + str(self.altitude) + " Formation: " + str(self.formation) + " ECM: " + str(self.ECM), file=outfile)

    def getReward(self, target, threat):
        rT = 5 # threat range
        rS = 5 # sensor target range
        f_factor = 1.3
        ecm_factor = 2
        detect_bonus = self.target_revenue
        destory_penalty = self.threat_revenue
        destory_prob = max(0,rT - self.altitude) / rT * ((1 - self.formation) + self.formation/f_factor) * ((1 - self.ECM) + self.ECM / ecm_factor)
        detect_prob = max(0,rS - self.altitude) / rS * ((1 - self.formation) + self.formation/f_factor) * ((1 - self.ECM) + self.ECM / ecm_factor)
        #ran = random.random()

        total_reward = 0
        total_reward += (target * detect_prob * detect_bonus - threat * destory_prob * destory_penalty)
        
        return total_reward
    
    def getPrincipal(self, u0):
        cost_adapt = 0
        cost_adapt += abs(self.d_u_a) * 0.25 + abs(self.d_u_f) * 0.5 
        return cost_adapt
    
    def getInterest(self):
        cost_ecm = 0.5
        return self.ECM * cost_ecm