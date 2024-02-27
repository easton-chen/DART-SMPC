import random

class Environment:
    def __init__(self, type="random", case=0):
        # cv0: target, cv1: threat
        self.target_case = [] # target for all case
        self.threat_case = []

        self.target = []
        self.threat = []
        
        if(type == "random"):
            env_file = "random_env-50.txt"
            with open(env_file, 'r') as efile:
                env_lines = efile.readlines()
                length = int(len(env_lines) / 2)
                
                for i in range(length):
                    target_line = env_lines[i * 2]
                    threat_line = env_lines[i * 2 + 1]

                    target_line = target_line.split(" ")
                    threat_line = threat_line.split(" ")                 
                    target_line = [float(target) for target in target_line[:-1]]
                    threat_line = [float(threat) for threat in threat_line[:-1]]
                                 
                    self.target_case.append(target_line)
                    self.threat_case.append(threat_line)
                  
            
                self.target_gen_prob = self.target_case[int(case)]
                self.threat_gen_prob = self.threat_case[int(case)]
            self.length = len(self.target_gen_prob)
        
        if(type == "fix"):
            for i in range(10):
                self.target.append(0)
                self.threat.append(1)
            #self.target.append(0)
            #self.threat.append(0)
            for i in range(10):
                self.target.append(1)
                self.threat.append(0)
            self.length = len(self.target)
                
    # now we directly generate in env file, but still keep this way
    def generateEnv(self):
        for i in range(len(self.target_gen_prob)):
            if(random.random() < self.target_gen_prob[i]):
                self.target.append(1)
            else:
                self.target.append(0)
            if(random.random() < self.threat_gen_prob[i]):
                self.threat.append(1)
            else:
                self.threat.append(0)
 

        