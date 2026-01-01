class sigmoid:
    
    def __init__(self, Rf, R1, thresh):
        self.Rf = Rf
        self.R1 = R1
        self.thresh = thresh

    def getGain(self):
        return (self.Rf / self.R1)
    
    def sig(self, z, Vccm, Vccp):
        result = self.getGain() * (z - self.thresh)
        result = Vccm if result < Vccm else (Vccp if result > Vccp else result)
        return result


