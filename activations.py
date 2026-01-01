class OpAmpClippedLinear:
    """
    Models an Op-Amp Difference Amplifier with 'Leaky' Saturation.
    """
    def __init__(self, Rf, R1, initial_thresh):
        self.Rf = float(Rf)
        self.R1 = float(R1)
        self.thresh = float(initial_thresh)

    def get_gain(self):
        return self.Rf / self.R1

    def forward(self, signal_sum, vccm, vccp):
        """
        Calculates: Clamp( Gain * (Signal - Threshold), Vccm, Vccp )
        """
        raw_output = self.get_gain() * (signal_sum - self.thresh)
        
        # Hard Saturation (Clamping)
        return max(vccm, min(vccp, raw_output))

    def derivative(self, output_voltage, vccm, vccp):
        """
        Returns the slope.
        - If Active: Returns Gain.
        - If Saturated: Returns Gain * 0.01 (Leaky Gradient).
        """
        swing = vccp - vccm
        epsilon = swing * 0.01 

        # Linear Region
        if (vccm + epsilon) < output_voltage < (vccp - epsilon):
            return self.get_gain()
        
        # Saturated Region (The Leak)
        # We return a tiny gradient so the weights can still nudge the 
        # neuron out of saturation if it's stuck.
        return self.get_gain() * 0.01
