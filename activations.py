class OpAmpClippedLinear:
    """
    Models an Op-Amp Difference Amplifier with 'Leaky' Saturation.
    This way we can have sigmoid like behavior with a simple op-amp circuit
    but we can avoid getting stuck during backpropagation.
    """

    # constructor
    def __init__(self, Rf, R1, initial_thresh, leak = 0.1):
        self.Rf = float(Rf)
        self.R1 = float(R1)
        self.thresh = float(initial_thresh)
        self.leak = leak

    def get_gain(self):
        """
        Calculates the gain of the op-amp
        :return: the gain
        """
        return self.Rf / self.R1

    def forward(self, signal_sum, vccm, vccp):
        """
        Calculates the forward output of the op-amp
        and accounts for saturation
        :param signal_sum: the input signal to the neuron
        :param vccm: the negative rail voltage
        :param vccp: the positive rail voltage
        :returns: the op-amp output
        """
        
        # Calculate unsaturated gain using differential amplifier formula
        raw_output = self.get_gain() * (signal_sum - self.thresh)

        self.last_raw_out = raw_output
        
        # Hard saturation
        return max(vccm, min(vccp, raw_output))

    def derivative(self, vccm, vccp):
        """
        Returns the slope from the previous forward pass of the object
        If Active: Returns Gain.
        If Saturated: Returns Gain * leak (Leaky Gradient).
        :param vccm: the negative rail voltage
        :param vccp: the positive rail voltage
        """

        # Calculate the range of the op-amp
        # Use a small offset to assume an op-amp saturates when
        # very close to the rail
        swing = vccp - vccm
        epsilon = swing * 0.05 

        # Linear region: get slope
        if (vccm + epsilon) < self.last_raw_out < (vccp - epsilon):
            return self.get_gain()
        
        # Leaky logic: to prevent getting stuck,
        # pretend there is a small slope in the flat regions
        return self.get_gain() * self.leak
