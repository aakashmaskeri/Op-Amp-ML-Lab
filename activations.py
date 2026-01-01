class OpAmpClippedLinear:
    """
    Models an Op-Amp Difference Amplifier with 'Leaky' Saturation.
    This way we can have sigmoid like behavior with a simple op-amp circuit
    but we can avoid getting stuck during backpropagation.
    """

    # constructor
    def __init__(self, Rf, R1, initial_thresh):
        self.Rf = float(Rf)
        self.R1 = float(R1)
        self.thresh = float(initial_thresh)

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
        
        # Hard saturation
        return max(vccm, min(vccp, raw_output))

    def derivative(self, output_voltage, vccm, vccp):
        """
        Returns the slope.
        If Active: Returns Gain.
        If Saturated: Returns Gain * 0.01 (Leaky Gradient).
        :param output_voltage: the op-amp output
        :param vccm: the negative rail voltage
        :param vccp: the positive rail voltage
        """

        # Calculate the range of the op-amp
        swing = vccp - vccm
        epsilon = swing * 0.01 

        # Linear region: get slope
        if (vccm + epsilon) < output_voltage < (vccp - epsilon):
            return self.get_gain()
        
        # Leaky logic: to prevent getting stuck,
        # pretend there is a small slope in the flat regions
        return self.get_gain() * 0.01
