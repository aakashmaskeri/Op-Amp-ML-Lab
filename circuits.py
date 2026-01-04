import numpy as np
from activations import OpAmpClippedLinear

class AnalogCircuitBase:
    # Constructor
    def __init__(self, eta=0.01):
        self.eta = eta      # greek letter eta, which controls step size for backprop
        self.vccm = -5.0     # negative rail voltage
        self.vccp = 5.0     # positive rail voltage
        self.weights = {}   
        self.voltages = {}
        self.gradients = {}
       
        # Midpoint is the halfway point
        midpoint = (self.vccp + self.vccm) / 2.0

        # Create the op-amp object
        self.act_fn = OpAmpClippedLinear(Rf=100.0, R1=10.0, initial_thresh=2.6)

    def set_rails(self, low, high):
        """
        Set op-amp rail voltages

        :param low: low rail voltage
        :param high: high rail voltage
        """
        self.vccm = float(low)
        self.vccp = float(high)
        # self.act_fn.thresh = (self.vccp + self.vccm) / 2.0
        self.act_fn.thresh = 2.6 #threshold is slightly above midpoint

    def get_pot_output(self, digi_vin, weight):
        """
        Calculates wiper voltage when Pot is connected between Input and Vcc-
        Formula: V_out = V_bottom + (V_top - V_bottom) * w

        :param digi_vin: the digital input voltage
        :weight: the pot weight
        """
        return self.vccm + (digi_vin - self.vccm) * weight

    def update_weights(self):
        """Standard Gradient Descent with Clipping (0.0 to 1.0)"""

        # Iterate through weights and update weights
        for key in self.weights:
            if key in self.gradients:
                # Use calculated gradients
                raw_change = self.eta * self.gradients[key]

                # Clip change to avoid violent swings (+/- 10%)
                # This was added as the linear region would send 
                # the weight from close to 1 to close to 0 
                # and vice-versa frequently
                clipped_change = max(-0.1, min(0.1, raw_change))
                self.weights[key] -= clipped_change
                
                # Physical Potentiometer Limits (0% to 100%)
                self.weights[key] = max(0.0, min(1.0, self.weights[key]))
        return self.weights

    def reset(self):
        self.__init__()

class AnalogAND(AnalogCircuitBase):
    """
    AND gate circuit representation
    Uses single neuron
    """

    # Constructor
    def __init__(self):
        super().__init__()
        self.weights = {
            'w1': np.random.uniform(0.1, 0.9),
            'w2': np.random.uniform(0.1, 0.9)
        }
        self.voltages = {'v_out': 0.0, 'out': 0.0}

    def forward(self, x1, x2):
        """
        Forward activation of AND gate
        
        :param x1: digital input 1
        :param x2: digital input 2
        """

        # Multiply and accumulate circuit
        s = ((self.get_pot_output(x1, self.weights['w1'])) + (self.get_pot_output(x2, self.weights['w2']))) / 2.0

        # Pass sum to neuron and update output
        val = self.act_fn.forward(s, self.vccm, self.vccp)
        self.voltages['out'] = val
        self.voltages['v_out'] = val
        return self.voltages

    def backward(self, target, x1, x2):
        """
        Backward pass of AND gate
        for backpropagation
        
        :param target: theoretical response of AND gate
        :param x1: digital input 1
        :param x2: digital input 2
        """

        # Determine error
        out = self.voltages['out']
        error = out - target
        slope = self.act_fn.derivative(self.vccm, self.vccp)
        delta = error * slope

        swing_w1 = x1 - self.vccm
        swing_w2 = x2 - self.vccm

        # Calculate gradients
        self.gradients['w1'] = delta * swing_w1 * 0.5
        self.gradients['w2'] = delta * swing_w2 * 0.5
        return error

class AnalogXOR(AnalogCircuitBase):
    """
    Excitatory (Blue) / Inhibitory (Red) Topology.
    Subtraction is handled structurally by the Output Diff Amp.
    Weights remain positive (0.0 - 1.0).
    """

    # Constructor
    def __init__(self):
        super().__init__()
        self.weights = {
            # Blue Box (Excitatory)
            'w0': np.random.uniform(0.1, 0.9), 'w1': np.random.uniform(0.1, 0.9),
            # Red Box (Inhibitory)
            'w2': np.random.uniform(0.1, 0.9), 'w3': np.random.uniform(0.1, 0.9),
            # Green Box (Output Inputs)
            'w4': np.random.uniform(0.1, 0.9), # + Input
            'w5': np.random.uniform(0.1, 0.9),  # - Input (Subtracted)

            # Thresholds (biases)
            'tpos': np.random.uniform(0.5, 0.7),
            'tneg': np.random.uniform(0.5, 0.7)

        }
        self.voltages = {'h_exc': 0.0, 'h_inh': 0.0, 'out': 0.0}

        # Blue Neuron (OR-like): 
        self.act_blue = OpAmpClippedLinear(Rf=200.0, R1=10.0, initial_thresh=0.0)
        
        # Red Neuron (AND-like): 
        self.act_red  = OpAmpClippedLinear(Rf=200.0, R1=10.0, initial_thresh=0.0)
        
        # Green Neuron uses the standard defined above so no specific object
        self.act_green = OpAmpClippedLinear(Rf=200.0, R1=10.0, initial_thresh=0.0)

    def forward(self, x1, x2):
        """
        Forward activation of XOR gate
        
        :param x1: digital input 1
        :param x2: digital input 2
        """

        # Get initial thresholds
        thresh_pos = -5 + (5 - (-5)) * self.weights['tpos']
        thresh_neg = -5 + (5 - (-5)) * self.weights['tneg']

        self.act_blue.thresh = self.act_green.thresh = thresh_pos
        self.act_red.thresh = thresh_neg

        # 1. Excitatory (Blue)
        s_exc = ((self.get_pot_output(x1, self.weights['w0'])) + (self.get_pot_output(x2, self.weights['w1']))) / 2.0
        h_exc = self.act_blue.forward(s_exc, self.vccm, self.vccp)

        # 2. Inhibitory (Red)
        s_inh = ((self.get_pot_output(x1, self.weights['w2'])) + (self.get_pot_output(x2, self.weights['w3']))) * -1.0
        h_inh = self.act_red.forward(s_inh, self.vccm, self.vccp)

        # 3. Output (Green) - DIFFERENCE AMPLIFIER
        # Output = (Blue * w4) + (Red * w5) where red is already negative
        s_out = ((self.get_pot_output(h_exc, self.weights['w4'])) + \
                (self.get_pot_output(h_inh, self.weights['w5']))) / 2.0
        
        out = self.act_green.forward(s_out, self.vccm, self.vccp)

        self.voltages.update({
            's_exc': s_exc, 'h_exc': h_exc,
            's_inh': s_inh, 'h_inh': h_inh,
            's_out': s_out, 'out': out
        })
        return self.voltages

    def backward(self, target, x1, x2):
        """
        Backward pass of XOR gate
        for backpropagation
        
        :param target: theoretical response of XOR gate
        :param x1: digital input 1
        :param x2: digital input 2
        """

        out = self.voltages['out']
        h_exc, h_inh = self.voltages['h_exc'], self.voltages['h_inh']

        # Calculate swings
        swing_x1 = x1 - self.vccm
        swing_x2 = x2 - self.vccm  
        swing_exc = h_exc - self.vccm
        swing_inh = h_inh - self.vccm

        # Output Gradients
        error = out - target
        slope_out = self.act_green.derivative(self.vccm, self.vccp)
        delta_out = error * slope_out

        # Green connections
        self.gradients['w4'] = delta_out * swing_exc * 0.5
        self.gradients['w5'] = delta_out * swing_inh * 0.5

        # Hidden Gradients
        # Blue (Exc) Path
        error_exc = delta_out * self.weights['w4'] * 0.5
        slope_exc = self.act_blue.derivative(self.vccm, self.vccp)
        delta_exc = error_exc * slope_exc
        self.gradients['w0'] = delta_exc * swing_x1 * 0.5
        self.gradients['w1'] = delta_exc * swing_x2 * 0.5

        # Red (Inh) Path - Backprop through negative connection
        error_inh = delta_out * self.weights['w5'] * 0.5
        slope_inh = self.act_red.derivative(self.vccm, self.vccp)
        delta_inh = error_inh * slope_inh
        self.gradients['w2'] = -delta_inh * swing_x1
        self.gradients['w3'] = -delta_inh * swing_x2

        self.gradients['tpos'] = delta_out * -10.0 + delta_exc * -10.0
        self.gradients['tneg'] = delta_inh * 10
        
        return error
