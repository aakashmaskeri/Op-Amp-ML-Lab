import numpy as np
from activations import OpAmpClippedLinear

class AnalogCircuitBase:
    def __init__(self, eta=0.05):
        self.eta = eta
        self.vccm = 0.0
        self.vccp = 5.0
        self.weights = {}
        self.voltages = {}
        self.gradients = {}
        # Default Op-Amp Setup (Gain = 5.0)
        # Rf=100k, R1=20k. Threshold centered at 2.5V
        midpoint = (self.vccp + self.vccm) / 2.0
        self.act_fn = OpAmpClippedLinear(Rf=100.0, R1=20.0, initial_thresh=midpoint)

    def set_rails(self, low, high):
        self.vccm = float(low)
        self.vccp = float(high)
        self.act_fn.thresh = (self.vccp + self.vccm) / 2.0

    def update_weights(self):
        """Standard Gradient Descent with Clipping (0.0 to 1.0)"""
        for key in self.weights:
            if key in self.gradients:
                raw_change = self.eta * self.gradients[key]
                # Clip change to avoid violent swings (+/- 10%)
                clipped_change = max(-0.1, min(0.1, raw_change))
                self.weights[key] -= clipped_change
                
                # Physical Potentiometer Limits (0% to 100%)
                self.weights[key] = max(0.0, min(1.0, self.weights[key]))
        return self.weights

    def reset(self):
        self.__init__()

class AnalogAND(AnalogCircuitBase):
    """Standard Perceptron Topology"""
    def __init__(self):
        super().__init__()
        self.weights = {
            'w1': np.random.uniform(0.1, 0.9),
            'w2': np.random.uniform(0.1, 0.9)
        }
        self.voltages = {'v_out': 0.0, 'out': 0.0}

    def forward(self, x1, x2):
        s = (x1 * self.weights['w1']) + (x2 * self.weights['w2'])
        val = self.act_fn.forward(s, self.vccm, self.vccp)
        self.voltages['out'] = val
        self.voltages['v_out'] = val
        return self.voltages

    def backward(self, target, x1, x2):
        out = self.voltages['out']
        error = out - target
        slope = self.act_fn.derivative(out, self.vccm, self.vccp)
        delta = error * slope

        self.gradients['w1'] = delta * x1
        self.gradients['w2'] = delta * x2
        return error

class AnalogXOR(AnalogCircuitBase):
    """
    Excitatory (Blue) / Inhibitory (Red) Topology.
    Subtraction is handled structurally by the Output Diff Amp.
    Weights remain positive (0.0 - 1.0).
    """
    def __init__(self):
        super().__init__()
        self.weights = {
            # Blue Box (Excitatory)
            'w0': np.random.uniform(0.1, 0.9), 'w1': np.random.uniform(0.1, 0.9),
            # Red Box (Inhibitory)
            'w2': np.random.uniform(0.1, 0.9), 'w3': np.random.uniform(0.1, 0.9),
            # Green Box (Output Inputs)
            'w4': np.random.uniform(0.1, 0.9), # + Input
            'w5': np.random.uniform(0.1, 0.9)  # - Input (Subtracted)
        }
        self.voltages = {'h_exc': 0.0, 'h_inh': 0.0, 'out': 0.0}

    def forward(self, x1, x2):
        # 1. Excitatory (Blue)
        s_exc = (x1 * self.weights['w0']) + (x2 * self.weights['w1'])
        self.voltages['h_exc'] = self.act_fn.forward(s_exc, self.vccm, self.vccp)

        # 2. Inhibitory (Red)
        s_inh = (x1 * self.weights['w2']) + (x2 * self.weights['w3'])
        self.voltages['h_inh'] = self.act_fn.forward(s_inh, self.vccm, self.vccp)

        # 3. Output (Green) - DIFFERENCE AMPLIFIER
        # Output = (Blue * w4) - (Red * w5)
        s_out = (self.voltages['h_exc'] * self.weights['w4']) - \
                (self.voltages['h_inh'] * self.weights['w5'])
        
        self.voltages['out'] = self.act_fn.forward(s_out, self.vccm, self.vccp)
        return self.voltages

    def backward(self, target, x1, x2):
        out = self.voltages['out']
        h_exc, h_inh = self.voltages['h_exc'], self.voltages['h_inh']

        # Output Gradients
        error = out - target
        slope_out = self.act_fn.derivative(out, self.vccm, self.vccp)
        delta_out = error * slope_out

        # w4 is positive connection
        self.gradients['w4'] = delta_out * h_exc
        
        # w5 is negative connection. Partial deriv of (-w5*h_inh) wrt w5 is (-h_inh)
        self.gradients['w5'] = delta_out * (-h_inh)

        # Hidden Gradients
        # Blue (Exc) Path
        error_exc = delta_out * self.weights['w4']
        slope_exc = self.act_fn.derivative(h_exc, self.vccm, self.vccp)
        delta_exc = error_exc * slope_exc
        self.gradients['w0'] = delta_exc * x1
        self.gradients['w1'] = delta_exc * x2

        # Red (Inh) Path - Backprop through negative connection
        error_inh = delta_out * (-self.weights['w5'])
        slope_inh = self.act_fn.derivative(h_inh, self.vccm, self.vccp)
        delta_inh = error_inh * slope_inh
        self.gradients['w2'] = delta_inh * x1
        self.gradients['w3'] = delta_inh * x2
        
        return error
