from flask import Flask, jsonify, request, render_template
import numpy as np
from activations import OpAmpClippedLinear

app = Flask(__name__)

class AnalogANDCircuit:
    def __init__(self):
        self.eta = 0.05  
        self.vccm = 0.0 
        self.vccp = 5.0 

        # Gain = 5.0 (100k / 20k)
        midpoint = (self.vccp + self.vccm) / 2.0
        self.act_fn = OpAmpClippedLinear(Rf=100.0, R1=20.0, initial_thresh=midpoint)

        self.weights = {
            'w1': np.random.uniform(0.1, 0.9),
            'w2': np.random.uniform(0.1, 0.9)
        }

        self.voltages = { 'v_signal': 0.0, 'v_out': 0.0, 'out': 0.0 }
        self.gradients = {}

    def set_rails(self, low, high):
        self.vccm = float(low)
        self.vccp = float(high)
        self.act_fn.thresh = (self.vccp + self.vccm) / 2.0

    def forward(self, x1, x2):
        signal_sum = (x1 * self.weights['w1']) + (x2 * self.weights['w2'])
        self.voltages['v_signal'] = signal_sum
        val = self.act_fn.forward(signal_sum, self.vccm, self.vccp)
        self.voltages['out'] = val
        self.voltages['v_out'] = val
        return self.voltages

    def backward(self, target, x1, x2):
        output = self.voltages['out']
        error = output - target
        slope = self.act_fn.derivative(output, self.vccm, self.vccp)
        delta = error * slope
        
        self.gradients['w1'] = delta * x1
        self.gradients['w2'] = delta * x2
        self.gradients['slope_val'] = slope
        
        return error 

    def update(self):
        for key in ['w1', 'w2']:
            if key in self.gradients:
                raw_change = self.eta * self.gradients[key]
                clipped_change = max(-0.1, min(0.1, raw_change))
                self.weights[key] -= clipped_change
                self.weights[key] = max(0.0, min(1.0, self.weights[key]))
        return self.weights

    def reset(self):
        self.__init__()

circuit = AnalogANDCircuit()

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/state', methods=['GET'])
def get_state():
    return jsonify({
        'weights': {**circuit.weights, 't': circuit.act_fn.thresh},
        'voltages': circuit.voltages,
        'rails': {'min': circuit.vccm, 'max': circuit.vccp}
    })

@app.route('/set_rails', methods=['POST'])
def set_rails():
    d = request.get_json(force=True, silent=True)
    if not d: return jsonify({"error": "No data"}), 400
    circuit.set_rails(d.get('min', 0.0), d.get('max', 5.0))
    return jsonify({'status': 'ok', 't': circuit.act_fn.thresh})

# --- MANUAL STEPS ---
@app.route('/step_forward', methods=['POST'])
def step_forward():
    d = request.get_json(force=True, silent=True)
    if not d: return jsonify({"error": "No data"}), 400
    
    x1_val = circuit.vccp if d.get('x1', 0) == 1 else circuit.vccm
    x2_val = circuit.vccp if d.get('x2', 0) == 1 else circuit.vccm
    return jsonify(circuit.forward(x1_val, x2_val))

@app.route('/step_backward', methods=['POST'])
def step_backward():
    d = request.get_json(force=True, silent=True)
    if not d: return jsonify({"error": "No data"}), 400
    
    x1_logic, x2_logic = d.get('x1', 0), d.get('x2', 0)
    target = circuit.vccp if (x1_logic == 1 and x2_logic == 1) else circuit.vccm
    x1_val = circuit.vccp if x1_logic == 1 else circuit.vccm
    x2_val = circuit.vccp if x2_logic == 1 else circuit.vccm

    error = circuit.backward(target, x1_val, x2_val)
    return jsonify({**circuit.gradients, 'error': error})

@app.route('/step_update', methods=['POST'])
def step_update():
    return jsonify(circuit.update())

@app.route('/reset', methods=['POST'])
def reset_circuit():
    circuit.reset()
    return jsonify({'status': 'ok'})

# --- AUTOMATIC LOOP ---
@app.route('/train_loop', methods=['POST'])
def train_loop():
    d = request.get_json(force=True, silent=True)
    if not d: return jsonify({"error": "No data"}), 400
    
    steps = int(d.get('steps', 10))
    x1_logic, x2_logic = d.get('x1', 0), d.get('x2', 0)
    
    history = []
    
    for i in range(steps):
        # 1. Setup Targets
        target = circuit.vccp if (x1_logic == 1 and x2_logic == 1) else circuit.vccm
        x1_val = circuit.vccp if x1_logic == 1 else circuit.vccm
        x2_val = circuit.vccp if x2_logic == 1 else circuit.vccm
        
        # 2. Forward
        state = circuit.forward(x1_val, x2_val)
        v_out = state['out']
        
        # 3. Backward
        error = circuit.backward(target, x1_val, x2_val)
        
        # 4. Update
        new_weights = circuit.update()
        
        # 5. Log Data (For the Table)
        history.append({
            'x1': x1_logic,
            'x2': x2_logic,
            'target': target,
            'output': v_out,
            'loss': error ** 2, # MSE
            'w1': new_weights['w1'],
            'w2': new_weights['w2']
        })
        
    return jsonify(history)

if __name__ == '__main__':
    app.run(debug=True)
