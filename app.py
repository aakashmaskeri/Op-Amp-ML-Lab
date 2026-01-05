from flask import Flask, jsonify, request, render_template
from circuits import AnalogAND, AnalogXOR
import random

app = Flask(__name__)

# Maintain state for both circuits
circuits = {
    'and': AnalogAND(),
    'xor': AnalogXOR()
}

# Webapp pages
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/and')
def page_and():
    return render_template('and.html')

@app.route('/xor')
def page_xor():
    return render_template('xor.html')

# --- API ROUTES ---

@app.route('/api/<ctype>/state', methods=['GET'])
def get_state(ctype):
    c = circuits.get(ctype)
    if not c: return jsonify({'error': 'Invalid circuit'}), 400
    return jsonify({
        'weights': {**c.weights, 't': c.act_fn.thresh},
        'voltages': c.voltages,
        'rails': {'min': c.vccm, 'max': c.vccp}
    })

@app.route('/api/<ctype>/step_forward', methods=['POST'])
def step_forward(ctype):
    c = circuits.get(ctype)
    d = request.get_json(force=True, silent=True)
    x1 = c.vccp if d.get('x1', 0) == 1 else 0.0
    x2 = c.vccp if d.get('x2', 0) == 1 else 0.0
    return jsonify(c.forward(x1, x2))

@app.route('/api/<ctype>/step_backward', methods=['POST'])
def step_backward(ctype):
    c = circuits.get(ctype)
    d = request.get_json(force=True, silent=True)
    
    x1_log, x2_log = d.get('x1', 0), d.get('x2', 0)
    
    # Target Logic
    is_high = False
    if ctype == 'and': is_high = (x1_log == 1 and x2_log == 1)
    if ctype == 'xor': is_high = (x1_log != x2_log)
    
    target = c.vccp if is_high else c.vccm
    x1 = c.vccp if x1_log == 1 else 0.0
    x2 = c.vccp if x2_log == 1 else 0.0

    error = c.backward(target, x1, x2)
    # Return gradients + squared error for logging
    return jsonify({**c.gradients, 'error': error, 'loss': error**2})

@app.route('/api/<ctype>/step_update', methods=['POST'])
def step_update(ctype):
    c = circuits.get(ctype)
    return jsonify(c.update_weights())

@app.route('/api/<ctype>/train_loop', methods=['POST'])
def train_loop(ctype):
    c = circuits.get(ctype)
    d = request.get_json(force=True, silent=True)
    epochs = int(d.get('steps', 10))
    
    truth_table = []
    if ctype == 'and':
        truth_table = [(0,0,0), (0,1,0), (1,0,0), (1,1,1)]
    elif ctype == 'xor':
        truth_table = [(0,0,0), (0,1,1), (1,0,1), (1,1,0)]
    
    history = []
    
    for epoch in range(epochs):
        random.shuffle(truth_table)
        for x1_log, x2_log, target_log in truth_table:
            target = c.vccp if target_log == 1 else c.vccm
            x1 = c.vccp if x1_log == 1 else 0.0
            x2 = c.vccp if x2_log == 1 else 0.0
            
            c.forward(x1, x2)
            error = c.backward(target, x1, x2)
            new_w = c.update_weights()
            
            history.append({
                'epoch': epoch + 1,
                'x1': x1_log, 'x2': x2_log,
                'output': c.voltages['out'],
                'loss': error ** 2,
                'weights': new_w.copy()
            })

    # for epoch in range(epochs):
    #     total_loss = 0.0
    #     c.zero_grad()

    #     # --- PROCESS BATCH: Accumulate Gradients for all 4 cases ---
    #     for x1_log, x2_log, target_log in truth_table:
    #         # Map Logic (0/1) to Voltages (Min/Max)
    #         target = c.vccp if target_log == 1 else c.vccm
    #         x1 = c.vccp if x1_log == 1 else 0.0
    #         x2 = c.vccp if x2_log == 1 else 0.0
            
    #         # Forward Pass
    #         c.forward(x1, x2)
            
    #         # Backward Pass (Calculates gradients for THIS input)
    #         error = c.backward(target, x1, x2)
    #         total_loss += error ** 2
            
    #         # Accumulate: Add this sample's gradients to the batch pile
    #         c.accumulate_grad()
            
    #     # --- BATCH END: Update Weights ONCE using the average ---
    #     # This prevents the "zig-zag" oscillation of simple SGD
    #     new_w = c.update_weights()
        
    #     # Logging: We log the average loss and the state after the update
    #     history.append({
    #         'epoch': epoch + 1,
    #         'x1': '-', 'x2': '-', # Inputs irrelevant for batch summary
    #         'output': 0.0,        # Output irrelevant for batch summary
    #         'loss': total_loss / 4.0, # Average MSE for the batch
    #         'weights': new_w.copy()
    #     })
            
    return jsonify(history)

@app.route('/api/<ctype>/reset', methods=['POST'])
def reset(ctype):
    circuits.get(ctype).reset()
    return jsonify({'status': 'ok'})

@app.route('/api/<ctype>/set_rails', methods=['POST'])
def set_rails(ctype):
    c = circuits.get(ctype)
    d = request.get_json(force=True, silent=True)
    c.set_rails(d.get('min', 0.0), d.get('max', 5.0))
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
