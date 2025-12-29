from flask import Flask, render_template, request, jsonify
import numpy as np
import json
from datetime import datetime

from environments import get_environment,GridWorld, FrozenLake, CliffWalking, MountainCar,CartPole
from algorithms import run_algorithm

app= Flask(__name__)

curr_env =None
curr_st= None

event_log= []

def add_log(event_typ, details):
    ts =datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    entry ={
        'timestamp': ts,
        'type': event_typ,
        'details':details
    }
    event_log.append(entry)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/init_env', methods=['POST'])
def init_env():
    global curr_env, curr_st

    try:
        data= request.json
        env_name =data.get('env', 'gridworld')
        params= data.get('params', {})

        add_log('INIT_ENV', {'environment': env_name,'params': params})

        curr_env =get_environment(env_name, **params)
        curr_st= curr_env.reset()

        info ={
            'n_states': curr_env.n_states,
            'n_actions': curr_env.n_actions,
            'actions': curr_env.moves if hasattr(curr_env, 'moves') else curr_env.actions,
            'state': list(curr_st) if isinstance(curr_st, tuple) else curr_st
        }

        # check frozenlake first (has both sz and hole_locs)
        if hasattr(curr_env, 'hole_locs'):
            info['holes']= curr_env.hole_locs
            info['goal']= curr_env.goal_loc
            info['grid_size']= curr_env.sz
        elif hasattr(curr_env, 'sz'):
            info['grid_size']= curr_env.sz
            info['goal'] =curr_env.target
            if hasattr(curr_env, 'blocks'):
                info['obstacles']= curr_env.blocks
        elif hasattr(curr_env, 'h'):
            info['grid_height']= curr_env.h
            info['grid_width'] =curr_env.w
            if hasattr(curr_env, 'danger_zone'):
                info['cliff'] =curr_env.danger_zone
            info['goal'] =curr_env.end_pos

        add_log('INIT_SUCCESS', {'n_states': curr_env.n_states, 'n_actions': curr_env.n_actions})

        return jsonify(info)

    except Exception as e:
        import traceback
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        print(f"ERROR initializing environment: {error_msg}")
        print(stack_trace)
        return jsonify({'error': error_msg, 'trace': stack_trace}), 500


@app.route('/api/step', methods=['POST'])
def step():
    global curr_env, curr_st

    if curr_env is None:
        return jsonify({'error': 'Environment not initialized'})

    data =request.json
    act =data.get('action', 0)

    nxt_st, rew, finished =curr_env.step(act)
    curr_st =nxt_st

    add_log('MANUAL_STEP', {
        'action': int(act),
        'state': list(nxt_st) if isinstance(nxt_st, tuple) else str(nxt_st),
        'reward': float(rew),
        'done': bool(finished)
    })

    return jsonify({
        'state': list(nxt_st) if isinstance(nxt_st, tuple) else nxt_st,
        'reward': float(rew),
        'done': bool(finished)
    })


@app.route('/api/reset', methods=['POST'])
def reset():
    global curr_env, curr_st

    if curr_env is None:
        return jsonify({'error': 'Environment not initialized'})

    curr_st =curr_env.reset()

    add_log('RESET_ENV', {
        'state': list(curr_st) if isinstance(curr_st, tuple) else str(curr_st)
    })

    return jsonify({
        'state': list(curr_st) if isinstance(curr_st, tuple) else curr_st
    })


@app.route('/api/train', methods=['POST'])
def train():
    global curr_env

    if curr_env is None:
        return jsonify({'error': 'Environment not initialized'})

    data= request.json
    algo =data.get('algorithm', 'q_learning')
    params= data.get('params', {})

    add_log('TRAIN_START', {'algorithm': algo,'params': params})

    result= run_algorithm(curr_env, algo, params)

    add_log('TRAIN_COMPLETE', {
        'algorithm': algo,
        'iterations': result.get('iterations', 'N/A'),
        'convergence': result.get('convergence', [])[-5:] if result.get('convergence') else []
    })

    return jsonify(result)


@app.route('/api/run_episode', methods=['POST'])
def run_episode():
    global curr_env

    if curr_env is None:
        return jsonify({'error': 'Environment not initialized'})

    data =request.json
    pol= data.get('policy', None)

    if pol is None:
        return jsonify({'error': 'No policy provided'})

    add_log('RUN_EPISODE_START', {})

    st =curr_env.reset()
    path =[]
    tot_rew =0
    finished= False
    max_st= 200

    step_cnt =0
    while not finished and step_cnt< max_st:
        s_idx= curr_env.state_to_idx(st)
        act= pol[s_idx]

        nxt_st, rew, finished =curr_env.step(act)

        path.append({
            'state': list(nxt_st) if isinstance(nxt_st, tuple) else nxt_st,
            'action': int(act),
            'reward': float(rew),
            'done': bool(finished)
        })

        tot_rew +=rew
        st= nxt_st
        step_cnt +=1

    add_log('RUN_EPISODE_COMPLETE', {
        'steps': step_cnt,
        'total_reward': float(tot_rew),
        'success': bool(finished)
    })

    return jsonify({
        'trajectory': path,
        'total_reward': tot_rew
    })


@app.route('/api/get_logs', methods=['GET'])
def get_logs():
    return jsonify({'logs': event_log})


if __name__== '__main__':
    app.run(debug=True, port=5000)
