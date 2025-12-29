import numpy as np
import random
from collections import defaultdict


def q_learning(env, gamma=0.99, alpha=0.1, epsilon=0.1, n_episodes=500):
    ns =env.n_states
    na= env.n_actions
    q_table = np.zeros((ns, na))
    rewards_history =[]

    for ep in range(n_episodes):
        s=env.reset()
        finished =False
        total_r=0
        step_cnt =0

        while not finished and step_cnt<1000:
            sid =env.state_to_idx(s)

            # epsilon greedy action selection
            if random.random()< epsilon:
                a= random.randint(0, na-1)
            else:
                a =np.argmax(q_table[sid])

            s_next,r, finished = env.step(a)
            sid_next =env.state_to_idx(s_next)
            total_r +=r

            # off-policy update with max q
            if finished:
                q_table[sid, a] +=alpha * (r - q_table[sid, a])
            else:
                best_next =np.max(q_table[sid_next])
                q_table[sid, a]+= alpha* (r+ gamma*best_next - q_table[sid,a])

            s =s_next
            step_cnt+= 1

        rewards_history.append(total_r)

    pol = np.argmax(q_table, axis=1)
    return pol, q_table, rewards_history


def sarsa(env, gamma=0.99, alpha=0.1, epsilon=0.1, n_episodes=500):
    ns= env.n_states
    na =env.n_actions
    q_vals =np.zeros((ns, na))
    hist= []

    for ep_num in range(n_episodes):
        curr_state = env.reset()
        curr_idx=env.state_to_idx(curr_state)

        if random.random() <epsilon:
            curr_act= random.randint(0,na-1)
        else:
            curr_act =np.argmax(q_vals[curr_idx])

        done_flag =False
        ep_reward =0
        steps =0

        while not done_flag and steps< 1000:
            new_state, rew, done_flag= env.step(curr_act)
            new_idx =env.state_to_idx(new_state)
            ep_reward+= rew

            if random.random()<epsilon:
                next_act =random.randint(0, na-1)
            else:
                next_act=np.argmax(q_vals[new_idx])

            # on-policy update
            if done_flag:
                q_vals[curr_idx, curr_act]+= alpha* (rew- q_vals[curr_idx, curr_act])
            else:
                target =rew + gamma*q_vals[new_idx, next_act]
                q_vals[curr_idx,curr_act] +=alpha * (target - q_vals[curr_idx, curr_act])

            curr_state= new_state
            curr_idx= new_idx
            curr_act =next_act
            steps +=1

        hist.append(ep_reward)

    policy =np.argmax(q_vals, axis=1)
    return policy,q_vals, hist


def monte_carlo(env, gamma=0.99, epsilon=0.1, n_episodes=500):
    ns =env.n_states
    na =env.n_actions

    q =np.zeros((ns,na))
    ret_sum =defaultdict(float)
    ret_cnt= defaultdict(int)
    h =[]

    for e in range(n_episodes):
        traj =[]
        st =env.reset()
        is_done= False
        step_count=0
        max_step =1000

        while not is_done and step_count <max_step:
            s_i =env.state_to_idx(st)

            if random.random() <epsilon:
                act =random.randint(0, na-1)
            else:
                act= np.argmax(q[s_i])

            st_next, reward,is_done = env.step(act)
            traj.append((st, act,reward))
            st= st_next
            step_count +=1

        # backwards return calculation
        g_val =0
        seen= set()

        for idx in reversed(range(len(traj))):
            state, action, r =traj[idx]
            g_val =gamma * g_val+ r
            s_idx= env.state_to_idx(state)

            # first-visit MC
            if (s_idx,action) not in seen:
                seen.add((s_idx, action))
                ret_sum[(s_idx,action)] +=g_val
                ret_cnt[(s_idx, action)]+=1
                q[s_idx,action] = ret_sum[(s_idx, action)]/ ret_cnt[(s_idx, action)]

        avg =np.mean(np.max(q, axis=1))
        h.append(avg)

    pol= np.argmax(q, axis=1)
    return pol, q,h


def value_iteration(env, gamma=0.99, theta=1e-6):
    num_s =env.n_states
    num_a= env.n_actions
    v= np.zeros(num_s)
    conv_hist =[]

    iter_count= 0
    while True:
        max_change =0

        for si in range(num_s):
            state =env.idx_to_state(si)
            old_v=v[si]

            action_vals =np.zeros(num_a)
            for act in range(num_a):
                trans= env.get_transitions(state, act)
                for nxt_state, prob,reward, terminal in trans:
                    nxt_i= env.state_to_idx(nxt_state)
                    if terminal:
                        action_vals[act] +=prob*reward
                    else:
                        action_vals[act]+=prob *(reward+ gamma*v[nxt_i])

            v[si]= np.max(action_vals)
            max_change =max(max_change, abs(old_v -v[si]))

        conv_hist.append(max_change)
        iter_count +=1

        if max_change< theta:
            break
        if iter_count>1000:
            break

    p =make_greedy_policy(env,v, gamma)
    return p, v,conv_hist


def make_greedy_policy(env, values, gamma=0.99):
    num_states =env.n_states
    num_actions=env.n_actions
    pol =np.zeros(num_states, dtype=int)

    for s_i in range(num_states):
        st= env.idx_to_state(s_i)

        q_arr =np.zeros(num_actions)
        for a in range(num_actions):
            transitions =env.get_transitions(st, a)
            for next_s,probability, rew, is_terminal in transitions:
                next_i =env.state_to_idx(next_s)
                if is_terminal:
                    q_arr[a] +=probability * rew
                else:
                    q_arr[a]+= probability* (rew +gamma * values[next_i])

        best_a= np.argmax(q_arr)
        pol[s_i] =best_a

    return pol


def policy_iteration(env, gamma=0.99, theta=1e-6):
    num_st =env.n_states
    num_act =env.n_actions

    pi= np.random.randint(0, num_act, size=num_st)
    val_func =np.zeros(num_st)

    all_conv_hist=[]
    iters =0

    while True:
        val_func, eval_h =evaluate_policy(env, pi,gamma, theta)
        all_conv_hist.extend(eval_h)

        prev_pi= pi.copy()
        pi= make_greedy_policy(env, val_func, gamma)

        iters+=1

        if np.array_equal(prev_pi, pi):
            break
        if iters >100:
            break

    return pi, val_func,all_conv_hist


def evaluate_policy(env, policy,gamma=0.99, theta=1e-6):
    n_st =env.n_states
    v_arr= np.zeros(n_st)
    convergence= []

    iteration=0
    while True:
        delta_max= 0
        for state_i in range(n_st):
            state_obj= env.idx_to_state(state_i)
            action_to_take =policy[state_i]

            trans_list =env.get_transitions(state_obj, action_to_take)

            new_v =0
            for nxt, p, r,done in trans_list:
                nxt_idx= env.state_to_idx(nxt)
                if done:
                    new_v+= p*r
                else:
                    new_v +=p* (r+ gamma* v_arr[nxt_idx])

            delta_max= max(delta_max, abs(v_arr[state_i]- new_v))
            v_arr[state_i] =new_v

        convergence.append(delta_max)
        iteration +=1

        if delta_max<theta:
            break
        if iteration> 1000:
            break

    return v_arr,convergence


def td_prediction(env, pol, gamma=0.99, alpha=0.1, n_episodes=500):
    n_states= env.n_states
    v_est =np.zeros(n_states)
    tracking =[]

    for episode in range(n_episodes):
        state_now =env.reset()
        finished= False
        cnt =0

        while not finished and cnt<1000:
            si =env.state_to_idx(state_now)
            act =pol[si]

            state_nxt, r, finished =env.step(act)
            si_nxt =env.state_to_idx(state_nxt)

            if finished:
                td_targ =r
            else:
                td_targ= r +gamma * v_est[si_nxt]

            v_est[si] =v_est[si] +alpha * (td_targ- v_est[si])

            state_now =state_nxt
            cnt+= 1

        tracking.append(np.mean(v_est))

    return v_est, tracking


def n_step_td(env, pol, n=4,gamma=0.99, alpha=0.1, n_episodes=500):
    n_st= env.n_states
    vals =np.zeros(n_st)
    progress =[]

    for episode_num in range(n_episodes):
        s_curr =env.reset()
        state_buf =[s_curr]
        reward_buf =[0]

        terminal_time =float('inf')
        time =0

        while True:
            if time< terminal_time:
                s_i= env.state_to_idx(s_curr)
                action= pol[s_i]
                s_nxt, r_val, done =env.step(action)

                state_buf.append(s_nxt)
                reward_buf.append(r_val)

                if done:
                    terminal_time= time+1
                else:
                    s_curr =s_nxt

            update_time =time - n+1

            if update_time>= 0:
                ret=0
                for i in range(update_time +1, min(update_time+ n, terminal_time)+ 1):
                    ret +=(gamma**(i- update_time-1)) * reward_buf[i]

                if update_time +n< terminal_time:
                    s_tau_n= state_buf[update_time+ n]
                    s_tau_n_i =env.state_to_idx(s_tau_n)
                    ret +=(gamma**n) *vals[s_tau_n_i]

                s_tau =state_buf[update_time]
                s_tau_i= env.state_to_idx(s_tau)
                vals[s_tau_i]+= alpha * (ret- vals[s_tau_i])

            if update_time ==terminal_time- 1:
                break

            time +=1
            if time>1000:
                break

        progress.append(np.mean(vals))

    return vals, progress


def run_algorithm(env, algo_name,params):
    g =params.get('gamma', 0.99)
    a= params.get('alpha', 0.1)
    eps= params.get('epsilon', 0.1)
    episodes =params.get('n_episodes', 500)
    convergence_thresh =params.get('theta', 1e-6)
    n_steps= params.get('n_step', 4)

    if algo_name== 'policy_iteration':
        p, v, h= policy_iteration(env, g, convergence_thresh)
        return {'policy': p.tolist(), 'values': v.tolist(),'history': h}

    elif algo_name== 'value_iteration':
        p,v, h =value_iteration(env, g, convergence_thresh)
        return {'policy': p.tolist(), 'values': v.tolist(), 'history': h}

    elif algo_name =='monte_carlo':
        p, q,h= monte_carlo(env, g, eps, episodes)
        v= np.max(q, axis=1)
        return {'policy': p.tolist(),'values': v.tolist(), 'history': h, 'q_values': q.tolist()}

    elif algo_name== 'td':
        rand_pol =np.random.randint(0, env.n_actions, size=env.n_states)
        v, h =td_prediction(env, rand_pol, g,a, episodes)
        return {'policy': rand_pol.tolist(), 'values':v.tolist(), 'history': h}

    elif algo_name =='n_step_td':
        rand_pol= np.random.randint(0, env.n_actions, size=env.n_states)
        v,h =n_step_td(env, rand_pol, n_steps, g, a,episodes)
        return {'policy': rand_pol.tolist(), 'values': v.tolist(), 'history':h}

    elif algo_name=='sarsa':
        p, q, h=sarsa(env, g, a, eps, episodes)
        v =np.max(q, axis=1)
        return {'policy': p.tolist(), 'values': v.tolist(),'history': h, 'q_values': q.tolist()}

    elif algo_name =='q_learning':
        p, q,h =q_learning(env, g, a,eps, episodes)
        v= np.max(q, axis=1)
        return {'policy':p.tolist(), 'values': v.tolist(), 'history': h, 'q_values': q.tolist()}

    else:
        return {'error': 'Unknown algorithm'}
