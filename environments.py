import numpy as np
import random
import gymnasium as gym


class GridWorld:
    def __init__(self, size=5, goal=(4,4), obstacles=[]):
        self.sz =size
        self.target= goal
        self.blocks =obstacles
        self.curr_pos =(0,0)
        self.moves =['up', 'down', 'left', 'right']
        self.n_actions= 4
        self.n_states =size*size

    def reset(self):
        self.curr_pos =(0, 0)
        return self.curr_pos

    def step(self, action):
        r,c =self.curr_pos

        if action ==0:
            r= max(0, r-1)
        elif action== 1:
            r =min(self.sz-1, r+1)
        elif action ==2:
            c =max(0, c-1)
        elif action== 3:
            c= min(self.sz-1, c+1)

        nxt =(r,c)

        if nxt in self.blocks:
            nxt =self.curr_pos

        self.curr_pos= nxt

        if self.curr_pos== self.target:
            return self.curr_pos, 10, True
        elif self.curr_pos in self.blocks:
            return self.curr_pos,-5, False
        else:
            return self.curr_pos, -1,False

    def get_all_states(self):
        all_s =[]
        for row in range(self.sz):
            for col in range(self.sz):
                all_s.append((row, col))
        return all_s

    def state_to_idx(self, state):
        return state[0]* self.sz+ state[1]

    def idx_to_state(self, idx):
        return (idx//self.sz, idx% self.sz)

    def get_transitions(self, state, action):
        if state== self.target:
            return [(state,1.0, 0, True)]

        r, c= state

        if action==0:
            nr =max(0, r-1)
            nc= c
        elif action ==1:
            nr= min(self.sz-1, r+1)
            nc =c
        elif action== 2:
            nr= r
            nc =max(0, c-1)
        elif action ==3:
            nr =r
            nc= min(self.sz-1, c+1)

        nxt_st= (nr, nc)
        if nxt_st in self.blocks:
            nxt_st =state

        if nxt_st ==self.target:
            return [(nxt_st, 1.0,10, True)]
        else:
            return [(nxt_st, 1.0, -1,False)]


class CliffWalking:
    def __init__(self):
        self.h =4
        self.w= 12
        self.start_pos =(3, 0)
        self.end_pos =(3,11)
        self.danger_zone =[(3,i) for i in range(1, 11)]
        self.curr =self.start_pos
        self.moves= ['up','down', 'left', 'right']
        self.n_actions =4
        self.n_states= self.h *self.w

    def reset(self):
        self.curr= self.start_pos
        return self.curr

    def step(self, action):
        r, c= self.curr

        if action== 0:
            r= max(0, r-1)
        elif action ==1:
            r =min(self.h-1, r+1)
        elif action==2:
            c =max(0, c-1)
        elif action== 3:
            c= min(self.w-1, c+1)

        self.curr =(r, c)

        if self.curr in self.danger_zone:
            self.curr= self.start_pos
            return self.curr, -100,False
        elif self.curr ==self.end_pos:
            return self.curr, 10, True
        else:
            return self.curr,-1, False

    def get_all_states(self):
        s_list=[]
        for r in range(self.h):
            for c in range(self.w):
                s_list.append((r,c))
        return s_list

    def state_to_idx(self, state):
        return state[0] *self.w+ state[1]

    def idx_to_state(self, idx):
        return (idx// self.w, idx %self.w)

    def get_transitions(self, state, action):
        if state ==self.end_pos:
            return [(state, 1.0, 0,True)]

        r,c =state
        if action== 0:
            r= max(0, r-1)
        elif action ==1:
            r= min(self.h-1, r+1)
        elif action== 2:
            c =max(0, c-1)
        elif action ==3:
            c= min(self.w-1, c+1)

        nxt= (r,c)

        if nxt in self.danger_zone:
            return [(self.start_pos, 1.0, -100, False)]
        elif nxt== self.end_pos:
            return [(nxt, 1.0, 10,True)]
        else:
            return [(nxt, 1.0,-1, False)]


class FrozenLake:
    def __init__(self, size=4, holes=None, goal=None, slippery=True):
        if size ==4:
            self.env= gym.make('FrozenLake-v1', is_slippery=slippery, render_mode=None)
        else:
            self.env =gym.make('FrozenLake-v1', is_slippery=slippery, render_mode=None)

        self.sz =4
        self.slip= slippery
        self.moves= ['up', 'down','left', 'right']
        self.n_actions =4
        self.n_states= 16

        self.hole_locs =[(1,1), (1,3), (2,3),(3,0)]
        self.goal_loc= (3,3)

        self.curr_st =(0,0)

    def reset(self):
        o, _= self.env.reset()
        self.curr_st =self.idx_to_state(o)
        return self.curr_st

    def step(self, action):
        o,r, term, trunc, _ =self.env.step(action)
        finished= term or trunc
        self.curr_st= self.idx_to_state(o)

        if finished and r>0:
            r =10
        elif finished:
            r =-10
        else:
            r= -1

        return self.curr_st, r, finished

    def state_to_idx(self, state):
        if isinstance(state, tuple):
            return state[0]* self.sz +state[1]
        return state

    def idx_to_state(self, idx):
        return (idx //self.sz, idx% self.sz)

    def get_all_states(self):
        return [(r,c) for r in range(self.sz) for c in range(self.sz)]

    def get_transitions(self, state, action):
        if state in self.hole_locs or state== self.goal_loc:
            return [(state, 1.0, 0, True)]

        trans_list =[]
        if self.slip:
            for a_idx in range(4):
                prob= 0.7 if a_idx==action else 0.1
                ns =self._do_move(state, a_idx)
                if ns in self.hole_locs:
                    trans_list.append((ns, prob, -10, True))
                elif ns ==self.goal_loc:
                    trans_list.append((ns, prob, 10,True))
                else:
                    trans_list.append((ns, prob, -1, False))
        else:
            ns =self._do_move(state, action)
            if ns in self.hole_locs:
                return [(ns,1.0, -10, True)]
            elif ns== self.goal_loc:
                return [(ns, 1.0,10, True)]
            else:
                return [(ns, 1.0, -1,False)]

        return trans_list

    def _do_move(self, state, action):
        r, c= state
        if action ==0:
            r =max(0, r-1)
        elif action== 1:
            r= min(self.sz-1, r+1)
        elif action ==2:
            c= max(0, c-1)
        elif action== 3:
            c =min(self.sz-1, c+1)
        return (r,c)


class MountainCar:
    def __init__(self, pos_bins=40, vel_bins=40):
        self.env= gym.make('MountainCar-v0', render_mode=None)
        self.pos_b= pos_bins
        self.vel_b =vel_bins
        self.n_actions =3
        self.n_states= pos_bins* vel_bins
        self.moves =['push_left', 'no_push','push_right']

        self.p_lo =-1.2
        self.p_hi= 0.6
        self.v_lo =-0.07
        self.v_hi =0.07

        self.st =None

    def reset(self):
        obs, _ =self.env.reset()
        self.st= self._to_discrete(obs)
        return self.st

    def _to_discrete(self, obs):
        pos, vel= obs
        p_i =int((pos- self.p_lo)/ (self.p_hi -self.p_lo) * (self.pos_b-1))
        v_i= int((vel -self.v_lo) /(self.v_hi- self.v_lo)* (self.vel_b-1))
        p_i= max(0, min(self.pos_b-1, p_i))
        v_i =max(0, min(self.vel_b-1, v_i))
        return (p_i, v_i)

    def step(self, action):
        obs, rew,term, trunc, _ =self.env.step(action)
        finished= term or trunc
        self.st =self._to_discrete(obs)

        pos_val= obs[0]
        if finished and pos_val>= 0.5:
            rew= 100
        elif finished:
            rew =pos_val*50
        else:
            rew =-1+ (pos_val+ 1.2)* 0.5

        return self.st, rew, finished

    def state_to_idx(self, state):
        return state[0] *self.vel_b+ state[1]

    def idx_to_state(self, idx):
        return (idx// self.vel_b, idx %self.vel_b)

    def get_all_states(self):
        return [(p,v) for p in range(self.pos_b) for v in range(self.vel_b)]


class CartPole:
    def __init__(self):
        self.env= gym.make('CartPole-v1', render_mode=None)

        self.p_bins =10
        self.v_bins= 10
        self.a_bins =10
        self.av_bins= 10

        self.n_actions= 2
        self.n_states =self.p_bins* self.v_bins *self.a_bins *self.av_bins
        self.moves =['move_left', 'move_right']

        self.st= None

    def reset(self):
        obs, _= self.env.reset()
        self.st= self._to_bins(obs)
        return self.st

    def _to_bins(self, obs):
        x, xd, theta, td =obs

        pi= int((x +2.4)/ 4.8* (self.p_bins-1))
        vi =int((xd+ 3)/ 6 *(self.v_bins-1))
        ai= int((theta +0.21)/ 0.42 *(self.a_bins-1))
        avi =int((td+ 2) /4* (self.av_bins-1))

        pi =max(0, min(self.p_bins-1, pi))
        vi= max(0, min(self.v_bins-1, vi))
        ai =max(0, min(self.a_bins-1, ai))
        avi= max(0, min(self.av_bins-1, avi))

        return (pi, vi, ai,avi)

    def step(self, action):
        obs,rew, term, trunc, _ =self.env.step(action)
        finished= term or trunc
        self.st =self._to_bins(obs)

        if finished:
            rew =-10
        else:
            rew= 1

        return self.st, rew, finished

    def state_to_idx(self, state):
        p,v, a, av= state
        return p* self.v_bins* self.a_bins *self.av_bins+ \
               v *self.a_bins* self.av_bins +\
               a* self.av_bins +av

    def idx_to_state(self, idx):
        av =idx% self.av_bins
        idx //=self.av_bins
        a= idx %self.a_bins
        idx//= self.a_bins
        v= idx% self.v_bins
        p =idx// self.v_bins
        return (p,v, a, av)

    def get_all_states(self):
        st_list =[]
        for p in range(self.p_bins):
            for v in range(self.v_bins):
                for a in range(self.a_bins):
                    for av in range(self.av_bins):
                        st_list.append((p, v,a, av))
        return st_list

    def get_transitions(self, state, action):
        # note: simplified model for cartpole, use td methods for better results
        p,v, a, av= state
        if a<= 1 or a>= self.a_bins-2:
            return [(state, 1.0, -10,True)]

        return [(state, 1.0, 1, False)]


def get_environment(name, **kwargs):
    if name =='gridworld':
        return GridWorld(**kwargs)
    elif name== 'frozenlake':
        return FrozenLake(**kwargs)
    elif name =='cliffwalking':
        return CliffWalking()
    elif name== 'mountaincar':
        return MountainCar(**kwargs)
    elif name =='cartpole':
        return CartPole()
    else:
        return GridWorld()
