import park
import agent_class
import numpy as np

NR_EXP=102
SEED=42

#BB
env = park.make('abr_sim')
env.seed(SEED)
traces=env.all_traces
save_experiences_bb=[]
BB=agent_class.BB_agent(env.observation_space, env.action_space)
for i in range(0,NR_EXP):
    save_experience = [] #[nr_repr,sta,nr_repr,sta]
    obs = env.reset()
    done = False
    #first action
    obs, reward, done, info = env.step(BB.do_action())
    save_experience.append([info['stall_time'],info['bitrate']])
    while not done:
        ###agent BB rule####
        BB.take_obs_info(obs,info)
        BB.obs_to_state()
        act = BB.do_action()
        obs, reward, done, info = env.step(act)
        save_experience.append([info['stall_time'],info['bitrate']])
    BB.reset_state()
    save_experiences_bb.append(save_experience)

#TH
env = park.make('abr_sim')
env.seed(SEED)
traces=env.all_traces
save_experiences_th=[]
TH=agent_class.ThroughputRule_agent(env.observation_space, env.action_space)
for i in range(0,NR_EXP):
    save_experience = [] #[nr_repr,sta,nr_repr,sta]
    obs = env.reset()
    done = False
    #first action
    obs, reward, done, info = env.step(TH.do_action())
    save_experience.append([info['stall_time'],info['bitrate']])
    while not done:
        ###agent BB rule####
        TH.take_obs_info(obs,info)
        TH.obs_to_state()
        act = TH.do_action()
        obs, reward, done, info = env.step(act)
        save_experience.append([info['stall_time'],info['bitrate']])
    TH.reset_state()
    save_experiences_th.append(save_experience)

#MPC
env = park.make('abr_sim')
env.seed(SEED)
traces=env.all_traces
save_experiences_mpc=[]
MPC=agent_class.MPC_agent(env.observation_space, env.action_space)
for i in range(0,NR_EXP):
    save_experience = [] #[nr_repr,sta,nr_repr,sta]
    obs = env.reset()
    done = False
    #first action
    obs, reward, done, info = env.step(MPC.do_action())
    save_experience.append([info['stall_time'],info['bitrate']])
    while not done:
        ###agent MPC rule####
        MPC.take_obs_info(obs,info)
        MPC.obs_to_state()
        act = MPC.do_action()
        obs, reward, done, info = env.step(act)
        save_experience.append([info['stall_time'],info['bitrate']])
    MPC.reset_state()
    save_experiences_mpc.append(save_experience)

exp_name=['exp_th','exp_bb','exp_mpc']
exp_vect=[save_experiences_th,save_experiences_bb,save_experiences_mpc]
for i in range(3):
    np.save(exp_name[i],exp_vect[i])


