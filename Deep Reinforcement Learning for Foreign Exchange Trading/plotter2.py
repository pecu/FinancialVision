import matplotlib.pyplot as plt
import pandas as pd
import pickle
import sys
from tqdm import trange

N = 100
const_path = "\\".join(sys.argv[1].split('\\')[:-2]) + '\\constant\\' + sys.argv[1].split('\\')[-1]
with open(const_path+'\\trades\\episode_0000.pkl','rb') as f:
    baseline_data = pickle.load(f)
    baseline = []
    acc = 0.0
    for d in baseline_data:
        if d['activated']:
            acc += round(d['profit'],5)
        baseline.append(acc)
    baseline_v = baseline[-1]

total_p = []
x = []
for i in trange(N+1, ascii=True):
    target = sys.argv[1]+'\\trades\\episode_%04d.pkl'%i
    with open(target,'rb') as f:
        data = pickle.load(f)
        
    s = 0
    p = []
    for d in data:
        if d['activated']:
            s += round(d['profit'],5)
        p.append(s)
    total_p.append(s)
    
    fig, (ax1,ax2) = plt.subplots(1,2)
    tmp_total_p = total_p.copy()
    #for _ in range(N-len(tmp_total_p)):
    #    tmp_total_p.append(None)
    ax1.plot(range(len(tmp_total_p)), tmp_total_p)
    ax1.axvline(x=i, color='green')
    ax1.axhline(y=baseline_v, color='purple', linestyle='--')
    #ax1.text(200, 0.27, 'baseline', va='center', ha='center', backgroundcolor='w')
    ax1.set_xlim(0,N)
    ax1.set_ylim(0.1,0.35)
    ax1.set_xlabel('episode')
    ax1.set_ylabel('net profit')
    ax2.plot(range(len(p)), p, color='blue')
    ax2.plot(range(len(baseline)), baseline, color='purple', linestyle=':')
    #ax2.axhline(y=baseline_v, color='purple', linestyle='--')
    #ax2.text(200, 0.27, 'baseline', va='center', ha='center', backgroundcolor='w')
    ax2.set_ylim(-0.1,0.35)
    ax2.set_xlim(-25,1025)
    ax2.set_xlabel('timestep')
    ax2.set_ylabel('accumulated return')
    plt.subplots_adjust(
        left  = 0.12,  # the left side of the subplots of the figure
        right = 0.98,    # the right side of the subplots of the figure
        bottom = 0.36,   # the bottom of the subplots of the figure
        top = 0.89,      # the top of the subplots of the figure
        wspace = 0.36,   # the amount of width reserved for space between subplots,
                       # expressed as a fraction of the average axis width
        hspace = 0.08   # the amount of height reserved for space between subplots,
    )
    fig.set_size_inches(8,5)
    plt.savefig('_tmp/%04d.jpg'%i)        
    #plt.show()
    plt.cla()
    plt.close()
    plt.clf()

import imageio, os
with imageio.get_writer('DQN_GBPUSD.gif', mode='I') as writer:
    for d in os.listdir('_tmp'):
        if "PPO" not in d and "DQN" not in d:
            #if int(d.split('.')[0])%15 == 0:
            image = imageio.imread('_tmp/'+d)
            writer.append_data(image)
    
    




