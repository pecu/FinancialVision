import matplotlib.pyplot as plt
import pandas as pd
import pickle
import sys
from tqdm import trange

average_r = []
total_r = []
x = []
for i in trange(1500+1, ascii=True):
    with open(sys.argv[1]+'/trades/episode_%04d.pkl'%i,'rb') as f:
        data = pickle.load(f)
    s = 0
    p = []
    n = 0
    for d in data:
        if d['activated']:
            s += round(d['profit'],5)
            n += 1
        p.append(s)
    total_r.append(s)
    a = s/n
    average_r.append(a)
    x.append(i)
    
    if len(sys.argv) > 2:
        #aa = [a*2500 for _ in range(5500)]
        plt.plot(range(len(p)),p)
        #plt.plot(range(5500),aa, color='r', linestyle='--')
        #plt.xlim(0,1000)
        #plt.ylim(-0.25,0.25)
        #plt.savefig('_tmp/%04d.jpg'%i)        
        #plt.cla()
        plt.show()
    
    
plt.plot(range(len(average_r)), average_r)
plt.title("average rewards")
plt.show()
plt.plot(range(len(total_r)), total_r)
plt.title("total rewards")
plt.show()

output_data = {'episode':x, 'average_reward':average_r, 'total_reward':total_r}
pd.DataFrame(output_data).to_csv('train_reward-episode.csv', index=False)


