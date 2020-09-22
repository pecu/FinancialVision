import pickle
import matplotlib.pyplot as plt
import sys
from datetime import datetime, timedelta
plt.style.use('ggplot')
pip_unit = 1e4

def reset_dt(ts):
    ts_str = str(ts)
    y, d, m = ts_str.split('-')
    m, h = m.split(' ')
    if int(m) > 12:
        t = d
        d = m
        m = t
    return datetime.strptime('%s %s %s %s'%(y,m,d,h), '%Y %m %d %H:%M:%S')

with open(sys.argv[1],'rb') as f:
    data = pickle.load(f)
    
start_time = datetime.strptime('2018-08-01 00:00:00','%Y-%m-%d %H:%M:%S')
end_time = datetime.strptime('2018-11-30 23:00:00','%Y-%m-%d %H:%M:%S')

wt = 0
lt = 0
peak = -1e9
mdd = 0
profit = 0.0
loss = 0.0

new_plot = [0.0]
curr_time = start_time
order_i = 0
while(curr_time < end_time):
    this_profit = 0.0
    for o in range(order_i, order_i+5, 1):
        if o < len(data):
            if data[o]['activated'] and reset_dt(data[o]['end_time']) == curr_time:
                order_prof = round(data[o]['profit'], 5)
                this_profit += order_prof
                if order_prof > 0:
                    profit += order_prof
                    wt += 1
                else:
                    loss += order_prof
                    lt += 1
                order_i = o
    new_plot.append(new_plot[-1] + this_profit*pip_unit)
    if new_plot[-1] > peak:
        peak = new_plot[-1]
    elif new_plot[-1] < new_plot[-2]:
        dd = peak - new_plot[-1]
        mdd = max(mdd, dd)
    
    curr_time = curr_time + timedelta(hours=4)
        
print("Net Profit: %f"%new_plot[-1])
print("Win: %d"%wt)
print("Lose: %d"%lt)
print("total: %d"%(wt+lt))
print("Max Drawdown: %f:"%(mdd/new_plot[-1]))
print("Profit Factor: %f"%(profit/loss))
        
        
plt.plot(range(len(new_plot)), new_plot)
plt.xlabel("time tick (4hr.)")
plt.ylabel("pips")
plt.ylim(-500,3500)
plt.show()
with open('profit_history.pkl', 'wb') as f:
    pickle.dump(new_plot,f,protocol=-1)





