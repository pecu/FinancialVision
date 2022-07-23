import matplotlib.pyplot as plt
import numpy as np

data_0 = np.load('menbershio_pros/member_probs_baseline.npy')
data_1 = np.load('menbershio_pros/member_probs_noise_01.npy')
data_2 = np.load('menbershio_pros/member_probs_noise_03.npy')
data_3 = np.load('menbershio_pros/member_probs_noise_05.npy')
data_4 = np.load('menbershio_pros/member_probs_noise_07.npy')
data_5 = np.load('menbershio_pros/member_probs_noise_1.npy')


figure, axes = plt.subplots(nrows=2,ncols=3)
figure.set_size_inches(25, 13)

axes[0,0].violinplot(data_0, vert=True, showmeans=True)
axes[0,1].violinplot(data_1, vert=True, showmeans=True) 
axes[0,2].violinplot(data_2, vert=True, showmeans=True)
axes[1,0].violinplot(data_3, vert=True, showmeans=True)
axes[1,1].violinplot(data_4, vert=True, showmeans=True)
axes[1,2].violinplot(data_5, vert=True, showmeans=True)


axes[0, 0].title.set_text("Baseline Model")
axes[0, 1].set_title("DP Model (Noise = 0.1)") 
axes[0, 2].set_title('DP Model (Noise = 0.3)')
axes[1, 0].set_title('DP Model (Noise = 0.5)')
axes[1, 1].set_title('DP Model (Noise = 0.7)')
axes[1, 2].set_title("DP Model (Noise = 1)")

plt.savefig('recognition_rate.png', dpi=300, bbox_inches='tight')

