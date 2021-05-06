from Free_Space.free_space_model import fspl_model, d11_and_phi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

h_RX = 50

# Experiment 1: distance vs amplitude (simulation vs. real-life data) at 5m height (Channel A)

##### 0 yard (front), height 5 m
# Amplitude (real data)
df_0f_5_amp = pd.read_csv("/Users/sang-hyunlee/Desktop/2021_Tarence_Data/Height_5/0y_front_5h_amp.csv", index_col=0)
amps_0f_5 = df_0f_5_amp.values.tolist()

# Phase (real data)
df_0f_5_phase = pd.read_csv("/Users/sang-hyunlee/Desktop/2021_Tarence_Data/Height_5/0y_front_5h_phase.csv", index_col=0)
phases_0f_5 = df_0f_5_phase.values.tolist()

# Amplitude & phase (simulation) : 4 antennas
_, amp_0f_5_sim, phase_0f_5_sim = fspl_model((29.71592, -95.40925), (29.71604, -95.40833), 5, 8, 0.039446375, 1.524,
                                             h_RX, 0.039446375, 0.08327568, 95, 70)

ant1_0f_5_amp_sim = amp_0f_5_sim[0][0]
ant2_0f_5_amp_sim = amp_0f_5_sim[1][1]
ant3_0f_5_amp_sim = amp_0f_5_sim[3][6]
ant4_0f_5_amp_sim = amp_0f_5_sim[4][7]

ant1_0f_5_phase_sim = phase_0f_5_sim[0][0]
ant2_0f_5_phase_sim = phase_0f_5_sim[1][1]
ant3_0f_5_phase_sim = phase_0f_5_sim[3][6]
ant4_0f_5_phase_sim = phase_0f_5_sim[4][7]

##### 0 yard (back), height 5 m
# Amplitude (real data)
df_0b_5_amp = pd.read_csv("/Users/sang-hyunlee/Desktop/2021_Tarence_Data/Height_5/0y_back_5h_amp.csv", index_col=0)
amps_0b_5 = df_0b_5_amp.values.tolist()

# Phase (real data)
df_0b_5_phase = pd.read_csv("/Users/sang-hyunlee/Desktop/2021_Tarence_Data/Height_5/0y_back_5h_phase.csv", index_col=0)
phases_0b_5 = df_0b_5_phase.values.tolist()

# Amplitude & phase (simulation) : 4 antennas
_, amp_0b_5_sim, phase_0b_5_sim = fspl_model((29.71592, -95.40938), (29.71604, -95.40833), 5, 8, 0.039446375, 1.524,
                                             h_RX, 0.039446375, 0.08327568, 95, 70)

ant1_0b_5_amp_sim = amp_0b_5_sim[0][0]
ant2_0b_5_amp_sim = amp_0b_5_sim[1][1]
ant3_0b_5_amp_sim = amp_0b_5_sim[3][6]
ant4_0b_5_amp_sim = amp_0b_5_sim[4][7]

ant1_0b_5_phase_sim = phase_0b_5_sim[0][0]
ant2_0b_5_phase_sim = phase_0b_5_sim[1][1]
ant3_0b_5_phase_sim = phase_0b_5_sim[3][6]
ant4_0b_5_phase_sim = phase_0b_5_sim[4][7]

##### 10 yard (front), height 5 m
# Amplitude (real data)
df_10f_5_amp = pd.read_csv("/Users/sang-hyunlee/Desktop/2021_Tarence_Data/Height_5/10y_front_5h_amp.csv", index_col=0)
amps_10f_5 = df_10f_5_amp.values.tolist()

# Phase (real data)
df_10f_5_phase = pd.read_csv("/Users/sang-hyunlee/Desktop/2021_Tarence_Data/Height_5/10y_front_5h_phase.csv",
                             index_col=0)
phases_10f_5 = df_10f_5_phase.values.tolist()

# Amplitude & phase (simulation) : 4 antennas
_, amp_10f_5_sim, phase_10f_5_sim = fspl_model((29.716, -95.40925), (29.71604, -95.40833), 5, 8, 0.039446375, 1.524,
                                               h_RX, 0.039446375, 0.08327568, 95, 70)

ant1_10f_5_amp_sim = amp_10f_5_sim[0][0]
ant2_10f_5_amp_sim = amp_10f_5_sim[1][1]
ant3_10f_5_amp_sim = amp_10f_5_sim[3][6]
ant4_10f_5_amp_sim = amp_10f_5_sim[4][7]

ant1_10f_5_phase_sim = phase_10f_5_sim[0][0]
ant2_10f_5_phase_sim = phase_10f_5_sim[1][1]
ant3_10f_5_phase_sim = phase_10f_5_sim[3][6]
ant4_10f_5_phase_sim = phase_10f_5_sim[4][7]

##### 10 yard (back), height 5 m
# Amplitude (real data)
df_10b_5_amp = pd.read_csv("/Users/sang-hyunlee/Desktop/2021_Tarence_Data/Height_5/10y_back_5h_amp.csv", index_col=0)
amps_10b_5 = df_10b_5_amp.values.tolist()

# Phase (real data)
df_10b_5_phase = pd.read_csv("/Users/sang-hyunlee/Desktop/2021_Tarence_Data/Height_5/10y_back_5h_phase.csv",
                             index_col=0)
phases_10b_5 = df_10b_5_phase.values.tolist()

# Amplitude & phase (simulation) : 4 antennas
_, amp_10b_5_sim, phase_10b_5_sim = fspl_model((29.716, -95.40938), (29.71604, -95.40833), 5, 8, 0.039446375, 1.524,
                                               h_RX, 0.039446375, 0.08327568, 95, 70)

ant1_10b_5_amp_sim = amp_10b_5_sim[0][0]
ant2_10b_5_amp_sim = amp_10b_5_sim[1][1]
ant3_10b_5_amp_sim = amp_10b_5_sim[3][6]
ant4_10b_5_amp_sim = amp_10b_5_sim[4][7]

ant1_10b_5_phase_sim = phase_10b_5_sim[0][0]
ant2_10b_5_phase_sim = phase_10b_5_sim[1][1]
ant3_10b_5_phase_sim = phase_10b_5_sim[3][6]
ant4_10b_5_phase_sim = phase_10b_5_sim[4][7]

##### 20 yard (front), height 5 m
# Amplitude (real data)
df_20f_5_amp = pd.read_csv("/Users/sang-hyunlee/Desktop/2021_Tarence_Data/Height_5/20y_front_5h_amp.csv", index_col=0)
amps_20f_5 = df_20f_5_amp.values.tolist()

# Phase (real data)
df_20f_5_phase = pd.read_csv("/Users/sang-hyunlee/Desktop/2021_Tarence_Data/Height_5/20y_front_5h_phase.csv",
                             index_col=0)
phases_20f_5 = df_20f_5_phase.values.tolist()

# Amplitude & phase (simulation) : 4 antennas
_, amp_20f_5_sim, phase_20f_5_sim = fspl_model((29.71608, -95.40925), (29.71604, -95.40833), 5, 8, 0.039446375, 1.524,
                                               h_RX, 0.039446375, 0.08327568, 95, 70)

ant1_20f_5_amp_sim = amp_20f_5_sim[0][0]
ant2_20f_5_amp_sim = amp_20f_5_sim[1][1]
ant3_20f_5_amp_sim = amp_20f_5_sim[3][6]
ant4_20f_5_amp_sim = amp_20f_5_sim[4][7]

ant1_20f_5_phase_sim = phase_20f_5_sim[0][0]
ant2_20f_5_phase_sim = phase_20f_5_sim[1][1]
ant3_20f_5_phase_sim = phase_20f_5_sim[3][6]
ant4_20f_5_phase_sim = phase_20f_5_sim[4][7]

## 20 yard (back), height 5 m
# Amplitude (real data)
df_20b_5_amp = pd.read_csv("/Users/sang-hyunlee/Desktop/2021_Tarence_Data/Height_5/20y_back_5h_amp.csv", index_col=0)
amps_20b_5 = df_20b_5_amp.values.tolist()

# Phase (real data)
df_20b_5_phase = pd.read_csv("/Users/sang-hyunlee/Desktop/2021_Tarence_Data/Height_5/20y_back_5h_phase.csv",
                             index_col=0)
phases_20b_5 = df_20b_5_phase.values.tolist()

# Amplitude & phase (simulation) : 4 antennas
_, amp_20b_5_sim, phase_20b_5_sim = fspl_model((29.71608, -95.40938), (29.71604, -95.40833), 5, 8, 0.039446375, 1.524,
                                               h_RX, 0.039446375, 0.08327568, 95, 70)

ant1_20b_5_amp_sim = amp_20b_5_sim[0][0]
ant2_20b_5_amp_sim = amp_20b_5_sim[1][1]
ant3_20b_5_amp_sim = amp_20b_5_sim[3][6]
ant4_20b_5_amp_sim = amp_20b_5_sim[4][7]

ant1_20b_5_phase_sim = phase_20b_5_sim[0][0]
ant2_20b_5_phase_sim = phase_20b_5_sim[1][1]
ant3_20b_5_phase_sim = phase_20b_5_sim[3][6]
ant4_20b_5_phase_sim = phase_20b_5_sim[4][7]

#xticklabels = ['0F', '0B', '10F', '10B', '20F', '20B']
xticklabels = [89.6112, 89.916, 90.5256, 101.803, 102.108, 102.718]

# Antenna 1 : Distance vs. amplitude at height 5 m (free space)

channel_a_ant1_0f_5_amp_mean = np.mean(np.array(amps_0f_5[14]))
channel_a_ant1_0f_5_amp_std = np.std(np.array(amps_0f_5[14]))
channel_a_ant1_0f_5_phase_mean = np.mean(np.array(phases_0f_5[14]))
channel_a_ant1_0f_5_phase_std = np.std(np.array(phases_0f_5[14]))

channel_a_ant1_0b_5_amp_mean = np.mean(np.array(amps_0b_5[14]))
channel_a_ant1_0b_5_amp_std = np.std(np.array(amps_0b_5[14]))
channel_a_ant1_0b_5_phase_mean = np.mean(np.array(phases_0b_5[14]))
channel_a_ant1_0b_5_phase_std = np.std(np.array(phases_0b_5[14]))

channel_a_ant1_10f_5_amp_mean = np.mean(np.array(amps_10f_5[14]))
channel_a_ant1_10f_5_amp_std = np.std(np.array(amps_10f_5[14]))
channel_a_ant1_10f_5_phase_mean = np.mean(np.array(phases_10f_5[14]))
channel_a_ant1_10f_5_phase_std = np.std(np.array(phases_10f_5[14]))

channel_a_ant1_10b_5_amp_mean = np.mean(np.array(amps_10b_5[14]))
channel_a_ant1_10b_5_amp_std = np.std(np.array(amps_10b_5[14]))
channel_a_ant1_10b_5_phase_mean = np.mean(np.array(phases_10b_5[14]))
channel_a_ant1_10b_5_phase_std = np.std(np.array(phases_10b_5[14]))

channel_a_ant1_20f_5_amp_mean = np.mean(np.array(amps_20f_5[14]))
channel_a_ant1_20f_5_amp_std = np.std(np.array(amps_20f_5[14]))
channel_a_ant1_20f_5_phase_mean = np.mean(np.array(phases_20f_5[14]))
channel_a_ant1_20f_5_phase_std = np.std(np.array(phases_20f_5[14]))

channel_a_ant1_20b_5_amp_mean = np.mean(np.array(amps_20b_5[14]))
channel_a_ant1_20b_5_amp_std = np.std(np.array(amps_20b_5[14]))
channel_a_ant1_20b_5_phase_mean = np.mean(np.array(phases_20b_5[14]))
channel_a_ant1_20b_5_phase_std = np.std(np.array(phases_20b_5[14]))

channel_b_ant1_0f_5_amp_mean = np.mean(np.array(amps_0f_5[15]))
channel_b_ant1_0f_5_amp_std = np.std(np.array(amps_0f_5[15]))
channel_b_ant1_0f_5_phase_mean = np.mean(np.array(phases_0f_5[15]))
channel_b_ant1_0f_5_phase_std = np.std(np.array(phases_0f_5[15]))

channel_b_ant1_0b_5_amp_mean = np.mean(np.array(amps_0b_5[15]))
channel_b_ant1_0b_5_amp_std = np.std(np.array(amps_0b_5[15]))
channel_b_ant1_0b_5_phase_mean = np.mean(np.array(phases_0b_5[15]))
channel_b_ant1_0b_5_phase_std = np.std(np.array(phases_0b_5[15]))

channel_b_ant1_10f_5_amp_mean = np.mean(np.array(amps_10f_5[15]))
channel_b_ant1_10f_5_amp_std = np.std(np.array(amps_10f_5[15]))
channel_b_ant1_10f_5_phase_mean = np.mean(np.array(phases_10f_5[15]))
channel_b_ant1_10f_5_phase_std = np.std(np.array(phases_10f_5[15]))

channel_b_ant1_10b_5_amp_mean = np.mean(np.array(amps_10b_5[15]))
channel_b_ant1_10b_5_amp_std = np.std(np.array(amps_10b_5[15]))
channel_b_ant1_10b_5_phase_mean = np.mean(np.array(phases_10b_5[15]))
channel_b_ant1_10b_5_phase_std = np.std(np.array(phases_10b_5[15]))

channel_b_ant1_20f_5_amp_mean = np.mean(np.array(amps_20f_5[15]))
channel_b_ant1_20f_5_amp_std = np.std(np.array(amps_20f_5[15]))
channel_b_ant1_20f_5_phase_mean = np.mean(np.array(phases_20f_5[15]))
channel_b_ant1_20f_5_phase_std = np.std(np.array(phases_20f_5[15]))

channel_b_ant1_20b_5_amp_mean = np.mean(np.array(amps_20b_5[15]))
channel_b_ant1_20b_5_amp_std = np.std(np.array(amps_20b_5[15]))
channel_b_ant1_20b_5_phase_mean = np.mean(np.array(phases_20b_5[15]))
channel_b_ant1_20b_5_phase_std = np.std(np.array(phases_20b_5[15]))

plt.figure(1)
plt.errorbar(xticklabels, [channel_a_ant1_10f_5_amp_mean, channel_a_ant1_20f_5_amp_mean, channel_a_ant1_0f_5_amp_mean,
                           channel_a_ant1_10b_5_amp_mean,
                           channel_a_ant1_20b_5_amp_mean, channel_a_ant1_0b_5_amp_mean],
             yerr=[channel_a_ant1_10f_5_amp_std, channel_a_ant1_20f_5_amp_std, channel_a_ant1_0f_5_amp_std,
                   channel_a_ant1_10b_5_amp_std, channel_a_ant1_20b_5_amp_std, channel_a_ant1_0b_5_amp_std],
             label='Experiment (Channel A)', ecolor="blue", fmt="bo--", capsize=5)

plt.errorbar(xticklabels, [channel_b_ant1_10f_5_amp_mean, channel_b_ant1_20f_5_amp_mean, channel_b_ant1_0f_5_amp_mean,
                           channel_b_ant1_10b_5_amp_mean,
                           channel_b_ant1_20b_5_amp_mean, channel_b_ant1_0b_5_amp_mean],
             yerr=[channel_b_ant1_10f_5_amp_std, channel_b_ant1_20f_5_amp_std, channel_b_ant1_0f_5_amp_std,
                   channel_b_ant1_10b_5_amp_std, channel_b_ant1_20b_5_amp_std, channel_b_ant1_0b_5_amp_std],
             label='Experiment (Channel B)', ecolor="green", fmt="go--", capsize=5)

plt.plot(xticklabels, [ant1_10f_5_amp_sim, ant1_20f_5_amp_sim, ant1_0f_5_amp_sim, ant1_10b_5_amp_sim, ant1_20b_5_amp_sim,
                       ant1_0b_5_amp_sim],
         label='Simulation', color="magenta", marker='o')  # plt.plot

#plt.xlim(89, 105)
# plt.ylim(0, 1.0)
legend_properties = {'size': 13}
plt.legend(prop=legend_properties, loc='upper left')
plt.xlabel("Distance (m)", fontsize=14, fontweight='bold')
plt.ylabel('Amplitude', fontsize=14, fontweight='bold')
plt.title("Distance vs. Amplitude at 5m Height for Antenna 1")
plt.grid("on")

plt.figure(2)
plt.errorbar(xticklabels, [channel_a_ant1_10f_5_phase_mean, channel_a_ant1_20f_5_phase_mean, channel_a_ant1_0f_5_phase_mean,
                           channel_a_ant1_10b_5_phase_mean,
                           channel_a_ant1_20b_5_phase_mean, channel_a_ant1_0b_5_phase_mean],
             yerr=[channel_a_ant1_10f_5_phase_std, channel_a_ant1_20f_5_phase_std, channel_a_ant1_0f_5_phase_std,
                   channel_a_ant1_10b_5_phase_std, channel_a_ant1_20b_5_phase_std, channel_a_ant1_0b_5_phase_std],
             label='Experiment (Channel A)', ecolor="blue", fmt="bo--", capsize=5)

plt.errorbar(xticklabels, [channel_b_ant1_10f_5_phase_mean, channel_b_ant1_20f_5_phase_mean, channel_b_ant1_0f_5_phase_mean,
                           channel_b_ant1_10b_5_phase_mean,
                           channel_b_ant1_20b_5_phase_mean, channel_b_ant1_0b_5_phase_mean],
             yerr=[channel_b_ant1_10f_5_phase_std, channel_b_ant1_20f_5_phase_std, channel_b_ant1_0f_5_phase_std,
                   channel_b_ant1_10b_5_phase_std, channel_b_ant1_20b_5_phase_std, channel_b_ant1_0b_5_phase_std],
             label='Experiment (Channel B)', ecolor="green", fmt="go--", capsize=5)

plt.plot(xticklabels, [ant1_10f_5_phase_sim, ant1_20f_5_phase_sim, ant1_0f_5_phase_sim, ant1_10b_5_phase_sim, ant1_20b_5_phase_sim,
                       ant1_0b_5_phase_sim],
         label='Simulation', color="magenta", marker='o')  # plt.plot

#plt.xlim(89, 105)
# plt.ylim(0, 1.0)
legend_properties = {'size': 13}
plt.legend(prop=legend_properties, loc='upper left')
plt.xlabel("Distance (m)", fontsize=14, fontweight='bold')
plt.ylabel('Phase (degrees)', fontsize=14, fontweight='bold')
plt.title("Distance vs. Phase at 5m Height for Antenna 1")
plt.grid("on")


# Antenna 2 : Distance vs. amplitude at height 5 m (free space)
plt.figure(3)

channel_a_ant2_0f_5_amp_mean = np.mean(np.array(amps_0f_5[28]))
channel_a_ant2_0f_5_amp_std = np.std(np.array(amps_0f_5[28]))
channel_a_ant2_0f_5_phase_mean = np.mean(np.array(phases_0f_5[28]))
channel_a_ant2_0f_5_phase_std = np.std(np.array(phases_0f_5[28]))

channel_a_ant2_0b_5_amp_mean = np.mean(np.array(amps_0b_5[28]))
channel_a_ant2_0b_5_amp_std = np.std(np.array(amps_0b_5[28]))
channel_a_ant2_0b_5_phase_mean = np.mean(np.array(phases_0b_5[28]))
channel_a_ant2_0b_5_phase_std = np.std(np.array(phases_0b_5[28]))

channel_a_ant2_10f_5_amp_mean = np.mean(np.array(amps_10f_5[28]))
channel_a_ant2_10f_5_amp_std = np.std(np.array(amps_10f_5[28]))
channel_a_ant2_10f_5_phase_mean = np.mean(np.array(phases_10f_5[28]))
channel_a_ant2_10f_5_phase_std = np.std(np.array(phases_10f_5[28]))

channel_a_ant2_10b_5_amp_mean = np.mean(np.array(amps_10b_5[28]))
channel_a_ant2_10b_5_amp_std = np.std(np.array(amps_10b_5[28]))
channel_a_ant2_10b_5_phase_mean = np.mean(np.array(phases_10b_5[28]))
channel_a_ant2_10b_5_phase_std = np.std(np.array(phases_10b_5[28]))

channel_a_ant2_20f_5_amp_mean = np.mean(np.array(amps_20f_5[28]))
channel_a_ant2_20f_5_amp_std = np.std(np.array(amps_20f_5[28]))
channel_a_ant2_20f_5_phase_mean = np.mean(np.array(phases_20f_5[28]))
channel_a_ant2_20f_5_phase_std = np.std(np.array(phases_20f_5[28]))

channel_a_ant2_20b_5_amp_mean = np.mean(np.array(amps_20b_5[28]))
channel_a_ant2_20b_5_amp_std = np.std(np.array(amps_20b_5[28]))
channel_a_ant2_20b_5_phase_mean = np.mean(np.array(phases_20b_5[28]))
channel_a_ant2_20b_5_phase_std = np.std(np.array(phases_20b_5[28]))

channel_b_ant2_0f_5_amp_mean = np.mean(np.array(amps_0f_5[29]))
channel_b_ant2_0f_5_amp_std = np.std(np.array(amps_0f_5[29]))
channel_b_ant2_0f_5_phase_mean = np.mean(np.array(phases_0f_5[29]))
channel_b_ant2_0f_5_phase_std = np.std(np.array(phases_0f_5[29]))

channel_b_ant2_0b_5_amp_mean = np.mean(np.array(amps_0b_5[29]))
channel_b_ant2_0b_5_amp_std = np.std(np.array(amps_0b_5[29]))
channel_b_ant2_0b_5_phase_mean = np.mean(np.array(phases_0b_5[29]))
channel_b_ant2_0b_5_phase_std = np.std(np.array(phases_0b_5[29]))

channel_b_ant2_10f_5_amp_mean = np.mean(np.array(amps_10f_5[29]))
channel_b_ant2_10f_5_amp_std = np.std(np.array(amps_10f_5[29]))
channel_b_ant2_10f_5_phase_mean = np.mean(np.array(phases_10f_5[29]))
channel_b_ant2_10f_5_phase_std = np.std(np.array(phases_10f_5[29]))

channel_b_ant2_10b_5_amp_mean = np.mean(np.array(amps_10b_5[29]))
channel_b_ant2_10b_5_amp_std = np.std(np.array(amps_10b_5[29]))
channel_b_ant2_10b_5_phase_mean = np.mean(np.array(phases_10b_5[29]))
channel_b_ant2_10b_5_phase_std = np.std(np.array(phases_10b_5[29]))

channel_b_ant2_20f_5_amp_mean = np.mean(np.array(amps_20f_5[29]))
channel_b_ant2_20f_5_amp_std = np.std(np.array(amps_20f_5[29]))
channel_b_ant2_20f_5_phase_mean = np.mean(np.array(phases_20f_5[29]))
channel_b_ant2_20f_5_phase_std = np.std(np.array(phases_20f_5[29]))

channel_b_ant2_20b_5_amp_mean = np.mean(np.array(amps_20b_5[29]))
channel_b_ant2_20b_5_amp_std = np.std(np.array(amps_20b_5[29]))
channel_b_ant2_20b_5_phase_mean = np.mean(np.array(phases_20b_5[29]))
channel_b_ant2_20b_5_phase_std = np.std(np.array(phases_20b_5[29]))

plt.errorbar(xticklabels, [channel_a_ant2_10f_5_amp_mean, channel_a_ant2_20f_5_amp_mean, channel_a_ant2_0f_5_amp_mean,
                           channel_a_ant2_10b_5_amp_mean,
                           channel_a_ant2_20b_5_amp_mean, channel_a_ant2_0b_5_amp_mean],
             yerr=[channel_a_ant2_10f_5_amp_std, channel_a_ant2_20f_5_amp_std, channel_a_ant2_0f_5_amp_std,
                   channel_a_ant2_10b_5_amp_std, channel_a_ant2_20b_5_amp_std, channel_a_ant2_0b_5_amp_std],
             label='Experiment (Channel A)', ecolor="blue", fmt="bo--", capsize=5)

plt.errorbar(xticklabels, [channel_b_ant2_10f_5_amp_mean, channel_b_ant2_20f_5_amp_mean, channel_b_ant2_0f_5_amp_mean,
                           channel_b_ant2_10b_5_amp_mean,
                           channel_b_ant2_20b_5_amp_mean, channel_b_ant2_0b_5_amp_mean],
             yerr=[channel_b_ant2_10f_5_amp_std, channel_b_ant2_20f_5_amp_std, channel_b_ant2_0f_5_amp_std,
                   channel_b_ant2_10b_5_amp_std, channel_b_ant2_20b_5_amp_std, channel_b_ant2_0b_5_amp_std],
             label='Experiment (Channel B)', ecolor="green", fmt="go--", capsize=5)

plt.plot(xticklabels, [ant2_10f_5_amp_sim, ant2_20f_5_amp_sim, ant2_0f_5_amp_sim, ant2_10b_5_amp_sim, ant2_20b_5_amp_sim,
                       ant2_0b_5_amp_sim],
         label='Simulation', color="magenta", marker='o')  # plt.plot

#plt.xlim(89, 105)
# plt.ylim(0, 1.0)
legend_properties = {'size': 13}
plt.legend(prop=legend_properties, loc='upper left')
plt.xlabel("Distance (m)", fontsize=14, fontweight='bold')
plt.ylabel('Amplitude', fontsize=14, fontweight='bold')
plt.title("Distance vs. Amplitude at 5m Height for Antenna 2")
plt.grid("on")


plt.figure(4)
plt.errorbar(xticklabels, [channel_a_ant2_10f_5_phase_mean, channel_a_ant2_20f_5_phase_mean, channel_a_ant2_0f_5_phase_mean,
                           channel_a_ant2_10b_5_phase_mean,
                           channel_a_ant2_20b_5_phase_mean, channel_a_ant2_0b_5_phase_mean],
             yerr=[channel_a_ant2_10f_5_phase_std, channel_a_ant2_20f_5_phase_std, channel_a_ant2_0f_5_phase_std,
                   channel_a_ant2_10b_5_phase_std, channel_a_ant2_20b_5_phase_std, channel_a_ant2_0b_5_phase_std],
             label='Experiment (Channel A)', ecolor="blue", fmt="bo--", capsize=5)

plt.errorbar(xticklabels, [channel_b_ant2_10f_5_phase_mean, channel_b_ant2_20f_5_phase_mean, channel_b_ant2_0f_5_phase_mean,
                           channel_b_ant2_10b_5_phase_mean,
                           channel_b_ant2_20b_5_phase_mean, channel_b_ant2_0b_5_phase_mean],
             yerr=[channel_b_ant2_10f_5_phase_std, channel_b_ant2_20f_5_phase_std, channel_b_ant2_0f_5_phase_std,
                   channel_b_ant2_10b_5_phase_std, channel_b_ant2_20b_5_phase_std, channel_b_ant2_0b_5_phase_std],
             label='Experiment (Channel B)', ecolor="green", fmt="go--", capsize=5)

plt.plot(xticklabels, [ant2_10f_5_phase_sim, ant2_20f_5_phase_sim, ant2_0f_5_phase_sim, ant2_10b_5_phase_sim, ant2_20b_5_phase_sim,
                       ant2_0b_5_phase_sim],
         label='Simulation', color="magenta", marker='o')  # plt.plot

#plt.xlim(89, 105)
# plt.ylim(0, 1.0)
legend_properties = {'size': 13}
plt.legend(prop=legend_properties, loc='upper left')
plt.xlabel("Distance (m)", fontsize=14, fontweight='bold')
plt.ylabel('Phase (degrees)', fontsize=14, fontweight='bold')
plt.title("Distance vs. Phase at 5m Height for Antenna 2")
plt.grid("on")


# Antenna 3 : Distance vs. amplitude at height 5 m (free space)

channel_a_ant3_0f_5_amp_mean = np.mean(np.array(amps_0f_5[48]))
channel_a_ant3_0f_5_amp_std = np.std(np.array(amps_0f_5[48]))
channel_a_ant3_0f_5_phase_mean = np.mean(np.array(phases_0f_5[48]))
channel_a_ant3_0f_5_phase_std = np.std(np.array(phases_0f_5[48]))

channel_a_ant3_0b_5_amp_mean = np.mean(np.array(amps_0b_5[48]))
channel_a_ant3_0b_5_amp_std = np.std(np.array(amps_0b_5[48]))
channel_a_ant3_0b_5_phase_mean = np.mean(np.array(phases_0b_5[48]))
channel_a_ant3_0b_5_phase_std = np.std(np.array(phases_0b_5[48]))

channel_a_ant3_10f_5_amp_mean = np.mean(np.array(amps_10f_5[48]))
channel_a_ant3_10f_5_amp_std = np.std(np.array(amps_10f_5[48]))
channel_a_ant3_10f_5_phase_mean = np.mean(np.array(phases_10f_5[48]))
channel_a_ant3_10f_5_phase_std = np.std(np.array(phases_10f_5[48]))

channel_a_ant3_10b_5_amp_mean = np.mean(np.array(amps_10b_5[48]))
channel_a_ant3_10b_5_amp_std = np.std(np.array(amps_10b_5[48]))
channel_a_ant3_10b_5_phase_mean = np.mean(np.array(phases_10b_5[48]))
channel_a_ant3_10b_5_phase_std = np.std(np.array(phases_10b_5[48]))

channel_a_ant3_20f_5_amp_mean = np.mean(np.array(amps_20f_5[48]))
channel_a_ant3_20f_5_amp_std = np.std(np.array(amps_20f_5[48]))
channel_a_ant3_20f_5_phase_mean = np.mean(np.array(phases_20f_5[48]))
channel_a_ant3_20f_5_phase_std = np.std(np.array(phases_20f_5[48]))

channel_a_ant3_20b_5_amp_mean = np.mean(np.array(amps_20b_5[48]))
channel_a_ant3_20b_5_amp_std = np.std(np.array(amps_20b_5[48]))
channel_a_ant3_20b_5_phase_mean = np.mean(np.array(phases_20b_5[48]))
channel_a_ant3_20b_5_phase_std = np.std(np.array(phases_20b_5[48]))

channel_b_ant3_0f_5_amp_mean = np.mean(np.array(amps_0f_5[49]))
channel_b_ant3_0f_5_amp_std = np.std(np.array(amps_0f_5[49]))
channel_b_ant3_0f_5_phase_mean = np.mean(np.array(phases_0f_5[49]))
channel_b_ant3_0f_5_phase_std = np.std(np.array(phases_0f_5[49]))

channel_b_ant3_0b_5_amp_mean = np.mean(np.array(amps_0b_5[49]))
channel_b_ant3_0b_5_amp_std = np.std(np.array(amps_0b_5[49]))
channel_b_ant3_0b_5_phase_mean = np.mean(np.array(phases_0b_5[49]))
channel_b_ant3_0b_5_phase_std = np.std(np.array(phases_0b_5[49]))

channel_b_ant3_10f_5_amp_mean = np.mean(np.array(amps_10f_5[49]))
channel_b_ant3_10f_5_amp_std = np.std(np.array(amps_10f_5[49]))
channel_b_ant3_10f_5_phase_mean = np.mean(np.array(phases_10f_5[49]))
channel_b_ant3_10f_5_phase_std = np.std(np.array(phases_10f_5[49]))

channel_b_ant3_10b_5_amp_mean = np.mean(np.array(amps_10b_5[49]))
channel_b_ant3_10b_5_amp_std = np.std(np.array(amps_10b_5[49]))
channel_b_ant3_10b_5_phase_mean = np.mean(np.array(phases_10b_5[49]))
channel_b_ant3_10b_5_phase_std = np.std(np.array(phases_10b_5[49]))

channel_b_ant3_20f_5_amp_mean = np.mean(np.array(amps_20f_5[49]))
channel_b_ant3_20f_5_amp_std = np.std(np.array(amps_20f_5[49]))
channel_b_ant3_20f_5_phase_mean = np.mean(np.array(phases_20f_5[49]))
channel_b_ant3_20f_5_phase_std = np.std(np.array(phases_20f_5[49]))

channel_b_ant3_20b_5_amp_mean = np.mean(np.array(amps_20b_5[49]))
channel_b_ant3_20b_5_amp_std = np.std(np.array(amps_20b_5[49]))
channel_b_ant3_20b_5_phase_mean = np.mean(np.array(phases_20b_5[49]))
channel_b_ant3_20b_5_phase_std = np.std(np.array(phases_20b_5[49]))

plt.figure(5)
plt.errorbar(xticklabels, [channel_a_ant3_10f_5_amp_mean, channel_a_ant3_20f_5_amp_mean, channel_a_ant3_0f_5_amp_mean,
                           channel_a_ant3_10b_5_amp_mean,
                           channel_a_ant3_20b_5_amp_mean, channel_a_ant3_0b_5_amp_mean],
             yerr=[channel_a_ant3_10f_5_amp_std, channel_a_ant3_20f_5_amp_std, channel_a_ant3_0f_5_amp_std,
                   channel_a_ant3_10b_5_amp_std, channel_a_ant3_20b_5_amp_std, channel_a_ant3_0b_5_amp_std],
             label='Experiment (Channel A)', ecolor="blue", fmt="bo--", capsize=5)

plt.errorbar(xticklabels, [channel_b_ant3_10f_5_amp_mean, channel_b_ant3_20f_5_amp_mean, channel_b_ant3_0f_5_amp_mean,
                           channel_b_ant3_10b_5_amp_mean,
                           channel_b_ant3_20b_5_amp_mean, channel_b_ant3_0b_5_amp_mean],
             yerr=[channel_b_ant3_10f_5_amp_std, channel_b_ant3_20f_5_amp_std, channel_b_ant3_0f_5_amp_std,
                   channel_b_ant3_10b_5_amp_std, channel_b_ant3_20b_5_amp_std, channel_b_ant3_0b_5_amp_std],
             label='Experiment (Channel B)', ecolor="green", fmt="go--", capsize=5)

plt.plot(xticklabels, [ant3_10f_5_amp_sim, ant3_20f_5_amp_sim, ant3_0f_5_amp_sim, ant3_10b_5_amp_sim, ant3_20b_5_amp_sim,
                       ant3_0b_5_amp_sim],
         label='Simulation', color="magenta", marker='o')  # plt.plot

#plt.xlim(89, 105)
# plt.ylim(0, 1.0)
legend_properties = {'size': 13}
plt.legend(prop=legend_properties, loc='upper left')
plt.xlabel("Distance (m)", fontsize=14, fontweight='bold')
plt.ylabel('Amplitude', fontsize=14, fontweight='bold')
plt.title("Distance vs. Amplitude at 5m Height for Antenna 3")
plt.grid("on")

plt.figure(6)
plt.errorbar(xticklabels, [channel_a_ant3_10f_5_phase_mean, channel_a_ant3_20f_5_phase_mean, channel_a_ant3_0f_5_phase_mean,
                           channel_a_ant3_10b_5_phase_mean,
                           channel_a_ant3_20b_5_phase_mean, channel_a_ant3_0b_5_phase_mean],
             yerr=[channel_a_ant3_10f_5_phase_std, channel_a_ant3_20f_5_phase_std, channel_a_ant3_0f_5_phase_std,
                   channel_a_ant3_10b_5_phase_std, channel_a_ant3_20b_5_phase_std, channel_a_ant3_0b_5_phase_std],
             label='Experiment (Channel A)', ecolor="blue", fmt="bo--", capsize=5)

plt.errorbar(xticklabels, [channel_b_ant3_10f_5_phase_mean, channel_b_ant3_20f_5_phase_mean, channel_b_ant3_0f_5_phase_mean,
                           channel_b_ant3_10b_5_phase_mean,
                           channel_b_ant3_20b_5_phase_mean, channel_b_ant3_0b_5_phase_mean],
             yerr=[channel_b_ant3_10f_5_phase_std, channel_b_ant3_20f_5_phase_std, channel_b_ant3_0f_5_phase_std,
                   channel_b_ant3_10b_5_phase_std, channel_b_ant3_20b_5_phase_std, channel_b_ant3_0b_5_phase_std],
             label='Experiment (Channel B)', ecolor="green", fmt="go--", capsize=5)

plt.plot(xticklabels, [ant3_10f_5_phase_sim, ant3_20f_5_phase_sim, ant3_0f_5_phase_sim, ant3_10b_5_phase_sim, ant3_20b_5_phase_sim,
                       ant3_0b_5_phase_sim],
         label='Simulation', color="magenta", marker='o')  # plt.plot

#plt.xlim(89, 105)
# plt.ylim(0, 1.0)
legend_properties = {'size': 13}
plt.legend(prop=legend_properties, loc='upper left')
plt.xlabel("Distance (m)", fontsize=14, fontweight='bold')
plt.ylabel('Phase (degrees)', fontsize=14, fontweight='bold')
plt.title("Distance vs. Phase at 5m Height for Antenna 3")
plt.grid("on")


# Antenna 4 : Distance vs. amplitude at height 5 m (free space)

channel_a_ant4_0f_5_amp_mean = np.mean(np.array(amps_0f_5[62]))
channel_a_ant4_0f_5_amp_std = np.std(np.array(amps_0f_5[62]))
channel_a_ant4_0f_5_phase_mean = np.mean(np.array(phases_0f_5[62]))
channel_a_ant4_0f_5_phase_std = np.std(np.array(phases_0f_5[62]))

channel_a_ant4_0b_5_amp_mean = np.mean(np.array(amps_0b_5[62]))
channel_a_ant4_0b_5_amp_std = np.std(np.array(amps_0b_5[62]))
channel_a_ant4_0b_5_phase_mean = np.mean(np.array(phases_0b_5[62]))
channel_a_ant4_0b_5_phase_std = np.std(np.array(phases_0b_5[62]))

channel_a_ant4_10f_5_amp_mean = np.mean(np.array(amps_10f_5[62]))
channel_a_ant4_10f_5_amp_std = np.std(np.array(amps_10f_5[62]))
channel_a_ant4_10f_5_phase_mean = np.mean(np.array(phases_10f_5[62]))
channel_a_ant4_10f_5_phase_std = np.std(np.array(phases_10f_5[62]))

channel_a_ant4_10b_5_amp_mean = np.mean(np.array(amps_10b_5[62]))
channel_a_ant4_10b_5_amp_std = np.std(np.array(amps_10b_5[62]))
channel_a_ant4_10b_5_phase_mean = np.mean(np.array(phases_10b_5[62]))
channel_a_ant4_10b_5_phase_std = np.std(np.array(phases_10b_5[62]))

channel_a_ant4_20f_5_amp_mean = np.mean(np.array(amps_20f_5[62]))
channel_a_ant4_20f_5_amp_std = np.std(np.array(amps_20f_5[62]))
channel_a_ant4_20f_5_phase_mean = np.mean(np.array(phases_20f_5[62]))
channel_a_ant4_20f_5_phase_std = np.std(np.array(phases_20f_5[62]))

channel_a_ant4_20b_5_amp_mean = np.mean(np.array(amps_20b_5[62]))
channel_a_ant4_20b_5_amp_std = np.std(np.array(amps_20b_5[62]))
channel_a_ant4_20b_5_phase_mean = np.mean(np.array(phases_20b_5[62]))
channel_a_ant4_20b_5_phase_std = np.std(np.array(phases_20b_5[62]))

channel_b_ant4_0f_5_amp_mean = np.mean(np.array(amps_0f_5[63]))
channel_b_ant4_0f_5_amp_std = np.std(np.array(amps_0f_5[63]))
channel_b_ant4_0f_5_phase_mean = np.mean(np.array(phases_0f_5[63]))
channel_b_ant4_0f_5_phase_std = np.std(np.array(phases_0f_5[63]))

channel_b_ant4_0b_5_amp_mean = np.mean(np.array(amps_0b_5[63]))
channel_b_ant4_0b_5_amp_std = np.std(np.array(amps_0b_5[63]))
channel_b_ant4_0b_5_phase_mean = np.mean(np.array(phases_0b_5[63]))
channel_b_ant4_0b_5_phase_std = np.std(np.array(phases_0b_5[63]))

channel_b_ant4_10f_5_amp_mean = np.mean(np.array(amps_10f_5[63]))
channel_b_ant4_10f_5_amp_std = np.std(np.array(amps_10f_5[63]))
channel_b_ant4_10f_5_phase_mean = np.mean(np.array(phases_10f_5[63]))
channel_b_ant4_10f_5_phase_std = np.std(np.array(phases_10f_5[63]))

channel_b_ant4_10b_5_amp_mean = np.mean(np.array(amps_10b_5[63]))
channel_b_ant4_10b_5_amp_std = np.std(np.array(amps_10b_5[63]))
channel_b_ant4_10b_5_phase_mean = np.mean(np.array(phases_10b_5[63]))
channel_b_ant4_10b_5_phase_std = np.std(np.array(phases_10b_5[63]))

channel_b_ant4_20f_5_amp_mean = np.mean(np.array(amps_20f_5[63]))
channel_b_ant4_20f_5_amp_std = np.std(np.array(amps_20f_5[63]))
channel_b_ant4_20f_5_phase_mean = np.mean(np.array(phases_20f_5[63]))
channel_b_ant4_20f_5_phase_std = np.std(np.array(phases_20f_5[63]))

channel_b_ant4_20b_5_amp_mean = np.mean(np.array(amps_20b_5[63]))
channel_b_ant4_20b_5_amp_std = np.std(np.array(amps_20b_5[63]))
channel_b_ant4_20b_5_phase_mean = np.mean(np.array(phases_20b_5[63]))
channel_b_ant4_20b_5_phase_std = np.std(np.array(phases_20b_5[63]))

plt.figure(7)
plt.errorbar(xticklabels, [channel_a_ant4_10f_5_amp_mean, channel_a_ant4_20f_5_amp_mean, channel_a_ant4_0f_5_amp_mean,
                           channel_a_ant4_10b_5_amp_mean,
                           channel_a_ant4_20b_5_amp_mean, channel_a_ant4_0b_5_amp_mean],
             yerr=[channel_a_ant4_10f_5_amp_std, channel_a_ant4_20f_5_amp_std, channel_a_ant4_0f_5_amp_std,
                   channel_a_ant4_10b_5_amp_std, channel_a_ant4_20b_5_amp_std, channel_a_ant4_0b_5_amp_std],
             label='Experiment (Channel A)', ecolor="blue", fmt="bo--", capsize=5)

plt.errorbar(xticklabels, [channel_b_ant4_10f_5_amp_mean, channel_b_ant4_20f_5_amp_mean, channel_b_ant4_0f_5_amp_mean,
                           channel_b_ant4_10b_5_amp_mean,
                           channel_b_ant4_20b_5_amp_mean, channel_b_ant4_0b_5_amp_mean],
             yerr=[channel_b_ant4_10f_5_amp_std, channel_b_ant4_20f_5_amp_std, channel_b_ant4_0f_5_amp_std,
                   channel_b_ant4_10b_5_amp_std, channel_b_ant4_20b_5_amp_std, channel_b_ant4_0b_5_amp_std],
             label='Experiment (Channel B)', ecolor="green", fmt="go--", capsize=5)

plt.plot(xticklabels, [ant4_10f_5_amp_sim, ant4_20f_5_amp_sim, ant4_0f_5_amp_sim, ant4_10b_5_amp_sim, ant4_20b_5_amp_sim,
                       ant4_0b_5_amp_sim],
         label='Simulation', color="magenta", marker='o')  # plt.plot

#plt.xlim(89, 105)
# plt.ylim(0, 1.0)
legend_properties = {'size': 13}
plt.legend(prop=legend_properties, loc='upper left')
plt.xlabel("Distance (m)", fontsize=14, fontweight='bold')
plt.ylabel('Amplitude', fontsize=14, fontweight='bold')
plt.title("Distance vs. Amplitude at 5m Height for Antenna 4")
plt.grid("on")

plt.figure(8)
plt.errorbar(xticklabels, [channel_a_ant4_10f_5_phase_mean, channel_a_ant4_20f_5_phase_mean, channel_a_ant4_0f_5_phase_mean,
                           channel_a_ant4_10b_5_phase_mean,
                           channel_a_ant4_20b_5_phase_mean, channel_a_ant4_0b_5_phase_mean],
             yerr=[channel_a_ant4_10f_5_phase_std, channel_a_ant4_20f_5_phase_std, channel_a_ant4_0f_5_phase_std,
                   channel_a_ant4_10b_5_phase_std, channel_a_ant4_20b_5_phase_std, channel_a_ant4_0b_5_phase_std],
             label='Experiment (Channel A)', ecolor="blue", fmt="bo--", capsize=5)

plt.errorbar(xticklabels, [channel_b_ant4_10f_5_phase_mean, channel_b_ant4_20f_5_phase_mean, channel_b_ant4_0f_5_phase_mean,
                           channel_b_ant4_10b_5_phase_mean,
                           channel_b_ant4_20b_5_phase_mean, channel_b_ant4_0b_5_phase_mean],
             yerr=[channel_b_ant4_10f_5_phase_std, channel_b_ant4_20f_5_phase_std, channel_b_ant4_0f_5_phase_std,
                   channel_b_ant4_10b_5_phase_std, channel_b_ant4_20b_5_phase_std, channel_b_ant4_0b_5_phase_std],
             label='Experiment (Channel B)', ecolor="green", fmt="go--", capsize=5)

plt.plot(xticklabels, [ant4_10f_5_phase_sim, ant4_20f_5_phase_sim, ant4_0f_5_phase_sim, ant4_10b_5_phase_sim, ant4_20b_5_phase_sim,
                       ant4_0b_5_phase_sim],
         label='Simulation', color="magenta", marker='o')  # plt.plot

#plt.xlim(89, 105)
# plt.ylim(0, 1.0)
legend_properties = {'size': 13}
plt.legend(prop=legend_properties, loc='upper left')
plt.xlabel("Distance (m)", fontsize=14, fontweight='bold')
plt.ylabel('Phase (degrees)', fontsize=14, fontweight='bold')
plt.title("Distance vs. Phase at 5m Height for Antenna 4")
plt.grid("on")

plt.show()