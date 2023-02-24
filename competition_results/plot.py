import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#KEY = 'Real Time Generation'
#KEY = 'Valid'
KEY = 'Relative Map Coverage'

data_bngai = pd.read_csv('BNGAI_stats_single.csv')
data_bngai['SUT'] = 'BeamNG.AI'
data_dave2 = pd.read_csv('DAVE2_stats_single.csv')
data_dave2['SUT'] = 'Dave-2'
data = pd.concat((data_bngai, data_dave2))
#data['AvgRoadLength'] = data['Simulated Time Execution'] / data['Valid']

plt.figure(figsize=(10, 7))
sns.set_theme(style="ticks", palette="deep")
sns.boxplot(x="Tool", y=KEY,
            hue='SUT', palette=["orange", "g"],
            data=data, whis=20000.0,
)
#sns.despine(offset=10, trim=True)
plt.show()
