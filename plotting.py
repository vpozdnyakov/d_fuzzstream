from importlib import reload
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import d_fuzzstream
import imageio

data = pd.read_csv('https://raw.githubusercontent.com/vpozdnyakov/DS_Datasets/master/Synthetic/Non-Stationary/Bench1_11k/Benchmark1_11000.csv')

n = 500
datastream = data.iloc[:n]
datastream = datastream.rename(columns={'class': 'target'})
datastream = datastream.rename(columns={'X1': 'x', 'X2': 'y'})
datastream['last'] = pd.Series([False] * n)
datastream.iloc[[-1], [-1]] = True
datastream = datastream.fillna(3)

def plot(fig, ax):
    for i_fc in sum_s.to_dataframe().itertuples():
        fill = True if i_fc.N == 1 else False
        alpha = 0.05 if i_fc.N == 1 else 1    
        circle = plt.Circle((i_fc.x, i_fc.y), 
                            i_fc.radius, 
                            color='b', 
                            fill=fill, 
                            alpha=alpha)
        ax.add_artist(circle)
    sns_fig = sns.scatterplot(
        x="x",
        y="y",
        hue="target",
        style='last',
        data=datastream.iloc[:i_example], 
        legend=False, 
        palette="Set1").get_figure()
    sns_fig.savefig('gif/sample_plot{}.png'.format(i_example))

sum_s = d_fuzzstream.SummaryStructure(threshold=1, max_fmics=200)

i_example = 0
for _ in range(n):
    sum_s.clustering(datastream.iloc[[i_example]], test=True)
    i_example += 1
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_aspect(1)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    plot(fig, ax)

with imageio.get_writer('gif/ds_plot.gif', mode='I') as writer:
    for i in range(n):
        writer.append_data(imageio.imread('gif/sample_plot{}.png'.format(i+1)))
        writer.append_data(imageio.imread('gif/sample_plot{}.png'.format(i+1)))