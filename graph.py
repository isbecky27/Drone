import matplotlib.pyplot as plt
import pandas as pd
import csv, argparse

def data_process(df):

    ## Sum of CPU 0 ~ 11 Load
    df['CPU Load (%)'] = df['CPU 0 Load (%)']
    for idx in range(1, 12):
        df['CPU Load (%)'] += df[f'CPU {idx} Load (%)'] 
    df['CPU Load (%)'] /= 12  

    # Used RAM (%)
    df['Used RAM (%)'] = (df['Used RAM (MB)'] / df['Total RAM (MB)']) * 100
    df['Used RAM (%)']

    # Current Power Consumption (mW)
    df['Power Consumption (mW)'] = 0
    for col in df.head()[1:]:
        if 'Current' in col:
            df['Power Consumption (mW)'] += df[col]
    df['Power Consumption (W)'] = df['Power Consumption (mW)'] * 0.001

    return df
    
def graph(csv_file, interval):

    df = pd.read_csv(csv_file, skiprows=1, header=0, index_col=0)
    df = data_process(df)

    fig, ax = plt.subplots(figsize=(12, 5))
    if interval == 0:
        plt.title(f'Tracking')
    elif interval == 1:
        plt.title(f'Detection')
    else:
        plt.title(f'Interval {interval} Frames for Detection')

    twin1 = ax.twinx()
    twin2 = ax.twinx()
    twin3 = ax.twinx()

    # Offset the right spine of twin2.  The ticks and label have already been
    # placed on the right by twinx above.
    # twin2.spines.right.set_position(("axes", 1.5))
    # twin1.spines.right.set_position(("axes", 1.2))

    p1, = ax.plot(df['Time (mS)'],  df['CPU Load (%)'], "b-", label="CPU Load(%)")
    p2, = twin1.plot(df['Time (mS)'], df['Used RAM (%)'], "r-", label="Used RAM (%)")
    p3, = twin2.plot(df['Time (mS)'], df['Used GR3D (%)'], "g-", label="Used GPU (%)")
    p4, = twin3.plot(df['Time (mS)'], df['Power Consumption (W)'], "y-", label="Power Consumption (W)")

    ax.set_ylim(0, 100)
    twin3.set_ylim(0, 60)

    ax.set_xlabel("Time (mS)")
    ax.set_ylabel("CPU Load(%) & Used RAM (%) & Used GPU (%)")
    twin1.get_yaxis().set_visible(False)
    twin2.get_yaxis().set_visible(False)
    twin3.set_ylabel("Power Consumption (W)")

    ax.yaxis.label.set_color(p1.get_color())
    twin3.yaxis.label.set_color(p4.get_color())

    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
    twin3.tick_params(axis='y', colors=p4.get_color(), **tkw)
    ax.tick_params(axis='x', **tkw)

    ax.legend(handles=[p1, p2, p3, p4], loc='upper left')

    plt.savefig(f'60W_{interval}.png')
    plt.show()

def scatter_plot(x, y):
    plt.figure()
    plt.title(f'{x} vs. {y}')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.plot(self.df.loc[:, x], self.df.loc[:, y])
    plt.savefig(f'{x} vs. {y}.png')

def plots():
    pairs = [('Time (mS)', col) for col in df.head()[1:]]
    for pair in pairs:
        scatter_plot(pair[0], pair[1])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, default='./output/UAV123/boat4_log_0.csv')
    parser.add_argument('--interval', type=int, default=0)
    args = parser.parse_args()

    graph(args.csv_file, args.interval)