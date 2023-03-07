import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def create_ridge_plot(dataframe):
    # Initialize the FacetGrid object
    #pal = sns.cubehelix_palette(len(dataframe["Gamma"].unique().tolist()), rot=-.25, light=.7)
    pal = sns.color_palette("Set2", len(dataframe["Gamma"].unique().tolist()))

    g = sns.FacetGrid(dataframe, row="Gamma", hue="Gamma", aspect=10, height=1, palette=pal)

    # Draw the densities in a few steps
  
    g.map(sns.histplot, "Frequency Bin", fill=True, element="poly")
    #g.map(sns.kdeplot, "Frequency Bin",
    #  bw_adjust=.5, clip_on=False,
    #  fill=True, alpha=1, linewidth=1.5, clip=[dataframe["Frequency Bin"].min(),dataframe["Frequency Bin"].max()])
    g.map(sns.histplot, "Frequency Bin", clip_on=True, color="w", lw=2)


    # passing color=None to refline() uses the hue mapping
    g.map(plt.axhline, y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, "Bin {}".format(label), fontweight="bold", color="black",
                ha="left", va="center", transform=ax.transAxes)


    g.map(label, "Gamma")

    
    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.5)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], xlabel="Gamma", ylabel="", xticks=list(range(0,110,10)))
    
    g.despine(bottom=True, left=True)
    plt.tight_layout()
    plt.savefig('ridge_test4.png')


df4 =  pd.read_pickle('/mnt/sda/home/ludeep/Desktop/PopGen/FinBank/moments_sfs/momentssfs/moments/df4.pkl')
df3 =  pd.read_pickle('/mnt/sda/home/ludeep/Desktop/PopGen/FinBank/moments_sfs/momentssfs/moments/df3.pkl')


#create_ridge_plot(df4)

#df4.columns=['Frequency Bin', 'Gamma']

for i in range(0,9):
    sns.histplot(df3[i])
    plt.savefig('bin_{}.png'.format(i+1))
    plt.close()