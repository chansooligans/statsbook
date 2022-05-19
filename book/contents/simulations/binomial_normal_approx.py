# %% [markdown]
"""
# Normal Approximation to the Binomial
""" 

# %% tags=['hide-cell']
from IPython import get_ipython
import numpy as np
np.random.seed(0)
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# %%
n = 3**2
p = 0.1
print(n*p)

# %% tags=["hide-output"]
if get_ipython() is not None:
    get_ipython().run_line_magic('matplotlib', 'ipympl')

fig, (ax1,ax2) = plt.subplots(2,1)
ax2b = ax2.twinx() 
fig.set_size_inches(8, 7, True)
frames = 50
samples = []

def init():
    sns.kdeplot(
        np.random.normal(
            n*p,
            np.sqrt(n*p*(1-p)),
            1000
        ),
        color="grey",
        ax=ax2
    )
    ax2.axvline(n*p,0,1, color="black")
    ax2.set_xlim(0-5,n+5)

def animate(frame_number):
    dim = int(np.sqrt(n))
    grid = np.ones((dim,dim)).astype(int)
    sample = np.random.binomial(grid, p)
    
    sample_sum = np.sum(sample)
    samples.append(sample_sum)
    
    if (frame_number+1) % 5 == 0:
        ax1.clear()
        ax2b.clear()
        
        sns.heatmap(
            sample, 
            annot=True, 
            fmt="d", 
            cbar=False, 
            linewidths=2, 
            linecolor="grey",
            ax=ax1
        )

        ax1.text(0.5,1.1,
            f"Simulation {frame_number+1} out of {frames}",
            bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
            transform=ax1.transAxes, 
            ha="center",
            weight='bold',
            size=12
        )

        sns.kdeplot(samples, color="tan", ax = ax2b)
        ax2b.axvline(np.mean(samples),0,1, color="orange")

ani = animation.FuncAnimation(
    fig, 
    animate, 
    frames=frames,
    init_func=init,
    repeat=False, 
    blit=True,
    interval=50
)
# %%
