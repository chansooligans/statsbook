# %% [markdown]
"""
# Test
"""


# %%
x=1
# %%
import nbformat
from nbclient import NotebookClient
# %%
nb = nbformat.read("test.ipynb", as_version=4)
# %%
client = NotebookClient(nb, timeout=600, kernel_name='python3')
# %%
client.execute()
# %%
