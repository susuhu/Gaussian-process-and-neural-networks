# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# %%
matplotlib.rcParams.update({'font.size': 13})
# %%
plaingploss = pd.read_csv('/Users/susuhu/Local_Docs/cms-project-hu/Results/plainGPloss.csv', header=0,sep=',')
print(plaingploss.info())

# %%
adam_loss = pd.read_csv('/Users/susuhu/Local_Docs/cms-project-hu/Results/adam_results01.txt',header=None, names=['loss'])
print(adam_loss.tail())
# %%
x = np.linspace(0,100,101)

# %%
plt.figure(figsize=(8,5))
plt.plot(x, plaingploss['loss']/10000,label='L-BFGS-B')
plt.plot(x[1:],adam_loss/10000,label='Adam (learning rate=0.1)')
plt.title("Plain GP loss")
plt.xlabel('Iterations',fontsize=14)
plt.ylabel('Loss (10^4)',fontsize=14)
plt.legend()
plt.grid("on")
plt.savefig('PlainGPloss2opt.png')
plt.show()

# %%
conv_adam_loss = pd.read_csv('/Users/susuhu/Local_Docs/cms-project-hu/Results/conv_adam_results01.txt',header=None, names=['loss'])
print(conv_adam_loss.tail())

conv_lbfgs_loss = pd.read_csv('/Users/susuhu/Local_Docs/cms-project-hu/Results/conv_lbfgs_loss.csv',header=0)
print(conv_lbfgs_loss.tail())
# %%
plt.figure(figsize=(8,5))
plt.plot(x, conv_lbfgs_loss['loss']/10000,label='L-BFGS-B')
plt.plot(x[1:],conv_adam_loss/10000,label='Adam (learning rate=0.1)')
plt.title("Convolutional GP loss")
plt.xlabel('Iterations',fontsize=14)
plt.ylabel('Loss (10^4)',fontsize=14)
plt.legend()
plt.grid("on")
plt.savefig('ConvGPloss.png')
plt.show()
# %%
