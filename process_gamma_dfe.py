import torch
import os

lsdirs = os.listdir('gamma_dfe2/')

for j, a_file in enumerate(lsdirs):

    if j > 0:
        temp = torch.load(a_file)
        temp = temp*120 # multiply by batch of samples to unaverage out
        data = data + temp
    else:
        data = torch.load(a_file)


print("Finished calculating files, now calculating average")

data = data / (120*(j+1))

torch.save(data, 'final_gamma_dfe.pkl')
    
        
    
