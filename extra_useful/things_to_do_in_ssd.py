
''' 

    What all things we can do in ssd, more to improve it further? 
    
    
    
>> on label embedding
    1.Change the label embedding dim, as acc to DDP 
    2.Change the current embedding to one hot vector embedding 
>> On time dimension
    1. Change it to 16, as acc to DDP 
>> Number of schedule steps 
    1. Change it to 10 steps
    2. Change it to 5 steps 
    # Can check at each step improvement in terms of miou
>> mutual information based noise scheduling type
    1. change it to cosine 
>> instead of gumbel max trick, can see to use argmax based sampling, if possible!  
>> change learning rate, if req from 6e-5 to 1e-4, or different!
>> how about using accumulation, i.e. accumulating all the time steps x0 parameterisation, then taking the mean of the accumulated value as the final output value. 
>> see in each diffusion step, how improvement is happening; this might help us to find what could be the more suituable # timesteps. 
'''

## can covert "checkpoint_config = dict(by_epoch=False, interval=16000, max_keep_ckpts=1)" this into 
## "checkpoint_config = dict(by_epoch=False, interval=16000, max_keep_ckpts=1, save_best='mIoU', rule='greater')"