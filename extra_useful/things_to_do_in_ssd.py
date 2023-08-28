
''' 

    What all things we can do in ssd, more to improve it further? 
    
    
    
>> on label embedding ## can happen quickly
    1.Change the label embedding dim, as acc to DDP  
    2.Change the current embedding to one hot vector embedding 
>> On time dimension ## can happen quickly 
    1. Change it to 16, as acc to DDP 
>> Number of schedule steps ## can happen quickly  
    1. Change it to 10 steps
    2. Change it to 5 steps 
    # Can check at each step improvement in terms of miou
>> mutual information based noise scheduling type ## can happen quickly  
    1. change it to cosine 
>> instead of gumbel max trick, can see to use argmax based sampling, if possible! ## can happen quickly   
>> change learning rate, if req from 6e-5 to 1e-4, or different! ## can happen quickly  
>> how about using accumulation, i.e. accumulating all the time steps x0 parameterisation, then taking the mean of the accumulated value as the final output value. ## can happen quickly (as it is already been done, before and it was the concept of DDP(mentioned in their code))
>> see in each diffusion step, how improvement is happening; this might help us to find what could be the more suituable # timesteps. ## now this thing might require some time, cause it is their in DDPS code, but ain't in DDP...have to see to include in our code.
>> with batch size of 8, our intial experiment was carried out, it can also be done with batch size of 16 (similar to DDP and DDPS code) ## can be done quickly
>> how to include background (sir, said it doesn't matter what you do, as along as you are predicting and optimising over 19 classes(classes except background) similar to DDP type, since we are not predicting at the inference stage, as our sole purpose is to improve segementation model predictions--corrector for segmentor)
    1. can use FN for background class and make a new confusion matrix, consequently q_mats and q_onestep mats. > a way of including background class 



>> AFTER THE 1ST COMPLETE TRAINING OF SSD, PLEASE COPY "mmseg" MODULE FROM TOOLS TO THE PARENT DIRECTORY  -> @DONE
'''

## can covert "checkpoint_config = dict(by_epoch=False, interval=16000, max_keep_ckpts=1)" this into 
## "checkpoint_config = dict(by_epoch=False, interval=16000, max_keep_ckpts=1, save_best='mIoU', rule='greater')"



##############################################################################################################################################
''' 
    below mentioning some modifications in the current ssd code
'''

# 1.  