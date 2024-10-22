
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
>> see in each diffusion step, how improvement is happening; this might help us to find what could be the more suituable # timesteps. ## now this thing might require some time, cause it is their in DDPS code, but ain't in DDP...have to see to include in our code. >> Done the inclusion of multi-step miou!
>> with batch size of 8, our intial experiment was carried out, it can also be done with batch size of 16 (similar to DDP and DDPS code) ## can be done quickly
>> how to include background (sir, said it doesn't matter what you do, as along as you are predicting and optimising over 19 classes(classes except background) similar to DDP type, since we are not predicting at the inference stage, as our sole purpose is to improve segementation model predictions--corrector for segmentor)
    1. can use FN for background class and make a new confusion matrix, consequently q_mats and q_onestep mats. > a way of including background class 
>> Training for more iteations on top of 160k saved iterations, with the similar training config as that of DDP.


>> AFTER THE 1ST COMPLETE TRAINING OF SSD, PLEASE COPY "mmseg" MODULE FROM TOOLS TO THE PARENT DIRECTORY  -> @DONE
'''

## can covert "checkpoint_config = dict(by_epoch=False, interval=16000, max_keep_ckpts=1)" this into 
## "checkpoint_config = dict(by_epoch=False, interval=16000, max_keep_ckpts=1, save_best='mIoU', rule='greater')"



##############################################################################################################################################
''' 
    below mentioning some modifications in the current ssd code
'''

# current work: resume from 160k iterations saved model, for checking if the performance improves on top of it? @Done : In training 

# current work: it is to see how multi step miou was been calculated by DDPS and have to inculcate in SSD.
# >> below inculcation code: 

'''
# Debugging with vs code the DDPS testing code so, as to understand how multi-step testing is taking place.
# now testing SSD code, for pointing out the first change which needs to be done for the including the DDPS type testing in our code. 
In SSD,
1. changes needs to be done in "simple_test" and "aug_test" functions in encoder_decoder module. >> then change in self.inference (both slide inference and whole inference)  
2. changes in self.whole_inference() and self.slide_inference() 
3. then, chnage the self.encode_decode() in whole inference and slide inference. 


For now, changing only in simple_test and self.whole_inference...later to bring the consequent changes in the aug_test and self.slide_inference. 

A> Changes in encode_decode & similarity_sample in ssd module:  @DONE 
B> Changes in whole_inference & inference & simple_test in encoder_decoder module: @DONE
C> Changes in multi_gpu_test function in api/test.py module: @DONE
D> Changes in main function of tools/test.py module: @DONE
E> Creating new functions in cityscapes.py dataset module for miou calculation: @DONE 

'''

## Changing the timesteps equal to 3 <<experiment1>> @DONE: but the performance was comparable to one with 20 scheduling steps, still letting it run for more iterations. 

""""

Now Doubts to ask/discuss with Sirs:

>> Currently confusion matrix of cityscapes val is being used, but it should be rather confusion matrix of train data (can be perturbed train), as can't have access to the validation GT!

>> Why not much change in miou is there over multi timestep of SSD. even this evident on DDP as well, where over 3 time step it improves upto 0.24 miou ..>DDPS claimed that the structure prior is already being corrected so not much gain over multiple timesteps of diffusion. But the fact of pre-existing confusion in the confusion matrix, is it being rectified?? 

"""

'''
Extras: 

Currently using normalised confusion matrix as the similarity matrix, from which transition rate matrix is made and then consequently 

>> how the transition matrix is made, should it be made one-hot encoding (similar to D3PM)?
>> Ok , I am going with similar to DDP, where everything would be similar to DDP: 

Now, SSD will have: 

################################################
label embed dim = 256  ## this is changed from previous '19' 
time sinpos embed dim = 1024  
mutual info based schedule of cosine kind, ## this is changed from previous 'linear' ## not working much, as over the timesteps miou is decreasing...even d3pm told that linear mutual information schedule was working better than its cosine counterpart!
time step will remain 3 only
################################################

# Differences in still there with the d3pm as well: 
# since our similarity matrix is not doubly stochastic so, the stationary distribution we don't know, and is ambiguious, whereas d3pm says to construct doubly stochastic rate matrix for uniform stationary distribution>> similarity matrix is symmetric for d3pm! 
# they construct similarity matrix using soft adjacency matrix derived from one hot nearest neighbour, whereas we directly use confusion matrix(of validation, for now, later we need to change to train(perturbed))

'''

## probabilty distribution of segformerb2 predictions 
## to visualise the distribution of ==//== 
## find the distribution of current stationary state distribution from our proposed diffusion forward process(assume for now it exists)  

'''
to make grid of latent state space of diffusion models of different noise schedules: 
    1. MI based linear schedule  ## not performing so well
    2. MI based cosine schedule  ## not performing so well 
    3. constant beta schedule : (1-beta_t)*I + beta_t*(normalised_confusion_matrix)   ## done, this also showing results similar to above two points
    
>> performing this now:: Next could be linear/cosine scheduling with beta schedule; where transition matrix is of form similar to constant beta schedule transition matrix.  
>> performing this as well: Instead of gumbel argmax sampling, directly...trying with directly argmax sampling, not introducing more noise...lets see and experiment... 

'''

## reproduced CCDM forward process diffusion >> Done 
## now visualise D3PM uniform forward process diffusion using cosine schdeule and timesteps same as CCDM >> Done 

''' 
    one issue is that we are not reaching uniform stationary distribution when, using cumulative transition matrices (as the consecutive products of individual single step transition matrices)  >>  would say that it is the default problem of discrete diffusion when compared with cumalphas to use! 
'''
## now inferring our proposed D3PM forward diffusion process using the mi linear  >> DONE  

## in order to arrive at a stationary distribution of segformer prediction we need to change default mutual information based schedule to schdeule where the original information linearly interpolates of mutual information of segformer prediction and cityscapes gt (I(x_T, x_0)) rather than to dropping to zero (as in the default case) >> I(x_T, x_0) = -\sum_{x_0 ~ p(x0), x_T ~ q(x_T|x_0)}p(x_0)q(x_T|x_0)log(p(x_0)q(x_T|x_0)) 

### >>>  so D3PM plays a trick where, it starts defining diffusion as the time non-homogeneous CTMC, but later approximates/tricks it as time homogeneous CTMC, and consequently discretizes it. 
### now, we will visualise it, D3PM structured transition to check, that even if the stationary distribution is uniform, bu can we take help from initial confusion of similar classes (at the starting stage) to denoise 

'''
    image softmax prediction have to copy/see from mic to replicate back in mmseg codebase ; 
    can i do one thing that: to work with mic only instead of segformer>>no cause we need to improve upon segformer rather than mic for now::ok then, just running mic to see the differences 

    doing one thing in MIC code base, first running as it is then, changing the config and checkpoint files to segformerb2, since, mic is also based on mmmsegmenation codebase >> below working it 


now onforth, I will do documentation in latex: overleaf!  
'''


'''
>> rough python scipts
## mic running script: 

python -m tools.test --config /home/sidd_s/scratch/mic_checkpoints/csHR2dzurHR_mic_hrda_aef07/csHR2dzurHR_mic_hrda_aef07.py --checkpoint /home/sidd_s/scratch/mic_checkpoints/csHR2dzurHR_mic_hrda_aef07/iter_40000_relevant.pth --test-set --format-only --eval-options imgfile_prefix=/home/sidd_s/scratch/results/MIC/  to_label_id=False  

## for setting up github push, if faced an issue: 
git remote set-url origin https://<personal_access_token>@github.com/<your_username or organization_name>/<repo_name>.git
1. 
git remote set-url origin https://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@github.com/Siddharth-Shrivastava7/MIC_mod.git
2. 
git remote set-url origin https://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@github.com/Siddharth-Shrivastava7/mmseg_mod.git
3.  
git remote set-url origin https://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@github.com/Siddharth-Shrivastava7/Diffusion_conditional_prior_segmentation.git

## just for testing, adding this>
'''


 

