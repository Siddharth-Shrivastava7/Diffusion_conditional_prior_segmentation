import torch 
import torch.nn as nn 
import numpy as np
from sklearn.neighbors import NearestNeighbors


def main():
    ## pretrained semantic classes embedding from DDP  ## as the low hanging fruit
    embedding_table = nn.Embedding(20, 16) 

    ## loading the embedding weights 
    # PATH = '/home/sidd_s/scratch/saved_models/DDP/embedding_table_pretrained_cityscapes.pth'  ## from DDP 
    PATH = '/home/sidd_s/scratch/saved_models/DDPS/segformer_b2_multistep_cityscapes/embedding_model.pth'
    embedding_table.load_state_dict(torch.load(PATH))
    embedding_table.eval()

    ## checking the embedding of the classes (0 to 19) with 19 as the background, so semantic meaningfull from 0 to 18
    # print(embedding_table(torch.tensor(19))) 
    
    '''
        calculating the distance of embedding vectors 
        
        will check with different distances (cosine similarity, eucledian distance, ham), as to which one is performing better 
    ''' 
    
    input_embedding_vectors = np.array([embedding_table(torch.tensor(i)).detach().numpy() for i in range(19)]) 
   
    model = NearestNeighbors(n_neighbors=3,
                         metric='cosine',
                         algorithm='auto',
                         n_jobs=-1)
    model.fit(input_embedding_vectors)
    distances, indices = model.kneighbors(input_embedding_vectors)
    print('cosine similarity >>',indices)
    
    model = NearestNeighbors(n_neighbors=3,
                         metric='euclidean',
                         algorithm='auto',
                         n_jobs=-1)
    model.fit(input_embedding_vectors)
    distances, indices = model.kneighbors(input_embedding_vectors)
    print('euclidean distances >>',indices) 
    
    model = NearestNeighbors(n_neighbors=3,
                         metric='manhattan',
                         algorithm='auto',
                         n_jobs=-1)
    model.fit(input_embedding_vectors)
    distances, indices = model.kneighbors(input_embedding_vectors)
    print('manhattan distances >>',indices)  



if __name__ == '__main__':
    main()    