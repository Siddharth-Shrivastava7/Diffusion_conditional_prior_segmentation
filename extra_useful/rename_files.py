import os
from tqdm import tqdm

# Function to rename multiple files
def main():
   
    folder = "/home/sidd_s/scratch/dataset/Foggy_Driving_Full/gtFine/"
    for filename in tqdm(os.listdir(folder)):
        if filename.find('_labelTrainIds.png')!=-1:
            src =f"{folder}/{filename}" 
            fine_or_course = filename.split('_')[-2]
            dst = f"{folder}/{filename.replace('_' + fine_or_course,'')}"
            
            # rename() function will
            # rename all the files
            os.rename(src, dst)
 
# Driver Code
if __name__ == '__main__':
     
    # Calling main() function
    main()