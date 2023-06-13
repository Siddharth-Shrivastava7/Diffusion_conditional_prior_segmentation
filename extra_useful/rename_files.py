import os
 
# Function to rename multiple files
def main():
   
    folder = "/home/sidd_s/scratch/dataset/night_city/NightCity-label/label/val/trainid"
    for count, filename in enumerate(os.listdir(folder)):
        src =f"{folder}/{filename}" 
        dst = f"{folder}/{filename.replace('_labelIds.png', '.png')}"
         
        # rename() function will
        # rename all the files
        os.rename(src, dst)
 
# Driver Code
if __name__ == '__main__':
     
    # Calling main() function
    main()