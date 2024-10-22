import torch
import torchvision
from torchvision import transforms
from diffusers import UNet2DModel
from torch.optim import Adam 
from tqdm import tqdm

def get_model():
    block_out_channels=(128, 128, 256, 256, 512, 512)
    down_block_types=( 
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D", 
        "DownBlock2D", 
        "DownBlock2D", 
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    )
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D"  
    )
    return UNet2DModel(block_out_channels=block_out_channels,out_channels=3, in_channels=3, up_block_types=up_block_types, down_block_types=down_block_types, add_attention=True)

@torch.no_grad()
def sample_iadb(model, x0, nb_step):
    x_alpha = x0
    for t in range(nb_step):
        alpha_start = (t/nb_step)
        alpha_end =((t+1)/nb_step)

        d = model(x_alpha, torch.tensor(alpha_start, device=x_alpha.device))['sample']
        x_alpha = x_alpha + (alpha_end-alpha_start)*d

    return x_alpha

def main(): 
    print('in the main function')
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    CELEBA_FOLDER = '/home/sidd_s/scratch/dataset/celeba/'
    target_transform = transforms.Compose([transforms.Resize(64),transforms.CenterCrop(64), transforms.RandomHorizontalFlip(0.5),transforms.ToTensor()])  
    stationary_transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.Resize(64),transforms.CenterCrop(64), transforms.RandomHorizontalFlip(0.5),transforms.ToTensor()])
    target_dataset = torchvision.datasets.CelebA(root=CELEBA_FOLDER, split='train',
                                            download=True, transform=target_transform)
    stationary_dataset = torchvision.datasets.CelebA(root=CELEBA_FOLDER, split='train',
                                            download=True, transform=stationary_transform)
    
    # dataset = torch.utils.data.TensorDataset(target_dataset, stationary_dataset)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
    target_dataloader = torch.utils.data.DataLoader(target_dataset, batch_size=64, shuffle=False, num_workers=0, drop_last=True)  
    stationary_dataloader = torch.utils.data.DataLoader(stationary_dataset, batch_size=64, shuffle=False, num_workers=0, drop_last=True)   
    assert len(stationary_dataset) == len(target_dataset)
    print('dataset loaded successfully!')

    model = get_model() 
    model = model.to(device)
    print('Model loaded into cuda')

    optimizer = Adam(model.parameters(), lr=1e-4)
    nb_iter = 0
    print('Start training')
    for current_epoch in tqdm(range(100)):
        for ind in range(len(target_dataloader)):
        # for i, (target_data, stationary_data) in enumerate(dataloader):
            x1 = (next(iter(target_dataloader))[0].to(device)*2)-1
            x0 = (next(iter(stationary_dataloader))[0].to(device)*2)-1
            # x1 = (target_data[0].to(device)*2)-1
            # x0 = (stationary_data[0].to(device)*2)-1
            
            bs = x0.shape[0]

            alpha = torch.rand(bs, device=device)
            x_alpha = alpha.view(-1,1,1,1) * x1 + (1-alpha).view(-1,1,1,1) * x0
            
            d = model(x_alpha, alpha)['sample']
            loss = torch.sum((d - (x1-x0))**2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            nb_iter += 1

            if nb_iter % 200 == 0: 
                print('In Sampling')
                with torch.no_grad():
                    print(f'Save export {nb_iter}')
                    sample = (sample_iadb(model, x0, nb_step=128) * 0.5) + 0.5
                    torchvision.utils.save_image(sample, f'/home/sidd_s/scratch/saved_models/iadb_mod/sample_imgs/export_{str(nb_iter).zfill(8)}.png')
                    torch.save(model.state_dict(), f'/home/sidd_s/scratch/saved_models/iadb_mod/celeba.ckpt')


if __name__ == '__main__':
    main()