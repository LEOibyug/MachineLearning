import torch.optim as optim
import itertools
from scripts.Networks import *
from scripts.directions import *
from scripts.imageUtils import *
from test.train.train_attention import batch_size


def train(G_type,device,check_gap,dataloader_real, dataloader_monet, G_R2M, G_M2R, D_R, D_M, criterion_GAN, criterion_cycle, optimizer_G,
          optimizer_D_R, optimizer_D_M, epoch_num):
    for epoch in range(epoch_num):
        for i, (real_images, monet_images) in enumerate(zip(dataloader_real, dataloader_monet)):
            real_images = real_images[0].to(device)
            monet_images = monet_images[0].to(device)

            optimizer_G.zero_grad()

            fake_monet = G_R2M(real_images)
            pred_fake = D_M(fake_monet)
            loss_GAN_R2M = criterion_GAN(pred_fake, torch.ones_like(pred_fake).to(device))

            fake_real = G_M2R(monet_images)
            pred_fake = D_R(fake_real)
            loss_GAN_M2R = criterion_GAN(pred_fake, torch.ones_like(pred_fake).to(device))

            rec_real = G_M2R(fake_monet)
            loss_cycle_real = criterion_cycle(rec_real, real_images)

            rec_monet = G_R2M(fake_real)
            loss_cycle_monet = criterion_cycle(rec_monet, monet_images)

            loss_G = loss_GAN_R2M + loss_GAN_M2R + 10.0 * (loss_cycle_real + loss_cycle_monet)
            loss_G.backward()
            optimizer_G.step()

            optimizer_D_R.zero_grad()

            pred_real = D_R(real_images)
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real).to(device))

            pred_fake = D_R(fake_real.detach())
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake).to(device))

            loss_D_R_total = (loss_D_real + loss_D_fake) * 0.5
            loss_D_R_total.backward()
            optimizer_D_R.step()

            optimizer_D_M.zero_grad()

            pred_real = D_M(monet_images)
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real).to(device))

            pred_fake = D_M(fake_monet.detach())
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake).to(device))

            loss_D_M_total = (loss_D_real + loss_D_fake) * 0.5
            loss_D_M_total.backward()
            optimizer_D_M.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch}/{epoch_num}], Step [{i}/{len(dataloader_real)}], "
                      f"Loss_G: {loss_G.item():.4f}, Loss_D_R: {loss_D_R_total.item():.4f}, Loss_D_M: {loss_D_M_total.item():.4f}")
        if epoch % check_gap == 0:
            torch.save(G_M2R.state_dict(), f'{G_R2M_SAVE}{G_type}_{epoch}.pth')
    torch.save(G_M2R.state_dict(), f'{G_R2M_SAVE}{G_type}_{epoch_num}.pth')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'using device {device}')

G_R2M = GeneratorMultiHeadAtt().to(device)
G_M2R = GeneratorMultiHeadAtt().to(device)
D_R = Discriminator().to(device)
D_M = Discriminator().to(device)

criterion_GAN = nn.MSELoss().to(device)
criterion_cycle = nn.L1Loss().to(device)
optimizer_G = optim.Adam(itertools.chain(G_R2M.parameters(), G_M2R.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_R = optim.Adam(D_R.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_M = optim.Adam(D_M.parameters(), lr=0.0002, betas=(0.5, 0.999))

dataloader_real = get_data_loader(T_REAL,batch_size)
dataloader_monet = get_data_loader(T_MONET,batch_size)

G_type = G_R2M.__class__.__name__
batch_size = 15
epoch_num = 500
check_gap = 10


train(G_type,device,check_gap,dataloader_real,dataloader_monet, G_R2M, G_M2R, D_R, D_M, criterion_GAN, criterion_cycle, optimizer_G,
      optimizer_D_R, optimizer_D_M, epoch_num)