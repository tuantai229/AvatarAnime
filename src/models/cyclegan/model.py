# Định nghĩa kiến trúc model
import torch
import torch.nn as nn
import itertools
from .networks import Generator, Discriminator, init_weights
import torch.nn.functional as F

class CycleGANModel:
    def __init__(self, device, lr=0.0002, beta1=0.5, beta2=0.999, lambda_cycle=10.0, lambda_identity=5.0):
        """
        Khởi tạo mô hình CycleGAN
        
        Args:
            device: thiết bị để chạy mô hình (cpu hoặc cuda)
            lr: learning rate
            beta1, beta2: tham số cho Adam optimizer
            lambda_cycle: trọng số cho cycle consistency loss
            lambda_identity: trọng số cho identity loss
        """
        self.device = device
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        
        # Khởi tạo generators
        self.G_A2B = Generator().to(device)  # Generator A->B (Person to Anime)
        self.G_B2A = Generator().to(device)  # Generator B->A (Anime to Person)
        
        # Khởi tạo weights
        init_weights(self.G_A2B)
        init_weights(self.G_B2A)
        
        # Khởi tạo discriminators
        self.D_A = Discriminator().to(device)  # Discriminator for domain A
        self.D_B = Discriminator().to(device)  # Discriminator for domain B
        
        # Khởi tạo weights
        init_weights(self.D_A)
        init_weights(self.D_B)
        
        # Định nghĩa các loss functions
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        
        # Khởi tạo optimizers
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.G_A2B.parameters(), self.G_B2A.parameters()),
            lr=lr, betas=(beta1, beta2)
        )
        self.optimizer_D = torch.optim.Adam(
            itertools.chain(self.D_A.parameters(), self.D_B.parameters()),
            lr=lr, betas=(beta1, beta2)
        )
        
        # Lưu trữ loss values để theo dõi
        self.losses = {
            'D_A': [], 'D_B': [], 'G_A2B': [], 'G_B2A': [],
            'cycle_A': [], 'cycle_B': [], 'identity_A': [], 'identity_B': []
        }

    def set_input(self, real_A, real_B):
        """Đặt input images"""
        self.real_A = real_A.to(self.device)
        self.real_B = real_B.to(self.device)

    def forward(self):
        """Thực hiện forward pass"""
        # G_A2B(A)
        self.fake_B = self.G_A2B(self.real_A)
        self.rec_A = self.G_B2A(self.fake_B)   # G_B2A(G_A2B(A))
        
        # G_B2A(B)
        self.fake_A = self.G_B2A(self.real_B)
        self.rec_B = self.G_A2B(self.fake_A)   # G_A2B(G_B2A(B))
        
        # Identity mapping
        self.identity_A = self.G_B2A(self.real_A)
        self.identity_B = self.G_A2B(self.real_B)

    def backward_D_basic(self, netD, real, fake):
        """Tính toán GAN loss cho discriminator"""
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterion_GAN(pred_real, torch.ones_like(pred_real))
        
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
        
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        
        return loss_D

    def backward_D_A(self):
        """Tính GAN loss cho discriminator D_A"""
        loss_D_A = self.backward_D_basic(self.D_A, self.real_A, self.fake_A)
        self.loss_D_A = loss_D_A.item()
        self.losses['D_A'].append(self.loss_D_A)

    def backward_D_B(self):
        """Tính GAN loss cho discriminator D_B"""
        loss_D_B = self.backward_D_basic(self.D_B, self.real_B, self.fake_B)
        self.loss_D_B = loss_D_B.item()
        self.losses['D_B'].append(self.loss_D_B)

    def backward_G(self):
        """Tính losses cho generators G_A2B và G_B2A"""
        # Adversarial loss
        # D_A(G_B2A(B))
        self.loss_G_B2A = self.criterion_GAN(self.D_A(self.fake_A), torch.ones_like(self.D_A(self.fake_A)))
        # D_B(G_A2B(A))
        self.loss_G_A2B = self.criterion_GAN(self.D_B(self.fake_B), torch.ones_like(self.D_B(self.fake_B)))
        
        # Cycle consistency loss
        self.loss_cycle_A = self.criterion_cycle(self.rec_A, self.real_A) * self.lambda_cycle
        self.loss_cycle_B = self.criterion_cycle(self.rec_B, self.real_B) * self.lambda_cycle
        
        # Identity loss (tùy chọn)
        self.loss_identity_A = self.criterion_identity(self.identity_A, self.real_A) * self.lambda_identity
        self.loss_identity_B = self.criterion_identity(self.identity_B, self.real_B) * self.lambda_identity
        
        # Combined loss
        self.loss_G = self.loss_G_A2B + self.loss_G_B2A + self.loss_cycle_A + self.loss_cycle_B + self.loss_identity_A + self.loss_identity_B
        self.loss_G.backward()
        
        # Lưu các loss values
        self.losses['G_A2B'].append(self.loss_G_A2B.item())
        self.losses['G_B2A'].append(self.loss_G_B2A.item())
        self.losses['cycle_A'].append(self.loss_cycle_A.item())
        self.losses['cycle_B'].append(self.loss_cycle_B.item())
        self.losses['identity_A'].append(self.loss_identity_A.item())
        self.losses['identity_B'].append(self.loss_identity_B.item())

    def optimize_parameters(self):
        """Cập nhật parameters của mô hình trong một iteration"""
        # Forward pass
        self.forward()
        
        # G_A2B and G_B2A
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
        # D_A and D_B
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()
    
    def save_models(self, path, epoch):
        """Lưu các models"""
        torch.save(self.G_A2B.state_dict(), f"{path}/G_A2B_epoch_{epoch}.pth")
        torch.save(self.G_B2A.state_dict(), f"{path}/G_B2A_epoch_{epoch}.pth")
        torch.save(self.D_A.state_dict(), f"{path}/D_A_epoch_{epoch}.pth")
        torch.save(self.D_B.state_dict(), f"{path}/D_B_epoch_{epoch}.pth")
    
    def load_models(self, path, epoch):
        """Load các models đã lưu trước đó"""
        self.G_A2B.load_state_dict(torch.load(f"{path}/G_A2B_epoch_{epoch}.pth"))
        self.G_B2A.load_state_dict(torch.load(f"{path}/G_B2A_epoch_{epoch}.pth"))
        self.D_A.load_state_dict(torch.load(f"{path}/D_A_epoch_{epoch}.pth"))
        self.D_B.load_state_dict(torch.load(f"{path}/D_B_epoch_{epoch}.pth"))