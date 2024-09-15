import torch
import torch.nn as nn
import torch.nn.functional as F

class VAELoss(nn.Module):

    def __init__(self, kld_weight: float = 1):
        super(VAELoss, self).__init__()
        self.kld_weight = kld_weight

    def forward(
        self,
        orig: torch.Tensor,
        recon: torch.Tensor,
        mu,
        log_var,
    ):
        N, C, H, W = orig.shape

        # compute per-pixel MSE loss
        recons_loss = F.mse_loss(
            recon,
            orig,
            reduction="none",
        ) #* (H * W)

        # compute average per-image loss across the batch
        recons_loss = torch.mean(torch.sum(recons_loss, dim=(1, 2, 3)))

        # compute average per-image KL loss across the batch
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))

        loss = recons_loss + self.kld_weight * kld_loss
        return loss


class CVAELoss(nn.Module):

    def __init__(self):
        super(CVAELoss, self).__init__()

    def forward(
        self,
        tg_inputs,
        bg_inputs,
        cvae_dict,
    ):

        N, C, H, W = tg_inputs.shape

        # compute per-pixel MSE loss
        tg_outputs = cvae_dict["tg_out"]
        reconstruction_loss_tg = F.mse_loss(
            tg_inputs,
            tg_outputs,
            reduction="none",
        ) # * H * W

        # compute average per-image loss across the batch
        reconstruction_loss_tg = torch.mean(torch.sum(reconstruction_loss_tg, dim=(1, 2, 3)))
        
        # calculate total MSE loss on backgrounds
        bg_outputs = cvae_dict["bg_out"]
        reconstruction_loss_bg = F.mse_loss(
            bg_inputs,
            bg_outputs,
            reduction="none",
        ) # * H * W

        reconstruction_loss_bg = torch.mean(torch.sum(reconstruction_loss_bg, dim=(1, 2, 3)))

        # sum tg and bg MSE loss
        reconstruction_loss = reconstruction_loss_tg + reconstruction_loss_bg

        # compute KL loss per image
        tg_s_log_var, tg_s_mu = cvae_dict["tg_s_log_var"], cvae_dict["tg_s_mu"]
        tg_z_log_var, tg_z_mu = cvae_dict["tg_z_log_var"], cvae_dict["tg_z_mu"]
        bg_z_log_var, bg_z_mu = cvae_dict["bg_z_log_var"], cvae_dict["bg_z_mu"]

        kl_loss = 1 + tg_s_log_var - torch.square(tg_s_mu) - torch.exp(tg_s_log_var)
        kl_loss += 1 + tg_z_log_var - torch.square(tg_z_mu) - torch.exp(tg_z_log_var)
        kl_loss += 1 + bg_z_log_var - torch.square(bg_z_mu) - torch.exp(bg_z_log_var)

        # compute average per-image KL loss across the batch
        kl_loss = torch.sum(kl_loss, dim=1)
        kl_loss *= -0.5
        kl_loss = torch.mean(kl_loss)

        cvae_loss = reconstruction_loss + kl_loss
        return cvae_loss

class DiscriminatorLoss(nn.Module):
    def __init__(self, gamma: float = 0.):
        super(DiscriminatorLoss, self).__init__()
        self.gamma = gamma

    def forward(self, q_score, q_bar_score):
        tc_loss = F.log(q_score / (1 - q_score)) 
        discriminator_loss = - F.log(q_score) - F.log(1 - q_bar_score)
        return self.gamma * F.mean(tc_loss) + F.mean(discriminator_loss)
