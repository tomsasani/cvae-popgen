import torch
import torch.nn as nn
import torch.nn.functional as F

class VAELoss(nn.Module):

    def __init__(self, kld_weight: float = 1.):
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
        # NOTE: is this being calculated correctly?
        recons_loss = F.mse_loss(
            recon,
            orig,
            reduction="none",
        )
        recons_loss *= (H * W)

        # compute KL loss
        kld_loss = torch.mean(
            -0.5
            * torch.sum(
                1 + log_var - mu.pow(2) - log_var.exp(),
                dim=1,
            ),
            dim=0,
        )

        loss = recons_loss.mean() + self.kld_weight * kld_loss
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

        tg_outputs = cvae_dict["tg_out"]
        reconstruction_loss = F.mse_loss(
            tg_inputs.flatten(),
            tg_outputs.flatten(),
            reduction="sum",
        )
        
        bg_outputs = cvae_dict["bg_out"]
        reconstruction_loss += F.mse_loss(
            bg_inputs,
            bg_outputs,
            reduction="sum",
        )

        tg_s_log_var, tg_s_mu = cvae_dict["tg_s_log_var"], cvae_dict["tg_s_mu"]
        tg_z_log_var, tg_z_mu = cvae_dict["tg_z_log_var"], cvae_dict["tg_z_mu"]
        bg_z_log_var, bg_z_mu = cvae_dict["bg_z_log_var"], cvae_dict["bg_z_mu"]

        kl_loss = 1 + tg_s_log_var - torch.square(tg_s_mu) - torch.exp(tg_s_log_var)
        kl_loss += 1 + tg_z_log_var - torch.square(tg_z_mu) - torch.exp(tg_z_log_var)
        kl_loss += 1 + bg_z_log_var - torch.square(bg_z_mu) - torch.exp(bg_z_log_var)

        kl_loss = torch.sum(kl_loss)#, dim=0)
        kl_loss *= -0.5

        cvae_loss = torch.mean(reconstruction_loss + kl_loss)
        return cvae_loss

class DiscriminatorLoss(nn.Module):
    def __init__(self, gamma: float = 0.):
        super(DiscriminatorLoss, self).__init__()
        self.gamma = gamma

    def forward(self, q_score, q_bar_score):
        tc_loss = F.log(q_score / (1 - q_score)) 
        discriminator_loss = - F.log(q_score) - F.log(1 - q_bar_score)
        return self.gamma * F.mean(tc_loss) + F.mean(discriminator_loss)