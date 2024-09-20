import torch
import torch.nn as nn
import torch.nn.functional as F

# from https://github.com/suinleelab/MM-cVAE/blob/main/utils.py

DEVICE = torch.device("cuda")

def mmd(x, y, gammas, device):
    gammas = gammas.to(device)

    cost = torch.mean(gram_matrix(x, x, gammas=gammas)).to(device)
    cost += torch.mean(gram_matrix(y, y, gammas=gammas)).to(device)
    cost -= 2 * torch.mean(gram_matrix(x, y, gammas=gammas)).to(device)

    if cost < 0:
        return torch.tensor(0).to(device)
    return cost


def gram_matrix(x, y, gammas):
    gammas = gammas.unsqueeze(1)
    pairwise_distances = torch.cdist(x, y, p=2.0)

    pairwise_distances_sq = torch.square(pairwise_distances)
    tmp = torch.matmul(gammas, torch.reshape(pairwise_distances_sq, (1, -1)))
    tmp = torch.reshape(torch.sum(torch.exp(-tmp), 0), pairwise_distances_sq.shape)
    return tmp


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
        recons_loss = F.binary_cross_entropy(
            recon,
            orig,
            reduction="none",
        )

        # compute average per-image loss across the batch
        recons_loss = torch.mean(torch.sum(recons_loss, dim=(1, 2, 3)))

        # compute average per-image KL loss across the batch
        kld_loss = torch.mean(
            -0.5
            * torch.sum(
                1 + log_var - torch.square(mu) - torch.exp(log_var),
                dim=1,
            )
        )

        loss = recons_loss + self.kld_weight * kld_loss
        return loss


class CVAELoss(nn.Module):

    def __init__(self, kld_weight: float = 1):
        super(CVAELoss, self).__init__()

        self.background_disentanglement_penalty = 10e3
        self.salient_disentanglement_penalty = 10e2

    def forward(
        self,
        tg_inputs,
        bg_inputs,
        cvae_dict,
    ):

        N, C, H, W = tg_inputs.shape

        tg_outputs = cvae_dict["tg_out"]
        MSE_tg = F.mse_loss(
            tg_outputs,
            tg_inputs,
            reduction="none",
        )
        # compute average per-image loss across the batch
        MSE_tg = torch.mean(torch.sum(MSE_tg, dim=(1, 2, 3)), dim=0)
        # print ("MSE_tg", MSE_tg.item())
        bg_outputs = cvae_dict["bg_out"]
        MSE_bg = F.mse_loss(
            bg_outputs,
            bg_inputs,
            reduction="none",
        )
        # compute average per-image loss across the batch
        MSE_bg = torch.mean(torch.sum(MSE_bg, dim=(1, 2, 3)), dim=0)
        # print ("MSE_bg", MSE_bg.item())
        # compute KL loss per image
        tg_s_log_var, tg_s_mu = cvae_dict["tg_s_log_var"], cvae_dict["tg_s_mu"]
        tg_z_log_var, tg_z_mu = cvae_dict["tg_z_log_var"], cvae_dict["tg_z_mu"]
        bg_z_log_var, bg_z_mu = cvae_dict["bg_z_log_var"], cvae_dict["bg_z_mu"]

        KLD_z_bg = torch.mean(-0.5 * torch.sum(1 + bg_z_log_var - bg_z_mu.pow(2) - bg_z_log_var.exp(), dim=1), dim=0)
        KLD_z_tg = torch.mean(-0.5 * torch.sum(1 + tg_z_log_var - tg_z_mu.pow(2) - tg_z_log_var.exp(), dim=1), dim=0)
        KLD_s_tg = torch.mean(-0.5 * torch.sum(1 + tg_s_log_var - tg_s_mu.pow(2) - tg_s_log_var.exp(), dim=1), dim=0)
        # print ("KLD_z_bg", KLD_z_bg.item())
        # print ("KLD_z_tg", KLD_z_tg.item())
        # print ("KLD_s_tg", KLD_s_tg.item())
        cvae_loss = (MSE_bg + KLD_z_bg) + (MSE_tg + KLD_z_tg + KLD_s_tg)
        # print ("loss", cvae_loss.item(), N, cvae_loss.item() / N)
        # kl_loss = 1 + tg_s_log_var - torch.square(tg_s_mu) - torch.exp(tg_s_log_var)
        # kl_loss += 1 + tg_z_log_var - torch.square(tg_z_mu) - torch.exp(tg_z_log_var)
        # kl_loss += 1 + bg_z_log_var - torch.square(bg_z_mu) - torch.exp(bg_z_log_var)

        # # compute average per-image KL loss across the batch
        # kl_loss = torch.sum(kl_loss, dim=1)
        # kl_loss *= -0.5
        # kl_loss = torch.mean(kl_loss)

        # cvae_loss = reconstruction_loss + kl_loss

        # from moment-matching paper
        # gammas = torch.FloatTensor([10 ** x for x in range(-6, 7, 1)])
        # background_mmd_loss = self.background_disentanglement_penalty * mmd(
        #     bg_z,
        #     tg_z,
        #     gammas=gammas,
        #     device=DEVICE,
        # )
        # salient_mmd_loss = self.salient_disentanglement_penalty * mmd(
        #     bg_s,
        #     torch.zeros_like(bg_s),
        #     gammas=gammas,
        #     device=DEVICE,
        # )
        # cvae_loss += background_mmd_loss + salient_mmd_loss

        return cvae_loss

class DiscriminatorLoss(nn.Module):
    def __init__(self, gamma: float = 0.):
        super(DiscriminatorLoss, self).__init__()
        self.gamma = gamma

    def forward(self, q_score, q_bar_score):
        tc_loss = F.log(q_score / (1 - q_score)) 
        discriminator_loss = - F.log(q_score) - F.log(1 - q_bar_score)
        return self.gamma * F.mean(tc_loss) + F.mean(discriminator_loss)
