import torch
import torch.nn as nn
import torch.nn.functional as F

# from https://github.com/suinleelab/MM-cVAE/blob/main/utils.py

DEVICE = torch.device("mps")


class PoissonMultinomial(nn.Module):
    def __init__(self):
        super(PoissonMultinomial, self).__init__()

        self.eps = 1e-6
        self.rescale = False
        self.total_weight = 1

    def forward(self, y_pred, y_true, reduction: str = "none"):

        n_kmers = y_pred.shape[1]

        # add epsilon to protect against tiny values
        y_true = y_true + self.eps
        y_pred = y_pred + self.eps

        multinomial_term = F.poisson_nll_loss(y_pred, y_true, log_input=False, reduction=reduction)

        loss_raw = multinomial_term
        

        return loss_raw


def mmd(x, y, gammas, device):
    gammas = gammas.to(device)

    cost = torch.mean(gram_matrix(x, x, gammas=gammas))# .to(device)
    cost += torch.mean(gram_matrix(y, y, gammas=gammas))# .to(device)
    cost -= 2 * torch.mean(gram_matrix(x, y, gammas=gammas))# .to(device)

    if cost < 0:
        return torch.tensor(0)# .to(device)
    return cost


def gram_matrix(x, y, gammas):
    gammas = gammas.unsqueeze(1)
    pairwise_distances = torch.cdist(x, y, p=2.0)

    pairwise_distances_sq = torch.square(pairwise_distances)
    tmp = torch.matmul(gammas, torch.reshape(pairwise_distances_sq, (1, -1)))
    tmp = torch.reshape(torch.sum(torch.exp(-tmp), 0), pairwise_distances_sq.shape)
    return tmp


class VAELoss(nn.Module):

    def __init__(self, reconstruction_loss_fn, kld_weight: float = 1):
        super(VAELoss, self).__init__()
        self.kld_weight = kld_weight
        self.loss_fn = reconstruction_loss_fn


    def forward(
        self,
        orig: torch.Tensor,
        recon: torch.Tensor,
        mu,
        log_var,
    ):
        # N, C, H, W = orig.shape

        # compute per-pixel MSE loss
        recons_loss = self.loss_fn(
            recon,
            orig,
            reduction="none",
        )
        # compute average of the per-pixel total loss for each image
        recons_loss = torch.mean(torch.sum(recons_loss, dim=1), dim=0)

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

    def __init__(self, reconstruction_loss_fn, kld_weight: float = 1):
        super(CVAELoss, self).__init__()

        self.background_disentanglement_penalty = 10e3
        self.salient_disentanglement_penalty = 10e2
        self.loss_fn = reconstruction_loss_fn

    def forward(
        self,
        tg_inputs,
        bg_inputs,
        cvae_dict,
    ):

        # inputs have been log1p normalized, so
        # we'll undo that to compare to output counts
        tg_outputs = cvae_dict["tg_out"]

        MSE_tg = self.loss_fn(
            tg_outputs,
            tg_inputs,
            reduction="none",
        )

        # compute average per-image loss across the batch
        MSE_tg = torch.mean(torch.sum(MSE_tg, dim=1), dim=0)

        bg_outputs = cvae_dict["bg_out"]

        MSE_bg = self.loss_fn(
            bg_outputs,
            bg_inputs,
            reduction="none",
        )

        # compute average per-image loss across the batch
        MSE_bg = torch.mean(torch.sum(MSE_bg, dim=1), dim=0)

        # compute KL loss per image
        tg_s_log_var, tg_s_mu = cvae_dict["tg_s_log_var"], cvae_dict["tg_s_mu"]
        tg_z_log_var, tg_z_mu = cvae_dict["tg_z_log_var"], cvae_dict["tg_z_mu"]
        bg_z_log_var, bg_z_mu = cvae_dict["bg_z_log_var"], cvae_dict["bg_z_mu"]

        bg_z, tg_z = cvae_dict["bg_z"], cvae_dict["tg_z"]
        bg_s = cvae_dict["bg_s"]

        KLD_z_bg = torch.mean(-0.5 * torch.sum(1 + bg_z_log_var - bg_z_mu.pow(2) - bg_z_log_var.exp(), dim=1), dim=0)
        KLD_z_tg = torch.mean(-0.5 * torch.sum(1 + tg_z_log_var - tg_z_mu.pow(2) - tg_z_log_var.exp(), dim=1), dim=0)
        KLD_s_tg = torch.mean(-0.5 * torch.sum(1 + tg_s_log_var - tg_s_mu.pow(2) - tg_s_log_var.exp(), dim=1), dim=0)

        # KLD_z_bg = -0.5 * torch.sum(1 + bg_z_log_var - bg_z_mu.pow(2) - bg_z_log_var.exp())
        # KLD_z_tg = -0.5 * torch.sum(1 + tg_z_log_var - tg_z_mu.pow(2) - tg_z_log_var.exp())
        # KLD_s_tg = -0.5 * torch.sum(1 + tg_s_log_var - tg_s_mu.pow(2) - tg_s_log_var.exp())

        # print (MSE_tg.item(), MSE_bg.item(), KLD_z_bg.item(), KLD_z_tg.item(), KLD_s_tg.item())
        # print ([el.item() for el in [MSE_bg, MSE_tg, KLD_s_tg, KLD_z_bg, KLD_z_tg]])
        cvae_loss = (MSE_bg + KLD_z_bg) + (MSE_tg + KLD_z_tg + KLD_s_tg)
        # print (cvae_loss.item())

        q_score, q_bar_score = cvae_dict["q_score"], cvae_dict["q_bar_score"]
        discriminator_loss = (-torch.log(q_score) - torch.log(1 - q_bar_score)).mean()
        # print ([(el.min().item(), el.max().item()) for el in [q_score, q_bar_score]])

        # discriminator_loss = (- torch.log(q_score) - torch.log(1 - q_bar_score)).sum()
        # print (discriminator_loss.item())
        cvae_loss += discriminator_loss

        # gammas = torch.FloatTensor([10 ** x for x in range(-6, 7, 1)])
        # # gammas = torch.FloatTensor([10 ** x for x in range(-10, 10, 1)])
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

        return cvae_loss# .to(DEVICE)

class DiscriminatorLoss(nn.Module):
    def __init__(self, gamma: float = 0.):
        super(DiscriminatorLoss, self).__init__()
        self.gamma = gamma

    def forward(self, q_score, q_bar_score):
        tc_loss = F.log(q_score / (1 - q_score)) 
        discriminator_loss = - F.log(q_score) - F.log(1 - q_bar_score)
        return self.gamma * F.mean(tc_loss) + F.mean(discriminator_loss)