import torch
import torch.nn as nn
import torch.nn.functional as F

# from https://github.com/suinleelab/MM-cVAE/blob/main/utils.py

DEVICE = torch.device("cuda")


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

        multinomial_term = F.poisson_nll_loss(y_pred, y_true, log_input=True, reduction=reduction)

        loss_raw = multinomial_term
        

        return loss_raw


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
        recons_loss = F.mse_loss(
            recon,
            orig,
            reduction="none",
        )
        # recons_loss = F.cosine_embedding_loss(recon, orig, torch.tensor([1]).to(DEVICE), reduction="none")
        # compute average of the per-pixel total loss for each image
        recons_loss = torch.mean(torch.sum(recons_loss, dim=1))

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

        # N, C, H, W = tg_inputs.shape

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

        # print (MSE_tg.item(), MSE_bg.item())
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

        return cvae_loss

class DiscriminatorLoss(nn.Module):
    def __init__(self, gamma: float = 0.):
        super(DiscriminatorLoss, self).__init__()
        self.gamma = gamma

    def forward(self, q_score, q_bar_score):
        tc_loss = F.log(q_score / (1 - q_score)) 
        discriminator_loss = - F.log(q_score) - F.log(1 - q_bar_score)
        return self.gamma * F.mean(tc_loss) + F.mean(discriminator_loss)


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, device, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device = device

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
