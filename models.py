import torch
from torch import nn
from typing import List, Tuple, Union
import torchvision
import torchvision.transforms.functional as F


class EncoderFC(nn.Module):
    def __init__(self, *, in_W: int, latent_dim: int, hidden_dims: List[int]):
        super(EncoderFC, self).__init__()

        layers = [
            nn.Linear(in_W, hidden_dims[0]),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dims[0]),
        ]

        for hi in range(1, len(hidden_dims)):
            layers.extend(
                [
                    nn.Linear(hidden_dims[hi - 1], hidden_dims[hi]),
                    nn.LeakyReLU(0.2),
                    nn.BatchNorm1d(
                        hidden_dims[hi],
                    ),
                ]
            )

        self.fc = nn.Sequential(*layers)

        self.fc_mu = nn.Linear(
            hidden_dims[-1],
            latent_dim,
        )
        self.fc_var = nn.Linear(
            hidden_dims[-1],
            latent_dim,
        )

    def forward(self, x):

        x = self.fc(x)

        mu, log_var = self.fc_mu(x), self.fc_var(x)

        return [mu, log_var]


class DecoderFC(nn.Module):
    def __init__(self, *, in_W: int, latent_dim: int, hidden_dims: List[int]):
        super(DecoderFC, self).__init__()

        bias = False

        hidden_dims = hidden_dims[::-1]

        layers = [
            nn.Linear(latent_dim, hidden_dims[0]),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dims[0]),
        ]

        for hi in range(1, len(hidden_dims)):
            layers.extend([
                nn.Linear(hidden_dims[hi - 1], hidden_dims[hi]),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(
                    hidden_dims[hi],
                ),]
            )

        self.fc = nn.Sequential(*layers)

        self.fc_out = nn.Linear(hidden_dims[-1], in_W, bias=bias)
        # self.scaling_factor = nn.Linear(hidden_dims[-1], 1)

        self.softmax = nn.Softmax(1)

    def forward(self, x: torch.Tensor, libsize: torch.Tensor):
        x = self.fc(x)
        x = self.fc_out(x)
        x = self.softmax(x)
        # multiply by libsize to get poisson lambda
        x = x * torch.unsqueeze(libsize, dim=1)
        return x


class Discriminator(nn.Module):
    def __init__(self, latent_dim_s: int = 2, latent_dim_z: int = 4):
        super(Discriminator, self).__init__()

        self.discriminator = nn.Sequential(
                nn.Linear(latent_dim_s + latent_dim_z, 1),
                nn.Sigmoid(),
            )

    def forward(self, tg_s, tg_z):

        # get shape of z, where N is batch
        # size and L is size of total latent space
        N, _ = tg_z.shape

        # if our batch size is odd, subset the batch
        if N % 2 != 0:
            N -= 1
            tg_z = tg_z[:N]
            tg_s = tg_s[:N]

        half_N = int(N / 2)

        # first half of batch of irrelevant latent space
        z1 = tg_z[:half_N, :]
        z2 = tg_z[half_N:, :]
        s1 = tg_s[:half_N, :]
        s2 = tg_s[half_N:, :]

        q = torch.cat(
            [
                torch.cat([s1, z1], dim=1),
                torch.cat([s2, z2], dim=1),
            ],
            dim=0,
        )

        q_bar = torch.cat(
            [
                torch.cat([s1, z2], dim=1),
                torch.cat([s2, z1], dim=1),
            ],
            dim=0,
        )

        q_bar_score = self.discriminator(q_bar)
        q_score = self.discriminator(q)

        return q_score, q_bar_score


class VAE(nn.Module):

    def __init__(
        self,
        encoder,
        decoder,
    ) -> None:
        super(VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return eps * std + mu

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        # figure out the library size (number of mutations) in each
        # training example
        log_libsize = torch.sum(x, dim=1)

        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z, log_libsize)
        return [decoded, mu, log_var, z]


# contrastive-vae-no-bias
class CVAE(nn.Module):

    def __init__(
        self,
        s_encoder,
        z_encoder,
        decoder,
        discriminator,
    ):
        super(CVAE, self).__init__()

        # instantiate two encoders q_s and q_z
        self.qs = s_encoder
        self.qz = z_encoder
        # instantiate one shared decoder
        self.decoder = decoder
        # instantitate the disrimintator
        self.discriminator = discriminator

    def reparameterize(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return eps * std + mu

    def forward(self, tg_inputs, bg_inputs) -> torch.Tensor:

        tg_log_libsize = torch.sum(tg_inputs, dim=1)
        bg_log_libsize = torch.sum(bg_inputs, dim=1)


        # step 1: pass target features through irrelevant
        # and salient encoders.

        # irrelevant variables
        tg_z_mu, tg_z_log_var = self.qz(tg_inputs)
        tg_z = self.reparameterize(tg_z_mu, tg_z_log_var)
        # salient variables
        tg_s_mu, tg_s_log_var = self.qs(tg_inputs)
        tg_s = self.reparameterize(tg_s_mu, tg_s_log_var)

        # step 2: pass background features through
        # the irrelevant and salient encoders
        bg_z_mu, bg_z_log_var = self.qz(bg_inputs)
        bg_z = self.reparameterize(bg_z_mu, bg_z_log_var)

        bg_s_mu, bg_s_log_var = self.qs(bg_inputs)
        bg_s = self.reparameterize(bg_s_mu, bg_s_log_var)

        # step 3: decode

        # decode the target outputs using both the salient and
        # irrelevant features

        tg_outputs = self.decoder(torch.cat([tg_s, tg_z], dim=1), tg_log_libsize)
        # we decode the background outputs using only the irrelevant
        # features
        bg_outputs = self.decoder(torch.cat([torch.zeros_like(bg_s), bg_z], dim=1), bg_log_libsize)
        # we decode the "foreground" using just the salient features
        fg_outputs = self.decoder(torch.cat([tg_s, torch.zeros_like(tg_z)], dim=1), tg_log_libsize)

        # step 4: (optional) discriminate
        q_score, q_bar_score = self.discriminator(tg_s, tg_z)

        out_dict = {
            "tg_out": tg_outputs,
            "bg_out": bg_outputs,
            "fg_out": fg_outputs,

            "tg_s_mu": tg_s_mu,
            "tg_s_log_var": tg_s_log_var,
            "tg_s": tg_s,

            "tg_z_mu": tg_z_mu,
            "tg_z_log_var": tg_z_log_var,
            "tg_z": tg_z,

            "bg_s": bg_s,
            "bg_s_mu": bg_s_mu,
            "bg_s_log_var": bg_s_log_var,

            "bg_z": bg_z,
            "bg_z_mu": bg_z_mu,
            "bg_z_log_var": bg_z_log_var,

            "q_score": q_score,
            "q_bar_score": q_bar_score,
        }

        return out_dict


if __name__ == "__main__":

    KERNEL_SIZE = (1, 3) #(1, 5)
    STRIDE = (1, 2) #(1, 2)
    PADDING = (0, 1) #(0, 2)
    OUTPUT_PADDING = (0, 1) #(0, 1)
    INTERMEDIATE_DIM = 128
    IN_HW = (1, 16)# (100, 32)
    HIDDEN_DIMS = [32]
    LATENT_DIM = 2

    encoder = EncoderFC(
    latent_dim=LATENT_DIM,
        in_W=96,
        hidden_dims=HIDDEN_DIMS,
        
)

    decoder = DecoderFC(
        latent_dim=LATENT_DIM,
        in_W=96,
        hidden_dims=HIDDEN_DIMS,
        
    )

    model = VAE(
        encoder=encoder,
        decoder=decoder,
    )
    model = model.to("cpu")

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print (pytorch_total_params)

    x = torch.rand(size=(100, 96)).to("cpu")

    print (model(x)[0].shape)
