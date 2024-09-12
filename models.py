import torch
from torch import nn
from typing import List, Tuple, Union
import torchvision
import torchvision.transforms.functional as F


class ConvBlock2D(nn.Module):

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: int,
        padding: int,
    ):
        super(ConvBlock2D, self).__init__()

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        relu = nn.ReLU()
        norm = nn.BatchNorm2d(out_channels)
        #layers = [conv, norm, relu]
        layers = [conv, relu]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ConvTransposeBlock2D(nn.Module):

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: int,
        padding: int,
        output_padding: int,
    ):
        super(ConvTransposeBlock2D, self).__init__()

        conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        relu = nn.ReLU()
        norm = nn.BatchNorm2d(out_channels)
        #layers = [conv, norm, relu]
        layers = [conv, relu]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        kernel_size: int = 5,
        stride: int = 2,
        padding: int = 1,
        hidden_dims: List[int] = None,
        intermediate_dim: int = 128,
        in_H: int = 32,
    ) -> None:
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [16, 32, 64, 128, 256]

        # figure out final size of image after convs
        out_H = int(in_H / (2 ** len(hidden_dims)))

        encoder_blocks = []
        for h_dim in hidden_dims:
            # initialize convolutional block
            block = ConvBlock2D(
                in_channels=in_channels,
                out_channels=h_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            encoder_blocks.append(block)

            in_channels = h_dim

        self.encoder_conv = nn.Sequential(*encoder_blocks)

        self.fc_intermediate = nn.Linear(
            hidden_dims[-1] * out_H * out_H,
            intermediate_dim,
        )
        self.relu = nn.ReLU()

        self.fc_mu = nn.Linear(
            intermediate_dim,
            latent_dim,
        )
        self.fc_var = nn.Linear(
            intermediate_dim,
            latent_dim,
        )

    def reparameterize(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return eps * std + mu

    def forward(self, x):
        x = self.encoder_conv(x)

        # flatten, but ignore batch
        x = torch.flatten(x, start_dim=1)

        x = self.fc_intermediate(x)
        x = self.relu(x)

        # split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return [mu, log_var]


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        latent_dim: int,
        kernel_size: int = 5,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 1,
        hidden_dims: List[int] = None,
        intermediate_dim: int = 128,
        in_H: int = 32,
    ) -> None:
        super(Decoder, self).__init__()

        if hidden_dims is None:
            hidden_dims = [16, 32, 64, 128, 256]

        self.final_dim = hidden_dims[-1]

        # figure out final size of image after convs
        out_H = int(in_H / (2 ** len(hidden_dims)))
        self.out_H = out_H

        self.decoder_input = nn.Linear(
            latent_dim,
            intermediate_dim,
        )

        self.decoder_upsize = nn.Linear(
            intermediate_dim,
            hidden_dims[-1] * out_H * out_H,
        )

        decoder_blocks = []

        # loop over hidden dims in reverse
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):

            block = ConvTransposeBlock2D(
                in_channels=hidden_dims[i],
                out_channels=hidden_dims[i + 1],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            )
            decoder_blocks.append(block)

        self.decoder_conv = nn.Sequential(*decoder_blocks)
        self.relu = nn.ReLU()

        final_block = [
            ConvTransposeBlock2D(
                in_channels=hidden_dims[-1],
                out_channels=hidden_dims[-1],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            ),
            nn.Conv2d(
                hidden_dims[-1],
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.Sigmoid(),
        ]
        self.final_block = nn.Sequential(*final_block)

    def forward(self, z: torch.Tensor):

        # fc from latent to intermediate
        x = self.decoder_input(z)
        x = self.relu(x)
        x = self.decoder_upsize(x)
        x = self.relu(x)
        # reshape

        x = x.view((-1, self.final_dim, self.out_H, self.out_H))
        x = self.decoder_conv(x)

        x = self.final_block(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, latent_dim: int = 2):
        super(Discriminator, self).__init__()

        self.discriminator = nn.Sequential(
                nn.Linear(latent_dim, 1),
                nn.Sigmoid(),
            )
    def forward(self, z):
        N, L = z.shape

        half_N = int(N / 2)
        half_L = int(L / 2)

        z1 = z[:half_N, :half_L]
        z2 = z[half_N:, :half_L]
        s1 = z[:half_N, half_L:]
        s2 = z[half_N:, half_L:]

        q_bar = torch.cat(
            [torch.cat([s1, z2], dim=1),
            torch.cat([s2, z1], dim=1)],
            dim=0)
        q = torch.cat(
            [torch.cat([s1, z1], dim=1),
            torch.cat([s2, z2], dim=1)],
            dim=0)
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
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z)
        return [decoded, mu, log_var, z]


# contrastive-vae-no-bias
class CVAE(nn.Module):

    def __init__(
        self,
        s_encoder,
        z_encoder,
        decoder,
        latent_dim: int = 2,
        disentangle: bool = False,
    ):
        super(CVAE, self).__init__()

        # instantiate two encoders q_s and q_z
        self.qs = s_encoder
        self.qz = z_encoder
        # instantiate one shared decoder
        self.decoder = decoder

    def reparameterize(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return eps * std + mu

    def forward(self, tg_inputs, bg_inputs) -> torch.Tensor:

        # step 1: pass target features through irrelevant
        # and salient encoders.

        # irrelevant variables
        tg_z_mu, tg_z_log_var = self.qz(tg_inputs)
        tg_z = self.reparameterize(tg_z_mu, tg_z_log_var)
        # salient variables
        tg_s_mu, tg_s_log_var = self.qs(tg_inputs)
        tg_s = self.reparameterize(tg_s_mu, tg_s_log_var)

        # step 2: pass background features through just
        # the irrelevant encoder

        bg_z_mu, bg_z_log_var = self.qz(bg_inputs)
        bg_z = self.reparameterize(bg_z_mu, bg_z_log_var)

        # step 3: decode

        # decode the target outputs using both the salient and
        # irrelevant features
        tg_outputs = self.decoder(torch.cat([tg_s, tg_z], dim=-1))
        # we decode the background outputs using only the irrelevant
        # features
        zeros = torch.zeros_like(bg_z)
        bg_outputs = self.decoder(torch.cat([zeros, bg_z], dim=-1))

        out_dict = {
            "tg_out": tg_outputs,
            "bg_out": bg_outputs,
            "tg_s_mu": tg_s_mu,
            "tg_s_log_var": tg_s_log_var,
            "tg_s": tg_s,
            "tg_z_mu": tg_z_mu,
            "tg_z_log_var": tg_z_log_var,
            "bg_z_mu": bg_z_mu,
            "bg_z_log_var": bg_z_log_var,
        }

        return out_dict


if __name__ == "__main__":

    encoder = Encoder(
        in_channels=1,
        latent_dim=2,
        kernel_size=3,
        stride=2,
        padding=1,
        intermediate_dim=128,
        in_H=64,
    )

    decoder = Decoder(
        latent_dim=2,
        kernel_size=3,
        stride=2,
        padding=1,
        output_padding=1,
        intermediate_dim=128,
        out_channels=1,
        in_H=64,
    )

    model = VAE(encoder=encoder, decoder=decoder,)
    model = model.to("cuda")

    test = torch.rand(size=(100, 1, 64, 64)).to("cuda")

    out, mu, log_var, z = model(test)
    print (out.shape)
