import torch
from torch import nn
from typing import List, Tuple, Union
import torchvision
import torchvision.transforms.functional as F


class Basic1DCNN(nn.Module):
    def __init__(self, in_W: int = 32, intermediate_dim: int = 128, latent_dim: int = 3):
        super(Basic1DCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=(1, 5), padding="same")
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 5), padding="same")
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (in_W // 4), intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, latent_dim)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = torch.amax(x, dim=2)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)

        return x

class FinetuneResnet(nn.Module):

    def __init__(
        self,
        pretrained_model,
        representation_dims: int = 512,
        latent_dims: int = 3,
    ):
        super(FinetuneResnet, self).__init__()

        pretrained_model.fc = nn.Identity()

        self.representation = pretrained_model
        self.latent = nn.Sequential(
                nn.Linear(representation_dims, representation_dims),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(representation_dims, latent_dims),
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        # produce output of final conv layer/avg pool
        representation = self.representation(x)

        # relu non-linearity in encoder
        representation = self.relu(representation)
        
        latent = self.latent(representation)        
        latent = self.relu(latent)
        
        return latent

class ConvBlock2D(nn.Module):

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]],
        padding: Union[int, Tuple[int]],
        batch_norm: bool = True,
        activation: bool = True,
    ):
        super(ConvBlock2D, self).__init__()

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

        relu = nn.LeakyReLU(0.2)
        norm = nn.BatchNorm2d(out_channels)
        layers = [conv]
        if activation:
            layers.append(relu)
        if batch_norm:
            layers.append(norm)
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
        stride: Union[int, Tuple[int]],
        padding: Union[int, Tuple[int]],
        output_padding: Union[int, Tuple[int]],
        batch_norm: bool = True,
        activation: bool = True,
    ):
        super(ConvTransposeBlock2D, self).__init__()

        conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=False,
        )

        relu = nn.LeakyReLU(0.2)
        norm = nn.BatchNorm2d(out_channels)
        layers = [conv]
        if activation:
            layers.append(relu)
        if batch_norm:
            layers.append(norm)
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        hidden_dims: List[int] = None,
        intermediate_dim: int = 128,
        in_H: int = 32,
    ) -> None:
        super(CNN, self).__init__()

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
        self.dropout = nn.Dropout(0.2)

        self.fc1 = nn.Linear(
            intermediate_dim,
            intermediate_dim,
        )
        self.fc2 = nn.Linear(
            intermediate_dim,
            latent_dim,
        )

    def forward(self, x):
        x = self.encoder_conv(x)

        # flatten, but ignore batch
        x = torch.flatten(x, start_dim=1)

        x = self.fc_intermediate(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.fc1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.fc2(x)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        latent_dim: int,
        kernel_size: Union[int, Tuple[int]] = 5,
        stride: Union[int, Tuple[int]] = 2,
        padding: Union[int, Tuple[int]] = 1,
        hidden_dims: List[int] = None,
        intermediate_dim: int = 128,
        in_HW: Tuple[int] = (32, 32),
    ) -> None:
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [16, 32, 64, 128, 256]

        # figure out final size of image after convs
        out_H, out_W = in_HW
        out_W //= 2 ** len(hidden_dims)

        # image height will only decrease if we're using square filters
        if type(kernel_size) is int or (
            len(kernel_size) > 1
            and kernel_size[0] == kernel_size[1]
            and kernel_size[0] > 1
        ):
            out_H //= 2 ** len(hidden_dims)


        encoder_blocks = []
        for h_dim in hidden_dims:
            # initialize convolutional block
            block = ConvBlock2D(
                in_channels=in_channels,
                out_channels=h_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                batch_norm=True,
                activation=True,
            )
            encoder_blocks.append(block)

            in_channels = h_dim

        self.encoder_conv = nn.Sequential(*encoder_blocks)

        self.fc_intermediate = nn.Linear(
            hidden_dims[-1] * out_H * out_W,
            intermediate_dim,
        )

        self.fc_mu = nn.Linear(
            intermediate_dim,
            latent_dim,
        )
        self.fc_var = nn.Linear(
            intermediate_dim,
            latent_dim,
        )

        self.relu = nn.LeakyReLU(0.2)


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
        kernel_size: Union[int, Tuple[int]] = (5, 5),
        stride: Union[int, Tuple[int]] = 2,
        padding: Union[int, Tuple[int]] = 1,
        output_padding: Union[int, Tuple[int]] = 1,
        hidden_dims: List[int] = None,
        intermediate_dim: int = 128,
        in_HW: Tuple[int] = (32, 32),
    ) -> None:
        super(Decoder, self).__init__()

        if hidden_dims is None:
            hidden_dims = [16, 32, 64, 128, 256]

        # figure out final dimension to which we 
        # need to reshape our filters before decoding
        self.final_dim = hidden_dims[-1]

        # figure out final size of image after convs
        out_H, out_W = in_HW
        out_W //= 2 ** len(hidden_dims)
        # image height will only decrease if we're using square filters
        if type(kernel_size) is int or (
            len(kernel_size) > 1
            and kernel_size[0] == kernel_size[1]
            and kernel_size[0] > 1
        ):
            out_H //= 2 ** len(hidden_dims)

        self.out_H = out_H
        self.out_W = out_W

        self.decoder_input = nn.Linear(
            latent_dim,
            intermediate_dim,
        )

        self.decoder_upsize = nn.Linear(
            intermediate_dim,
            hidden_dims[-1] * out_H * out_W,
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
                batch_norm=True,
                activation=True,
            )
            decoder_blocks.append(block)

        self.decoder_conv = nn.Sequential(*decoder_blocks)
        self.relu = nn.LeakyReLU(0.2)

        # NOTE: no batch norm and no ReLu in final block
        final_block = [
            ConvTransposeBlock2D(
                in_channels=hidden_dims[-1],
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                batch_norm=False,
                activation=False,
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

        x = x.view((-1, self.final_dim, self.out_H, self.out_W))
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

        # step 2: pass background features through
        # the irrelevant and salient encoders

        bg_z_mu, bg_z_log_var = self.qz(bg_inputs)
        bg_z = self.reparameterize(bg_z_mu, bg_z_log_var)

        bg_s_mu, bg_s_log_var = self.qs(bg_inputs)
        bg_s = self.reparameterize(bg_s_mu, bg_s_log_var)

        # step 3: decode

        # decode the target outputs using both the salient and
        # irrelevant features
        tg_outputs = self.decoder(torch.cat([tg_s, tg_z], dim=1))
        # we decode the background outputs using only the irrelevant
        # features
        bg_outputs = self.decoder(torch.cat([torch.zeros_like(bg_s), bg_z], dim=1))

        fg_outputs = self.decoder(torch.cat([tg_s, torch.zeros_like(tg_z)], dim=1))


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
        }

        return out_dict


if __name__ == "__main__":

    KERNEL_SIZE = (5, 5) #(1, 5)
    STRIDE = (2, 2) #(1, 2)
    PADDING = (2, 2) #(0, 2)
    OUTPUT_PADDING = (1, 1) #(0, 1)
    INTERMEDIATE_DIM = 64
    IN_HW = (64, 64)# (100, 32)
    HIDDEN_DIMS = [32, 64, 128, 256]
    LATENT_S = 2
    LATENT_Z = 4

    s_encoder = Encoder(
        in_channels=3,
        latent_dim=LATENT_S,
        kernel_size=KERNEL_SIZE,
        stride=STRIDE,
        padding=PADDING,
        intermediate_dim=INTERMEDIATE_DIM,
        in_HW=IN_HW,
        hidden_dims=HIDDEN_DIMS,
    )

    z_encoder = Encoder(
        in_channels=3,
        latent_dim=LATENT_Z,
        kernel_size=KERNEL_SIZE,
        stride=STRIDE,
        padding=PADDING,
        intermediate_dim=INTERMEDIATE_DIM,
        in_HW=IN_HW,
        hidden_dims=HIDDEN_DIMS,
    )

    decoder = Decoder(
        out_channels=3,
        latent_dim=LATENT_S,
        kernel_size=KERNEL_SIZE,
        stride=STRIDE,
        padding=PADDING,
        intermediate_dim=INTERMEDIATE_DIM,
        output_padding=OUTPUT_PADDING,
        in_HW=IN_HW,
        hidden_dims=HIDDEN_DIMS,
    )

    model = VAE(encoder=s_encoder, decoder=decoder,)
    model = model.to("cpu")

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print (pytorch_total_params)

    test = torch.rand(size=(100, 3, 64, 64)).to("cpu")

    out, mu, log_var, z = model(test)
    print (out.shape)

    decoder = Decoder(
        out_channels=3,
        latent_dim=LATENT_S + LATENT_Z,
        kernel_size=KERNEL_SIZE,
        stride=STRIDE,
        padding=PADDING,
        intermediate_dim=INTERMEDIATE_DIM,
        output_padding=OUTPUT_PADDING,
        in_HW=IN_HW,
        hidden_dims=HIDDEN_DIMS,
    )
    print (decoder.final_dim)

    model = CVAE(s_encoder=s_encoder, z_encoder=z_encoder, decoder=decoder,)
    model = model.to("cpu")

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print (pytorch_total_params)

    x = torch.rand(size=(100, 3, 64, 64)).to("cpu")
    x_ = torch.rand(size=(100, 3, 64, 64)).to("cpu")

    cvae_dict = model(x, x_)
    print ([(k, v.shape) for k,v in cvae_dict.items()])
