import torch
from torch import nn
from typing import List, Tuple, Union
import torchvision
import torchvision.transforms.functional as F


class FinetuneResnet(nn.Module):

    def __init__(
        self,
        pretrained_model,
        representation_dims: int = 512,
        latent_dims: int = 3,
        apply_norm: bool = False,
    ):
        super(FinetuneResnet, self).__init__()

        self.representation = pretrained_model
        self.latent = nn.Sequential(
                nn.Linear(representation_dims, representation_dims),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(representation_dims, latent_dims),
            )
        self.relu = nn.ReLU()

        self.apply_norm = apply_norm

    def forward(self, x):
        # produce output of final conv layer/avg pool
        representation = self.representation(x)

        # relu non-linearity in encoder
        representation = self.relu(representation)
        if self.apply_norm:
            representation = torch.nn.functional.normalize(representation, p=2, dim=1)
        latent = self.latent(representation)        
        latent = self.relu(latent)
        if self.apply_norm:
            latent = torch.nn.functional.normalize(latent, p=2, dim=1)
        return representation, latent

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
        bias: bool = False,
    ):
        super(ConvBlock2D, self).__init__()

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
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
        bias: bool = False,
    ):
        super(ConvTransposeBlock2D, self).__init__()

        conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
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
        in_HW: Tuple[int] = (32, 32),
    ) -> None:
        super(CNN, self).__init__()

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
                    activation=True,
                    batch_norm=True,
                    bias=True,
                )

            encoder_blocks.append(block)

            in_channels = h_dim

        self.encoder_conv = nn.Sequential(*encoder_blocks)

        self.fc_intermediate = nn.Linear(
            hidden_dims[-1] * out_H * out_W,
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
        x = self.dropout(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


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
                [nn.Linear(hidden_dims[hi - 1], hidden_dims[hi]),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(
                    hidden_dims[hi],
                ),]
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
        
        self.softmax = nn.LogSoftmax(1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.fc(x)
        x = self.fc_out(x)
        # x = self.softplus(x)
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
        bias = False

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
                bias=bias,
            )
            encoder_blocks.append(block)

            in_channels = h_dim

        self.encoder_conv = nn.Sequential(*encoder_blocks)

        self.fc_intermediate = nn.Linear(
            hidden_dims[-1] * out_H * out_W,
            intermediate_dim,
            bias=bias,
        )

        self.fc_mu = nn.Linear(
            intermediate_dim,
            latent_dim,
            bias=bias,
        )
        self.fc_var = nn.Linear(
            intermediate_dim,
            latent_dim,
            bias=bias,
        )

        self.relu = nn.LeakyReLU(0.2)


    def forward(self, x):
        # print ("BEFORE ENCODING", x.min(), x.max())
        x = self.encoder_conv(x)
        # print ("BEFORE FLATTENING", x.min(), x.max())
        # flatten, but ignore batch
        x = torch.flatten(x, start_dim=1)
        # print ("AFTER FLATTENING", x.min(), x.max())

        x = self.fc_intermediate(x)
        x = self.relu(x)

        # print ("AFTER INTERMEIDATE", x.min(), x.max())

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

        bias = False

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
            bias=bias,
        )

        self.decoder_upsize = nn.Linear(
            intermediate_dim,
            hidden_dims[-1] * out_H * out_W,
            bias=bias,
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
                bias=bias,
            )
            decoder_blocks.append(block)

        self.decoder_conv = nn.Sequential(*decoder_blocks)
        self.relu = nn.LeakyReLU(0.2)

        # NOTE: no batch norm and no ReLu in final block
        final_block = [
            ConvTransposeBlock2D(
                in_channels=hidden_dims[-1],
                out_channels=hidden_dims[-1],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                batch_norm=True,
                activation=True,
                bias=bias,
            ),
            ConvBlock2D(
                in_channels=hidden_dims[-1],
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=(1, 1),
                padding=padding,
                batch_norm=False,
                activation=False,
                bias=bias,
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
            [torch.cat([s1, z1], dim=1),
            torch.cat([s2, z2], dim=1)],
            dim=0)
        
        q_bar = torch.cat(
            [torch.cat([s1, z2], dim=1),
            torch.cat([s2, z1], dim=1)],
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
        # we decode the "foreground" using just the salient features
        fg_outputs = self.decoder(torch.cat([tg_s, torch.zeros_like(tg_z)], dim=1))


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
