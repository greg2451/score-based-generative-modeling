from torch import nn, cat, sin, cos, randn, sigmoid
from numpy import pi


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * pi
        return cat([sin(x_proj), cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


class ScoreNetColor(nn.Module):
    def __init__(self, marginal_prob_std, embed_dim=256):
        super(ScoreNetColor, self).__init__()

        # Embed
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        # ----------------
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=5,
            stride=1,
            dilation=1,
            padding=2,
        )
        self.norm1 = nn.BatchNorm2d(64)
        self.dense1 = Dense(embed_dim, 64)
        # ---------------- CHANGE IN THE NUMBER OF OUTPUTS
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=2,
            dilation=1,
            padding=1,
        )
        self.norm2 = nn.BatchNorm2d(128)
        self.dense2 = Dense(embed_dim, 128)

        self.conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            dilation=1,
            padding=1,
        )
        self.norm3 = nn.BatchNorm2d(128)
        self.dense3 = Dense(embed_dim, 128)

        self.conv4 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            dilation=1,
            padding=1,
        )
        self.norm4 = nn.BatchNorm2d(128)
        self.dense4 = Dense(embed_dim, 128)

        self.dilatedconv5 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            dilation=2,
            padding=2,
        )
        self.norm5 = nn.BatchNorm2d(128)
        self.dense5 = Dense(embed_dim, 128)

        self.conv6 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            dilation=1,
            padding=1,
        )
        self.norm6 = nn.BatchNorm2d(128)
        self.dense6 = Dense(embed_dim, 128)

        # ----------------- CHANGE IN THE NUMBER OF OUTPUTS
        self.deconv15 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=4,
            stride=2,
            dilation=1,
            padding=1,
        )
        self.norm15 = nn.BatchNorm2d(64)
        self.dense15 = Dense(embed_dim, 64)

        self.conv16 = nn.Conv2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=1,
            dilation=1,
            padding=1,
        )
        self.norm16 = nn.BatchNorm2d(32)
        self.dense16 = Dense(embed_dim, 32)

        self.output = nn.Conv2d(
            in_channels=32,
            out_channels=3,
            kernel_size=3,
            stride=1,
            dilation=1,
            padding=1,
        )

        self.inside_activation = nn.ReLU(True)
        self.output_activation = nn.Sigmoid()
        self.marginal_prob_std = marginal_prob_std
        self.act = lambda x: x * sigmoid(x)

    def forward(self, x, t):
        embed = self.act(self.embed(t))
        out = self.norm1(self.inside_activation(self.conv1(x))) + self.dense1(embed)
        out = self.norm2(self.inside_activation(self.conv2(out))) + self.dense2(embed)
        out = self.norm3(self.inside_activation(self.conv3(out))) + self.dense3(embed)
        out = self.norm4(self.inside_activation(self.conv4(out))) + self.dense4(embed)
        out = self.norm5(self.inside_activation(self.dilatedconv5(out))) + self.dense5(
            embed
        )
        out = self.norm6(self.inside_activation(self.conv6(out))) + self.dense6(embed)
        out = self.norm15(self.inside_activation(self.deconv15(out))) + +self.dense15(
            embed
        )
        out = self.norm16(self.inside_activation(self.conv16(out))) + self.dense16(
            embed
        )
        return (
            self.output_activation(self.output(out))
            / self.marginal_prob_std(t)[:, None, None, None]
        )


class ScoreNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, marginal_prob_std, channels=None, embed_dim=256):
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        if channels is None:
            channels = [32, 64, 128, 256]
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(
            channels[3], channels[2], 3, stride=2, bias=False
        )
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(
            channels[2] + channels[2],
            channels[1],
            3,
            stride=2,
            bias=False,
            output_padding=1,
        )
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(
            channels[1] + channels[1],
            channels[0],
            3,
            stride=2,
            bias=False,
            output_padding=1,
        )
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)

        # The swish activation function
        self.act = lambda x: x * sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.embed(t))
        # Encoding path
        h1 = self.conv1(x)
        # Incorporate information from t
        h1 += self.dense1(embed)
        # Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        # Decoding path
        h = self.tconv4(h4)
        # Skip connection from the encoding path
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(cat([h, h1], dim=1))

        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h
