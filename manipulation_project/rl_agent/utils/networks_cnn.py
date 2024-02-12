import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_out_shape(in_shape, layers, attn=False):
    x = torch.randn(*in_shape).unsqueeze(0)
    if attn:
        return layers(x, x, x).squeeze(0).shape
    else:
        return layers(x).squeeze(0).shape


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability"""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function, see appendix C from https://arxiv.org/pdf/1812.05905.pdf"""
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def orthogonal_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class NormalizeImg(nn.Module):
    def __init__(self, mean_zero=False):
        super().__init__()
        self.mean_zero = mean_zero

    def forward(self, x):
        if self.mean_zero:
            return x / 255. - 0.5
        return x / 255.


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):
    def __init__(self, obs_shape=None, out_dim=None):
        super().__init__()
        self.out_shape = obs_shape
        self.out_dim = out_dim

    def forward(self, x):
        return x


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_query = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.conv_key = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.conv_value = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.in_channels = in_channels

    def forward(self, query, key, value):
        N, C, H, W = query.shape
        assert query.shape == key.shape == value.shape, "Key, query and value inputs must be of the same dimensions in this implementation"
        q = self.conv_query(query).reshape(N, C, H * W)  # .permute(0, 2, 1)
        k = self.conv_key(key).reshape(N, C, H * W)  # .permute(0, 2, 1)
        v = self.conv_value(value).reshape(N, C, H * W)  # .permute(0, 2, 1)
        attention = k.transpose(1, 2) @ q / C ** 0.5
        attention = attention.softmax(dim=1)
        output = v @ attention
        output = output.reshape(N, C, H, W)
        return query + output  # Add with query and output


class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm,
                 contextualReasoning=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.attn = SelfAttention(dim[0])
        self.context = contextualReasoning
        temp_shape = _get_out_shape(dim, self.attn, attn=True)
        self.out_shape = _get_out_shape(temp_shape, nn.Flatten())
        self.apply(orthogonal_init)

    def forward(self, query, key, value):
        x = self.attn(self.norm1(query), self.norm2(key), self.norm3(value))
        if self.context:
            return x
        else:
            x = x.flatten(start_dim=1)
            return x


class SharedCNN(nn.Module):
    def __init__(self, obs_shape, num_layers=11, num_filters=32, mean_zero=False):
        super().__init__()
        assert len(obs_shape) == 3
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.layers = [NormalizeImg(mean_zero), nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        for _ in range(1, num_layers):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.layers = nn.Sequential(*self.layers)
        self.out_shape = _get_out_shape(obs_shape, self.layers)
        self.apply(orthogonal_init)

    def forward(self, x):
        return self.layers(x)


class HeadCNN(nn.Module):
    def __init__(self, in_shape, num_layers=0, num_filters=32, flatten=True):
        super().__init__()
        self.layers = []
        for _ in range(0, num_layers):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        if flatten:
            self.layers.append(Flatten())
        self.layers = nn.Sequential(*self.layers)
        self.out_shape = _get_out_shape(in_shape, self.layers)
        self.apply(orthogonal_init)

    def forward(self, x):
        return self.layers(x)


class Integrator(nn.Module):
    def __init__(self, in_shape_1, in_shape_2, num_filters=32, concatenate=True):
        super().__init__()
        self.relu = nn.ReLU()
        if concatenate:
            self.conv1 = nn.Conv2d(in_shape_1[0] + in_shape_2[0], num_filters, (1, 1))
        else:
            self.conv1 = nn.Conv2d(in_shape_1[0], num_filters, (1, 1))
        self.apply(orthogonal_init)

    def forward(self, x):
        x = self.conv1(self.relu(x))
        return x


class Encoder(nn.Module):
    def __init__(self, shared_cnn, head_cnn, projection, attention=None):
        super().__init__()
        self.shared_cnn = shared_cnn
        self.head_cnn = head_cnn
        self.projection = projection
        self.attention = attention
        self.out_dim = projection.out_dim

    def forward(self, x, detach=False):
        x = self.shared_cnn(x)
        x = self.head_cnn(x)
        if detach:
            x = x.detach()
        x = self.projection(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(orthogonal_init)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)


class MultiViewEncoder(nn.Module):
    def __init__(self, shared_cnn_1, shared_cnn_2, integrator, head_cnn, projection, attention1=None, attention2=None,
                 mlp1=None, mlp2=None, norm1=None, norm2=None, concatenate=True, contextualReasoning1=False,
                 contextualReasoning2=False):
        super().__init__()
        self.shared_cnn_1 = shared_cnn_1
        self.shared_cnn_2 = shared_cnn_2
        self.integrator = integrator
        self.head_cnn = head_cnn
        self.projection = projection
        self.relu = nn.ReLU()
        self.contextualReasoning1 = contextualReasoning1
        self.contextualReasoning2 = contextualReasoning2
        self.attention1 = attention1
        self.attention2 = attention2

        self.mlp1 = mlp1
        self.norm1 = norm1
        self.mlp2 = mlp2
        self.norm2 = norm2

        self.out_dim = projection.out_dim
        self.concatenate = concatenate

    def forward(self, x1, x2, detach=False):

        x1 = self.shared_cnn_1(x1)  # 3rd Person
        x2 = self.shared_cnn_2(x2)

        B, C, H, W = x1.shape

        if self.contextualReasoning1:
            x1 = self.attention1(x1, x2, x2)  # Contextual reasoning on 3rd person image based on 1st person image
            x1 = self.norm1(x1)
            x1 = x1.view(B, C, -1).permute(0, 2, 1)
            x1 = self.mlp1(x1).permute(0, 2, 1).contiguous().view(B, C, H, W)

        if self.contextualReasoning2:
            x2 = self.attention2(x2, x1, x1)  # Contextual reasoning on 1st person image based on 3rd person image
            x2 = self.norm2(x2)
            x2 = x2.view(B, C, -1).permute(0, 2, 1)
            x2 = self.mlp2(x2).permute(0, 2, 1).contiguous().view(B, C, H, W)

        if self.concatenate:
            # Concatenate features along channel dimension
            x = torch.cat((x1, x2), dim=1)  # 1, 64, 21, 21
        else:
            x = x1 + x2  # 1, 32, 21, 21

        x = self.integrator(x)
        x = self.head_cnn(x)

        if self.attention1 is not None and not self.contextualReasoning1:
            x = self.relu(self.attention1(x, x, x))

        if detach:
            x = x.detach()

        x = self.projection(x)

        return x


class Critic(nn.Module):
    def __init__(self, out_dim, projection_dim, state_shape, action_dim, hidden_dim, hidden_dim_state=16):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(out_dim, projection_dim),
                                        nn.LayerNorm(projection_dim), nn.Tanh())

        if state_shape:
            self.state_encoder = nn.Sequential(nn.Linear(state_shape[0], hidden_dim_state),
                                               nn.ReLU(inplace=True),
                                               nn.Linear(hidden_dim_state, projection_dim),
                                               nn.LayerNorm(projection_dim), nn.Tanh())
        else:
            self.state_encoder = None

        self.Q1 = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, action_dim))
        self.Q2 = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, action_dim))
        self.apply(orthogonal_init)

    def forward(self, obs, state=None):
        obs = self.projection(obs)

        if self.state_encoder:
            obs = obs + self.state_encoder(state)

        return self.Q1(obs), self.Q2(obs)

    def get_action(self, obs, state=None, argmax_dim=0):
        values, _ = self.forward(obs, state)
        return torch.argmax(values, dim=argmax_dim)


def make_encoder(image_size):
    obs_shape = (3, image_size, image_size)
    shared_1 = SharedCNN(obs_shape, 11, 32, False)
    shared_2 = SharedCNN(obs_shape, 11, 32, False)

    integrator = Integrator(shared_1.out_shape, shared_2.out_shape, 32, concatenate=True)  # Change channel dimensions of concatenated features

    assert shared_1.out_shape == shared_2.out_shape, 'Image features must be the same'

    head = HeadCNN(shared_1.out_shape, 3, 32, flatten=True)
    mlp_hidden_dim = int(shared_1.out_shape[0] * 4)

    attention1 = AttentionBlock(dim=shared_1.out_shape, contextualReasoning=True)
    mlp1 = Mlp(in_features=shared_1.out_shape[0], hidden_features=mlp_hidden_dim, act_layer=nn.GELU)
    norm1 = nn.LayerNorm(shared_1.out_shape)

    attention2 = AttentionBlock(dim=shared_1.out_shape, contextualReasoning=True)
    mlp2 = Mlp(in_features=shared_1.out_shape[0], hidden_features=mlp_hidden_dim, act_layer=nn.GELU)
    norm2 = nn.LayerNorm(shared_1.out_shape)

    encoder = MultiViewEncoder(
        shared_1,
        shared_2,
        integrator,
        head,
        Identity(out_dim=head.out_shape[0]),
        attention1,
        attention2,
        mlp1,
        mlp2,
        norm1,
        norm2,
        concatenate=True,
        contextualReasoning1=True,
        contextualReasoning2=True
        )

    return encoder
