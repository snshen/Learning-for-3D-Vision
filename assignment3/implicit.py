import torch
import torch.nn.functional as F

from ray_utils import RayBundle


# Sphere SDF class
class SphereSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.radius = torch.nn.Parameter(
            torch.tensor(cfg.radius.val).float(), requires_grad=cfg.radius.opt
        )
        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )

    def forward(self, ray_bundle):
        sample_points = ray_bundle.sample_points.view(-1, 3)

        return torch.linalg.norm(
            sample_points - self.center,
            dim=-1,
            keepdim=True
        ) - self.radius


# Box SDF class
class BoxSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )
        self.side_lengths = torch.nn.Parameter(
            torch.tensor(cfg.side_lengths.val).float().unsqueeze(0), requires_grad=cfg.side_lengths.opt
        )

    def forward(self, ray_bundle):
        sample_points = ray_bundle.sample_points.view(-1, 3)
        diff = torch.abs(sample_points - self.center) - self.side_lengths / 2.0

        signed_distance = torch.linalg.norm(
            torch.maximum(diff, torch.zeros_like(diff)),
            dim=-1
        ) + torch.minimum(torch.max(diff, dim=-1)[0], torch.zeros_like(diff[..., 0]))

        return signed_distance.unsqueeze(-1)


sdf_dict = {
    'sphere': SphereSDF,
    'box': BoxSDF,
}


# Converts SDF into density/feature volume
class SDFVolume(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.sdf = sdf_dict[cfg.sdf.type](
            cfg.sdf
        )

        self.rainbow = cfg.feature.rainbow if 'rainbow' in cfg.feature else False
        self.feature = torch.nn.Parameter(
            torch.ones_like(torch.tensor(cfg.feature.val).float().unsqueeze(0)), requires_grad=cfg.feature.opt
        )

        self.alpha = torch.nn.Parameter(
            torch.tensor(cfg.alpha.val).float(), requires_grad=cfg.alpha.opt
        )
        self.beta = torch.nn.Parameter(
            torch.tensor(cfg.beta.val).float(), requires_grad=cfg.beta.opt
        )

    def _sdf_to_density(self, signed_distance):
        # Convert signed distance to density with alpha, beta parameters
        return torch.where(
            signed_distance > 0,
            0.5 * torch.exp(-signed_distance / self.beta),
            1 - 0.5 * torch.exp(signed_distance / self.beta),
        ) * self.alpha

    def forward(self, ray_bundle):
        sample_points = ray_bundle.sample_points.view(-1, 3)
        depth_values = ray_bundle.sample_lengths[..., 0]
        deltas = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                1e10 * torch.ones_like(depth_values[..., :1]),
            ),
            dim=-1,
        ).view(-1, 1)

        # Transform SDF to density
        signed_distance = self.sdf(ray_bundle)
        density = self._sdf_to_density(signed_distance)

        # Outputs
        if self.rainbow:
            base_color = torch.clamp(
                torch.abs(sample_points - self.sdf.center),
                0.02,
                0.98
            )
        else:
            base_color = 1.0

        out = {
            'density': -torch.log(1.0 - density) / deltas,
            'feature': base_color * self.feature * density.new_ones(sample_points.shape[0], 1)
        }

        return out


class HarmonicEmbedding(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        n_harmonic_functions: int = 6,
        omega0: float = 1.0,
        logspace: bool = True,
        include_input: bool = True,
    ) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", omega0 * frequencies, persistent=False)
        self.include_input = include_input
        self.output_dim = n_harmonic_functions * 2 * in_channels

        if self.include_input:
            self.output_dim += in_channels

    def forward(self, x: torch.Tensor):
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)

        if self.include_input:
            return torch.cat((embed.sin(), embed.cos(), x), dim=-1)
        else:
            return torch.cat((embed.sin(), embed.cos()), dim=-1)


class LinearWithRepeat(torch.nn.Linear):
    def forward(self, input):
        n1 = input[0].shape[-1]
        output1 = F.linear(input[0], self.weight[:, :n1], self.bias)
        output2 = F.linear(input[1], self.weight[:, n1:], None)
        return output1 + output2.unsqueeze(-2)


class MLPWithInputSkips(torch.nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        output_dim: int,
        skip_dim: int,
        hidden_dim: int,
        input_skips,
    ):
        super().__init__()

        layers = []

        for layeri in range(n_layers):
            if layeri == 0:
                dimin = input_dim
                dimout = hidden_dim
            elif layeri in input_skips:
                dimin = hidden_dim + skip_dim
                dimout = hidden_dim
            else:
                dimin = hidden_dim
                dimout = hidden_dim

            linear = torch.nn.Linear(dimin, dimout)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))

        self.mlp = torch.nn.ModuleList(layers)
        self._input_skips = set(input_skips)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y = x

        for li, layer in enumerate(self.mlp):
            if li in self._input_skips:
                y = torch.cat((y, z), dim=-1)

            y = layer(y)

        return y


# TODO (3.1): Implement NeRF MLP
class NeuralRadianceField(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        self.harmonic_embedding_dir = HarmonicEmbedding(3, cfg.n_harmonic_functions_dir)

        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim
        embedding_dim_dir = self.harmonic_embedding_dir.output_dim

        hidden_dims = [cfg.n_hidden_neurons_xyz, cfg.n_hidden_neurons_dir]
        self.layers_xyz_init = torch.nn.Linear(embedding_dim_xyz, hidden_dims[0])

        self.layers_xyz = torch.nn.ModuleList()
        for layeri in range(cfg.n_layers_xyz):
            if layeri == 0:
                self.layers_xyz.append(self.layers_xyz_init)
            elif layeri == 4:
                self.layers_xyz.append(torch.nn.Linear(embedding_dim_xyz+hidden_dims[0], hidden_dims[0]))
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_dims[0], hidden_dims[0]))
        self.relu = torch.nn.ReLU()

        self.layer_sigma = torch.nn.Sequential(
                torch.nn.Linear(hidden_dims[0], 1),
                torch.nn.ReLU()
            )
        
        self.layer_feature = torch.nn.Sequential(
                torch.nn.Linear(hidden_dims[0], hidden_dims[0]),
                torch.nn.ReLU()
            )
        
        self.layers_dir = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim_dir+hidden_dims[0], hidden_dims[1]),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dims[1], 3),
                torch.nn.Sigmoid()
            )
        
        torch.nn.init.xavier_normal_(self.layers_xyz_init.weight)

    def forward(self, ray_bundle):
        
        # first processes the input 3D coordinate x with 8 fully-connected layers (using ReLU activations and 256 channels per layer)
        x = ray_bundle.sample_points
        harmonic_xyz = self.harmonic_embedding_xyz(x)
        
        for layeri, layer in enumerate(self.layers_xyz):
            if layeri == 0: x = harmonic_xyz
            elif layeri == 4: x = torch.cat((harmonic_xyz, x), dim=-1)

            x = layer(x)

            if layeri != 7: x = self.relu(x)

        # outputs σ and a 256-dimensional feature vector
        sigma = self.layer_sigma(x)
        feature = self.layer_feature(x)

        # feature vector is concatenated with the camera ray’s viewing direction and passed to fully-connected layer (using a ReLU activation and 128 channels) 
        # output the view-dependent RGB color
        harmonic_dir = self.harmonic_embedding_dir(ray_bundle.directions).unsqueeze(1)
        harmonic_dir = torch.tile(harmonic_dir, (1, feature.shape[1], 1))
        x = torch.cat((harmonic_dir, feature), dim=-1)
        rgb =  self.layers_dir(x)

        return {'density': sigma, 'feature': rgb}




volume_dict = {
    'sdf_volume': SDFVolume,
    'nerf': NeuralRadianceField,
}