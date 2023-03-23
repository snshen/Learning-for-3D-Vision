import torch
import torch.nn.functional as F

from torch import autograd

from ray_utils import RayBundle

from a4.lighting_functions import relighting_dict

# Sphere SDF class
class SphereSDF(torch.nn.Module):
    def __init__(
        self,
        cfg = None,
        center=[0,0,0], 
        radius=[1.0],
        c_opt = True,
        r_opt = False

    ):
        super().__init__()

        if cfg != None:
            center = cfg.center.val
            radius = cfg.radius.val
            c_opt = True
            r_opt = False

        self.center = torch.nn.Parameter(
            torch.tensor(center).float().unsqueeze(0), requires_grad=c_opt
        )
        self.radius = torch.nn.Parameter(
            torch.tensor(radius).float(), requires_grad=r_opt
        )

    def forward(self, points):
        points = points.view(-1, 3)

        return torch.linalg.norm(
            points - self.center,
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

    def forward(self, points):
        points = points.view(-1, 3)
        diff = torch.abs(points - self.center) - self.side_lengths / 2.0

        signed_distance = torch.linalg.norm(
            torch.maximum(diff, torch.zeros_like(diff)),
            dim=-1
        ) + torch.minimum(torch.max(diff, dim=-1)[0], torch.zeros_like(diff[..., 0]))

        return signed_distance.unsqueeze(-1)


# Torus SDF class
class TorusSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )
        self.radii = torch.nn.Parameter(
            torch.tensor(cfg.radii.val).float().unsqueeze(0), requires_grad=cfg.radii.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)
        diff = points - self.center
        q = torch.stack(
            [
                torch.linalg.norm(diff[..., :2], dim=-1) - self.radii[..., 0],
                diff[..., -1],
            ],
            dim=-1
        )
        return (torch.linalg.norm(q, dim=-1) - self.radii[..., 1]).unsqueeze(-1)


sdf_dict = {
    'sphere': SphereSDF,
    'box': BoxSDF,
    'torus': TorusSDF,
}
    
class SDFSurface(torch.nn.Module):
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
    
    def get_distance(self, points):
        points = points.view(-1, 3)
        return self.sdf(points)

    def get_color(self, points):
        points = points.view(-1, 3)

        # Outputs
        if self.rainbow:
            base_color = torch.clamp(
                torch.abs(points - self.sdf.center),
                0.02,
                0.98
            )
        else:
            base_color = 1.0

        return base_color * self.feature * points.new_ones(points.shape[0], 1)
    
    def forward(self, points):
        return self.get_distance(points)

  
class LargeSDFSurface(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.sdf = sdf_dict[cfg.sdf.type](
            cfg.sdf
        )

        #########
        self.sdfs=[]
        #tail
        self.sdfs.append(SphereSDF(center=torch.tensor([1.3,-0.15,-0.05]).to('cuda:0'), radius=torch.tensor(0.25).to('cuda:0')))
        self.sdfs.append(SphereSDF(center=torch.tensor([1.27,-0.15,-0.15]).to('cuda:0'), radius=torch.tensor(0.35).to('cuda:0')))
        self.sdfs.append(SphereSDF(center=torch.tensor([1.25,-0.15,-0.19]).to('cuda:0'), radius=torch.tensor(0.35).to('cuda:0')))
        #thigh
        self.sdfs.append(SphereSDF(center=torch.tensor([0.4,-0.45,0]).to('cuda:0'), radius=torch.tensor(0.7).to('cuda:0')))
        self.sdfs.append(SphereSDF(center=torch.tensor([0.43,-0.3,-0.1]).to('cuda:0'), radius=torch.tensor(0.7).to('cuda:0')))
        self.sdfs.append(SphereSDF(center=torch.tensor([0.44,-0.15,-0.2]).to('cuda:0'), radius=torch.tensor(0.7).to('cuda:0')))
        #back leg
        self.sdfs.append(SphereSDF(center=torch.tensor([0.12,-0.8,-0.32]).to('cuda:0'), radius=torch.tensor(0.35).to('cuda:0')))
        #haunches
        self.sdfs.append(SphereSDF(center=torch.tensor([0.35,0,0]).to('cuda:0'), radius=torch.tensor(1.05).to('cuda:0')))
        self.sdfs.append(SphereSDF(center=torch.tensor([0.2,0,0]).to('cuda:0'), radius=torch.tensor(1.0).to('cuda:0')))
        self.sdfs.append(SphereSDF(center=torch.tensor([0.18,0,0]).to('cuda:0'), radius=torch.tensor(0.98).to('cuda:0')))
        #front legs
        self.sdfs.append(SphereSDF(center=torch.tensor([-0.55,-0.7,-0.38]).to('cuda:0'), radius=torch.tensor(0.22).to('cuda:0')))
        self.sdfs.append(SphereSDF(center=torch.tensor([-0.6,-0.7,-0.42]).to('cuda:0'), radius=torch.tensor(0.18).to('cuda:0')))
        self.sdfs.append(SphereSDF(center=torch.tensor([-0.65,-0.25,-0.5]).to('cuda:0'), radius=torch.tensor(0.25).to('cuda:0')))
        self.sdfs.append(SphereSDF(center=torch.tensor([-0.75,-0.25,-0.53]).to('cuda:0'), radius=torch.tensor(0.2).to('cuda:0')))
        #chest
        self.sdfs.append(SphereSDF(center=torch.tensor([-0.38,-0.05,0.28]).to('cuda:0'), radius=torch.tensor(0.8).to('cuda:0')))
        self.sdfs.append(SphereSDF(center=torch.tensor([-0.53,-0.1,0.32]).to('cuda:0'), radius=torch.tensor(0.78).to('cuda:0')))
        #head
        self.sdfs.append(SphereSDF(center=torch.tensor([-0.65,-0.15,1.05]).to('cuda:0'), radius=torch.tensor(0.58).to('cuda:0')))
        self.sdfs.append(SphereSDF(center=torch.tensor([-0.65,-0.25,1.05]).to('cuda:0'), radius=torch.tensor(0.55).to('cuda:0')))
        self.sdfs.append(SphereSDF(center=torch.tensor([-0.65,-0.48,1.0]).to('cuda:0'), radius=torch.tensor(0.43).to('cuda:0')))
        #ears
        self.sdfs.append(SphereSDF(center=torch.tensor([-0.55,0.02,1.5]).to('cuda:0'), radius=torch.tensor(0.18).to('cuda:0')))
        self.sdfs.append(SphereSDF(center=torch.tensor([-0.50,0.14,1.56]).to('cuda:0'), radius=torch.tensor(0.21).to('cuda:0')))
        self.sdfs.append(SphereSDF(center=torch.tensor([-0.45,0.26,1.62]).to('cuda:0'), radius=torch.tensor(0.21).to('cuda:0')))
        self.sdfs.append(SphereSDF(center=torch.tensor([-0.40,0.38,1.68]).to('cuda:0'), radius=torch.tensor(0.20).to('cuda:0')))
        self.sdfs.append(SphereSDF(center=torch.tensor([-0.35,0.50,1.74]).to('cuda:0'), radius=torch.tensor(0.18).to('cuda:0')))
        self.sdfs.append(SphereSDF(center=torch.tensor([-0.30,0.62,1.80]).to('cuda:0'), radius=torch.tensor(0.15).to('cuda:0')))

        self.sdfs.append(SphereSDF(center=torch.tensor([-0.93,0.01,1.45]).to('cuda:0'), radius=torch.tensor(0.18).to('cuda:0')))
        self.sdfs.append(SphereSDF(center=torch.tensor([-1.03,0.11,1.50]).to('cuda:0'), radius=torch.tensor(0.21).to('cuda:0')))
        self.sdfs.append(SphereSDF(center=torch.tensor([-1.13,0.21,1.55]).to('cuda:0'), radius=torch.tensor(0.21).to('cuda:0')))
        self.sdfs.append(SphereSDF(center=torch.tensor([-1.23,0.31,1.60]).to('cuda:0'), radius=torch.tensor(0.20).to('cuda:0')))
        self.sdfs.append(SphereSDF(center=torch.tensor([-1.33,0.41,1.65]).to('cuda:0'), radius=torch.tensor(0.18).to('cuda:0')))
        self.sdfs.append(SphereSDF(center=torch.tensor([-1.43,0.51,1.70]).to('cuda:0'), radius=torch.tensor(0.15).to('cuda:0')))


        # self.sdfs.append(SphereSDF(center=torch.tensor([0.8,0,0.5]).to('cuda:0'), radius=torch.tensor(0.7).to('cuda:0')))

        #########
        self.rainbow = cfg.feature.rainbow if 'rainbow' in cfg.feature else False
        self.feature = torch.nn.Parameter(
            torch.ones_like(torch.tensor(cfg.feature.val).float().unsqueeze(0)), requires_grad=cfg.feature.opt
        )
    
    def get_distance(self, points):
        points = points.view(-1, 3)
        dists = []
        for sdf in self.sdfs:
            dists.append(sdf(points))
        dists = torch.cat(dists, dim=1)
        dist,_ = torch.min(dists,dim=1,keepdim=True)
        return dist

    def get_color(self, points):
        points = points.view(-1, 3)

        # Outputs
        if self.rainbow:
            base_color = torch.clamp(
                torch.abs(points - self.sdf.center),
                0.02,
                0.98
            )
        else:
            base_color = 1.0

        return base_color * self.feature * points.new_ones(points.shape[0], 1)
    
    def forward(self, points):
        return self.get_distance(points)


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


class NeuralSurface(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        # TODO (Q2): Implement Neural Surface MLP to output per-point SDF
        # TODO (Q3): Implement Neural Surface MLP to output per-point color
        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        self.embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim
        
        self.n_dist = cfg.n_layers_distance
        self.n_color = cfg.n_layers_color
        hidden_dims = [cfg.n_hidden_neurons_distance, cfg.n_hidden_neurons_color]
        self.skip_ind = self.n_dist//2

        self.layers_dist = torch.nn.ModuleList()
        for layeri in range(self.n_dist):
            if layeri == 0:
                self.layers_dist.append(torch.nn.Linear(self.embedding_dim_xyz, hidden_dims[0]))
            elif layeri == self.skip_ind:
                self.layers_dist.append(torch.nn.Linear(self.embedding_dim_xyz+hidden_dims[0], hidden_dims[0]))
            else:
                self.layers_dist.append(torch.nn.Linear(hidden_dims[0], hidden_dims[0]))
        self.relu = torch.nn.ReLU()
        self.layer_sigma = torch.nn.Linear(hidden_dims[0], 1)
        
        self.rgb = torch.nn.ModuleList()
        for layeri in range(self.n_color):
            if layeri == 0: 
                self.rgb.append(torch.nn.Linear(3+hidden_dims[0], hidden_dims[1]))
            else: 
                self.rgb.append(torch.nn.Linear(hidden_dims[1], hidden_dims[1]))
            self.rgb.append(torch.nn.ReLU())
        self.rgb.append(torch.nn.Linear(hidden_dims[1], 3))
        self.rgb.append(torch.nn.Sigmoid())

    def get_distance(
        self,
        points
    ):
        '''
        TODO: Q2
        Output:
            distance: N X 1 Tensor, where N is number of input points
        '''
        x = points.view(-1, 3)
        harmonic_xyz = self.harmonic_embedding_xyz(x)
        
        for layeri, layer in enumerate(self.layers_dist):
            if layeri == 0: x = harmonic_xyz
            elif layeri == self.skip_ind: x = torch.cat((x, harmonic_xyz), dim=-1)
            x = layer(x)
            x = self.relu(x)

        return self.layer_sigma(x)
    
    def get_color(
        self,
        points
    ):
        '''
        TODO: Q3
        Output:
            distance: N X 3 Tensor, where N is number of input points
        '''
        x = points.view(-1, 3)
        xyz = points.view(-1, 3)
        harmonic_xyz = self.harmonic_embedding_xyz(x)

        for layeri, layer in enumerate(self.layers_dist):
            if layeri == 0: x = harmonic_xyz
            elif layeri == self.skip_ind: x = torch.cat((x, harmonic_xyz), dim=-1)
            x = layer(x)
            x = self.relu(x)

        x = torch.cat((x, xyz), dim=-1)
        for layeri, layer in enumerate(self.rgb):
            x = layer(x)

        return x
    
    def get_distance_color(
        self,
        points
    ):
        '''
        TODO: Q3
        Output:
            distance, points: N X 1, N X 3 Tensors, where N is number of input points
        You may just implement this by independent calls to get_distance, get_color
            but, depending on your MLP implementation, it maybe more efficient to share some computation
        '''
        x = points.view(-1, 3)
        xyz = points.view(-1, 3)
        harmonic_xyz = self.harmonic_embedding_xyz(x)
        
        for layeri, layer in enumerate(self.layers_dist):
            if layeri == 0: x = harmonic_xyz
            elif layeri == self.skip_ind: x = torch.cat((x, harmonic_xyz), dim=-1)
            x = layer(x)
            x = self.relu(x)
        distance =  self.layer_sigma(x)
        
        x = torch.cat((x, xyz), dim=-1)
        for layeri, layer in enumerate(self.rgb):
            x = layer(x)
        points = x

        return distance, points
        
    def forward(self, points):
        return self.get_distance(points)

    def get_distance_and_gradient(
        self,
        points
    ):
        has_grad = torch.is_grad_enabled()
        points = points.view(-1, 3)

        # Calculate gradient with respect to points
        with torch.enable_grad():
            points = points.requires_grad_(True)
            distance = self.get_distance(points)
            gradient = autograd.grad(
                distance,
                points,
                torch.ones_like(distance, device=points.device),
                create_graph=has_grad,
                retain_graph=has_grad,
                only_inputs=True
            )[0]
        
        return distance, gradient

    def get_surface_normal(
        self,
        points
    ):
        '''
        TODO: Q4
        Input:
            points: N X 3 Tensor, where N is number of input points
        Output:
            surface_normal: N X 3 Tensor, where N is number of input points
        '''
        _, gradient = self.get_distance_and_gradient(points)
        normal = torch.nn.functional.normalize(gradient, dim=-1).view(-1, 3)
        return normal


implicit_dict = {
    'sdf_surface': SDFSurface,
    'large_sdf_surface': LargeSDFSurface,
    'neural_surface': NeuralSurface,
}
