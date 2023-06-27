import torch
import torch.nn as nn
import torch.nn.functional as F


class NeRF(nn.Module):
    def __init__(self, 
        D=8, # layder depth
        W=256, # layer width (dimension)
        input_ch=3, 
        input_ch_views=3, 
        output_ch=4,
        skips=[4],
        use_viewdirs=False 
    ) -> None:
        """
        
        """
        super().__init__()

        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs


        # NOTE: layers
        self.pts_linears = nn.ModuleList(
            [
                nn.Linear(input_ch, W)
            ] + 
            [
                nn.Linear(W, W) if i not in self.skips 
                else nn.Linear(W+input_ch, W) 
                for i in range(D-1)
            ]
        )
        self.views_linears = nn.ModuleList([
            nn.Linear(input_ch_views+W, W//2) # i.e., 128
        ])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1) # NOTE: density
            self.rgb_linear = nn.Linear(W//2, 3) # NOTE: input from `direction_linears`
        else:
            # NOTE: RGB+density inference without view direction condition
            self.output_linear = nn.Linear(W, output_ch) 


    def forward(self, x):
        # NOTE: split input into original/feature, 


        input_pts, input_views = torch.split(
            x, 
            [self.input_ch, self.input_ch_views], 
            dim=-1
        ) # (Batch, 63), (Batch, 27)
        h = input_pts

        # NOTE: Position layers (with input feeding in the middle layer)
        for idx, layer in enumerate(self.pts_linears):
            
            h = self.pts_linears[idx](h)

            # NOTE: we're not going to implement activation in the layer function, thus implement in here
            h = F.relu(h)

            # NOTE: be sure to concat input on 5th(idx==4) layer!
            if idx in self.skips:
                h = torch.cat([input_pts, h], dim=-1) # (Batch, 63+256)


        # NOTE: Direction layers (the length should be 1, as there is only a single layer)
        if self.use_viewdirs:            
            alpha = self.alpha_linear(h) # NOTE: infer density on the last layer
            feature = self.feature_linear(h) # NOTE: last layer: FCN with direction
            h = torch.cat([feature, input_views], dim=-1)

            for idx, layer in enumerate(self.views_linears):
                h = self.views_linears[idx](h)
                h = F.relu(h)


            # NOTE: RGB layer
            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], dim=-1)
        else:
            outputs = self.output_linear(h)

        return outputs