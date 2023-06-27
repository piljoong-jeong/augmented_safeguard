import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import NeRF.embedder as embedder

def create_NeRF(args):

    embed_fn, input_ch = embedder.get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = embedder.get_embedder(args.multires_views, args.i_embed)

    output_ch = 5 if args.N_importance > 0 else 4

    skips = args.skips

    # TODO: parameterize
    device="cuda"

    model = NeRF(
        D=args.netdepth, 
        W=args.netwidth, 
        input_ch=input_ch,
        output_ch=output_ch, 
        skips=skips,
        input_ch_views=input_ch_views,
        use_viewdirs=args.use_viewdirs,
    ).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(
            D=args.netdepth_fine, 
            W=args.netwidth_fine, 
            input_ch=input_ch, 
            output_ch=output_ch, 
            skips=skips, 
            input_ch_views=input_ch_views, 
            use_viewdirs=args.use_viewdirs,
        ).to(device)
        grad_vars += list(model_fine.parameters())

    # NOTE: wrapper for `run_network(...)`; as embedding functions won't change
    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(
        inputs, viewdirs, 
        network_fn, 
        embed_fn=embed_fn, embeddirs_fn=embeddirs_fn, 
        netchunk=args.netchunk
    )

    
    # NOTE: create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname


    #######################

    # NOTE: load checkpoints
    if args.ft_path is not None and args.ft_path != "None":
        ckpts = [args.ft_path]
    else:
        ckpts = [
            os.path.join(basedir, expname, f)
            for f
            in sorted(
                os.listdir(os.path.join(basedir, expname))
            )
            if "tar" in f
        ]

    print(f"[DEBUG] Found ckpts: {ckpts}")

    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1] # NOTE: the latest checkpoint
        print(f"[DEBUG] Reloading from {ckpt_path}")
        ckpt = torch.load(ckpt_path)
    
        start = ckpt["global_step"] # NOTE: update starting step
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # NOTE: load model
        model.load_state_dict(ckpt["network_fn_state_dict"])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt["network_fine_state_dict"])

    #######################

    render_kwargs_train = {
        "network_query_fn": network_query_fn, 
        "perturb": args.perturb, 
        "N_importance": args.N_importance, # NOTE: amount of sampling positions from PDF generated from coarse NeRF, generally 128
        "network_fine": model_fine, 
        "N_samples": args.N_samples, # NOTE: amount of sampling positions uniformly in order to estimate PDF in coarse NeRF, generally 64
        "network_fn": model, 
        "use_viewdirs": args.use_viewdirs, 
        "white_bkgd": args.white_bkgd, 
        "raw_noise_std": args.raw_noise_std, # NOTE: prevent overfitting when training
    }

    # NOTE: NDC is effective only for LLFF-style forward facing data
    if args.dataset_type != "llff" or args.no_ndc:
        print(f"[DEBUG] No NDC!")

        render_kwargs_train["ndc"] = False
        render_kwargs_train["lindisp"] = args.lindisp

    render_kwargs_test = {
        k: render_kwargs_train[k]
        for k in render_kwargs_train
    }
    render_kwargs_test["perturb"] = False
    render_kwargs_test["raw_noise_std"] = 0.

    # NOTE: try patch from https://github.com/pytorch/pytorch/issues/80809#issuecomment-1173481031
    optimizer.param_groups[0]["capturable"] = True

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


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