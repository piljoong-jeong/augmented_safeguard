""" 
### embedder.Embedder


"""

import torch

class Embedder:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def embed(self, inputs):
        # NOTE: for organization purpose
        return self.__embed_impl(inputs)
    
    def create_embedding_func(self):
        """
        ### Embedder.create_embedding_func

        Generates sinusoidal embedding functions w.r.t given octave levels
        """

        embed_funcs = []

        in_dim = self.kwargs["input_dims"]
        out_dim = 0 # NOTE: value will be accumulated

        if self.kwargs["include_input"]:
            embed_funcs.append(lambda x: x)
            out_dim += in_dim

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]
        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(
                0.0, 
                max_freq, 
                steps=N_freqs
            )
        else: # NOTE: linear sampling (unused)
            pass
            freq_bands = torch.linspace(
                2.0 ** 0, 
                2.0 ** max_freq, 
                steps=N_freqs
            )
        
        # NOTE: generate embedding
        for freq in freq_bands:
            for periodic_func in self.kwargs["periodic_funcs"]: # NOTE: sin & cos
                embed_funcs.append(
                    lambda x, periodic_func=periodic_func, freq=freq: periodic_func(x * freq)
                )
                out_dim += in_dim
        self.embed_funcs = embed_funcs
        self.out_dim = out_dim

        return

    def __embed_impl(self, inputs):
        """
        Returns octave-wise embedded `inputs`
        """
        return torch.cat([
            embed_func(inputs) 
            for embed_func
            in self.embed_funcs
        ], dim=-1)