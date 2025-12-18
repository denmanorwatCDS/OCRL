import torch
from torch import nn
from torch.nn import functional

class HyperNetwork(nn.Module):
    def __init__(self, parameterizer_dim, net_in_dim, net_out_dim,
                       hypernet_arch, compressed_dim, hypernet_act,
                       net_arch, net_act):
        super().__init__()
        hypernetwork_modules, HypernetAct = [], getattr(nn, hypernet_act)
        self.net_act = net_act
        hypernet_arch = hypernet_arch

        hypernetwork_modules.append(nn.Linear(parameterizer_dim, hypernet_arch[0]))
        for i in range(1, len(hypernet_arch)):
            hypernetwork_modules.append(HypernetAct())
            hypernetwork_modules.append(nn.Linear(hypernet_arch[i - 1], hypernet_arch[i]))
        hypernetwork_modules.append(HypernetAct())

        net_arch = [net_in_dim, *net_arch, net_out_dim]
        self.linear_matricies = [((net_arch[i - 1], compressed_dim), (compressed_dim, net_arch[i]))\
                                 for i in range(1, len(net_arch))]
        self.tot_params = 0
        for matrix_pair in self.linear_matricies:
            self.tot_params += (matrix_pair[0][0] * matrix_pair[0][1] + matrix_pair[1][0] * matrix_pair[1][1])
        hypernetwork_modules.append(nn.Linear(hypernet_arch[-1], self.tot_params))
        self.hypernetwork = nn.Sequential(*hypernetwork_modules)

    def forward(self, obs, parameterizer):
        params_for_task = self.hypernetwork(parameterizer)
        act = getattr(nn, self.net_act)()
        outp, latest_used_param = obs.reshape(-1, 1, obs.shape[-1]), 0
        for first_mat_dims, second_mat_dims in self.linear_matricies:
            first_param_qty = first_mat_dims[0] * first_mat_dims[1]
            second_param_qty = second_mat_dims[0] * second_mat_dims[1]
            first_mat = params_for_task[:, latest_used_param: latest_used_param + first_param_qty].reshape(
                                       -1, *first_mat_dims)
            latest_used_param += first_param_qty
            second_mat = params_for_task[:, latest_used_param: latest_used_param + second_param_qty].reshape(
                                         -1, *second_mat_dims)
            latest_used_param += second_param_qty
            outp = outp @ first_mat @ second_mat
            if latest_used_param != self.tot_params:
                outp = act(outp)
        return torch.squeeze(outp)