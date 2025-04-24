import abc
import torch
import itertools

from torch import nn
from torch.distributions import Normal
from torch.distributions.independent import Independent
from utils.distributions.tanh import TanhNormal

from networks.mlp import MLPModule, MultiHeadedMLPModule

class GaussianMLPBaseModule(nn.Module):
    """Base of GaussianMLPModel.

    Args:
        input_dim (int): Input dimension of the model.
        output_dim (int): Output dimension of the model.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        std_hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for std. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        std_hidden_nonlinearity (callable): Nonlinearity for each hidden layer
            in the std network.
        std_hidden_w_init (callable):  Initializer function for the weight
            of hidden layer (s).
        std_hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s).
        std_output_nonlinearity (callable): Activation function for output
            dense layer in the std network. It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
        std_output_w_init (callable): Initializer function for the weight
            of output dense layer(s) in the std network.
        std_parameterization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation.
            - softplus: the std will be computed as log(1+exp(x)).
        layer_normalization (bool): Bool for using layer normalization or not.
        normal_distribution_cls (torch.distribution): normal distribution class
            to be constructed and returned by a call to forward. By default, is
            `torch.distributions.Normal`.

    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_std=True,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_hidden_sizes=(32, 32),
                 std_hidden_nonlinearity=torch.tanh,
                 std_hidden_w_init=nn.init.xavier_uniform_,
                 std_hidden_b_init=nn.init.zeros_,
                 std_output_nonlinearity=None,
                 std_output_w_init=nn.init.xavier_uniform_,
                 std_parameterization='exp',
                 layer_normalization=False,
                 normal_distribution_cls=Normal):
        super().__init__()

        self._input_dim = input_dim
        self._hidden_sizes = hidden_sizes
        self._action_dim = output_dim
        self._learn_std = learn_std
        self._std_hidden_sizes = std_hidden_sizes
        self._min_std = min_std
        self._max_std = max_std
        self._std_hidden_nonlinearity = std_hidden_nonlinearity
        self._std_hidden_w_init = std_hidden_w_init
        self._std_hidden_b_init = std_hidden_b_init
        self._std_output_nonlinearity = std_output_nonlinearity
        self._std_output_w_init = std_output_w_init
        self._std_parameterization = std_parameterization
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._layer_normalization = layer_normalization
        self._norm_dist_class = normal_distribution_cls

        if self._std_parameterization not in ('exp', 'softplus', 'sigmoid'):
            raise NotImplementedError

        init_std_param = torch.Tensor([init_std]).log()
        if self._learn_std:
            self._init_std = torch.nn.Parameter(init_std_param)
        else:
            self._init_std = init_std_param
            self.register_buffer('init_std', self._init_std)

        self._min_std_param = self._max_std_param = None
        if min_std is not None:
            if self._std_parameterization in ['sigmoid', 'softplus', 'exp']:
                self._min_std_param = torch.Tensor([min_std])
            self.register_buffer('min_std_param', self._min_std_param)
        if max_std is not None:
            if self._std_parameterization in ['sigmoid', 'softplus', 'exp']:
                self._max_std_param = torch.Tensor([max_std])
            self.register_buffer('max_std_param', self._max_std_param)

    def to(self, *args, **kwargs):
        """Move the module to the specified device.

        Args:
            *args: args to pytorch to function.
            **kwargs: keyword args to pytorch to function.

        """
        ret = super().to(*args, **kwargs)
        buffers = dict(self.named_buffers())
        if not isinstance(self._init_std, torch.nn.Parameter):
            self._init_std = buffers['init_std']
        self._min_std_param = buffers.get('min_std_param', None)
        self._max_std_param = buffers.get('max_std_param', None)
        return ret

    # Parent module's .to(), .cpu(), and .cuda() call children's ._apply().
    def _apply(self, *args, **kwargs):
        ret = super()._apply(*args, **kwargs)
        buffers = dict(self.named_buffers())
        if not isinstance(self._init_std, torch.nn.Parameter):
            self._init_std = buffers['init_std']
        self._min_std_param = buffers.get('min_std_param', None)
        self._max_std_param = buffers.get('max_std_param', None)
        return ret

    @abc.abstractmethod
    def _get_mean_and_log_std(self, *inputs):
        pass

    def forward(self, inp):
        """Forward method.

        Args:
            *inputs: Input to the module.

        Returns:
            torch.distributions.independent.Independent: Independent
                distribution.

        """
        mean, log_std_uncentered = self._get_mean_and_log_std(inp)

        if self._std_parameterization == 'exp':
            std = log_std_uncentered.exp()
        elif self._std_parameterization == 'softplus':
            std = log_std_uncentered.exp().add(1.).log()
        elif self._std_parameterization == 'sigmoid':
            std = torch.sigmoid(log_std_uncentered) * (self._max_std_param - self._min_std_param) + self._min_std_param
        else:
            assert False

        if (self._min_std_param or self._max_std_param) and (self._std_parameterization != 'sigmoid'):
            std = std.clamp(
                    min=(None if self._min_std_param is None else
                         self._min_std_param.item()),
                    max=(None if self._max_std_param is None else
                         self._max_std_param.item()))

        dist = self._norm_dist_class(mean, std)
        # This control flow is needed because if a TanhNormal distribution is
        # wrapped by torch.distributions.Independent, then custom functions
        # such as rsample_with_pretanh_value of the TanhNormal distribution
        # are not accessable.
        if not isinstance(dist, TanhNormal):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist.batch_shape samples.
            dist = Independent(dist, 1)

        return dist
    
    def forward_mode(self, inp):
        dist = self.forward(inp)
        return dist.mean

class GaussianMLPIndependentStdModule(GaussianMLPBaseModule):
    """GaussianMLPModule which has two different mean and std network.

    Args:
        input_dim (int): Input dimension of the model.
        output_dim (int): Output dimension of the model.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        std_hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for std. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        std_hidden_nonlinearity (callable): Nonlinearity for each hidden layer
            in the std network.
        std_hidden_w_init (callable):  Initializer function for the weight
            of hidden layer (s).
        std_hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s).
        std_output_nonlinearity (callable): Activation function for output
            dense layer in the std network. It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
        std_output_w_init (callable): Initializer function for the weight
            of output dense layer(s) in the std network.
        std_parameterization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        layer_normalization (bool): Bool for using layer normalization or not.
        normal_distribution_cls (torch.distribution): normal distribution class
            to be constructed and returned by a call to forward. By default, is
            `torch.distributions.Normal`.

    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_std=True,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_hidden_sizes=(32, 32),
                 std_hidden_nonlinearity=torch.tanh,
                 std_hidden_w_init=nn.init.xavier_uniform_,
                 std_hidden_b_init=nn.init.zeros_,
                 std_output_nonlinearity=None,
                 std_output_w_init=nn.init.xavier_uniform_,
                 std_parameterization='exp',
                 layer_normalization=False,
                 normal_distribution_cls=Normal,
                 **kwargs):
        super(GaussianMLPIndependentStdModule,
              self).__init__(input_dim=input_dim,
                             output_dim=output_dim,
                             hidden_sizes=hidden_sizes,
                             hidden_nonlinearity=hidden_nonlinearity,
                             hidden_w_init=hidden_w_init,
                             hidden_b_init=hidden_b_init,
                             output_nonlinearity=output_nonlinearity,
                             output_w_init=output_w_init,
                             output_b_init=output_b_init,
                             learn_std=learn_std,
                             init_std=init_std,
                             min_std=min_std,
                             max_std=max_std,
                             std_hidden_sizes=std_hidden_sizes,
                             std_hidden_nonlinearity=std_hidden_nonlinearity,
                             std_hidden_w_init=std_hidden_w_init,
                             std_hidden_b_init=std_hidden_b_init,
                             std_output_nonlinearity=std_output_nonlinearity,
                             std_output_w_init=std_output_w_init,
                             std_parameterization=std_parameterization,
                             layer_normalization=layer_normalization,
                             normal_distribution_cls=normal_distribution_cls)

        self._mean_module = MLPModule(
            input_dim=self._input_dim,
            output_dim=self._action_dim,
            hidden_sizes=self._hidden_sizes,
            hidden_nonlinearity=self._hidden_nonlinearity,
            hidden_w_init=self._hidden_w_init,
            hidden_b_init=self._hidden_b_init,
            output_nonlinearity=self._output_nonlinearity,
            output_w_init=self._output_w_init,
            output_b_init=self._output_b_init,
            layer_normalization=self._layer_normalization,
            **kwargs)

        self._log_std_module = MLPModule(
            input_dim=self._input_dim,
            output_dim=self._action_dim,
            hidden_sizes=self._std_hidden_sizes,
            hidden_nonlinearity=self._std_hidden_nonlinearity,
            hidden_w_init=self._std_hidden_w_init,
            hidden_b_init=self._std_hidden_b_init,
            output_nonlinearity=self._std_output_nonlinearity,
            output_w_init=self._std_output_w_init,
            output_b_init=self._init_std_b,
            layer_normalization=self._layer_normalization,
            **kwargs)

    def _init_std_b(self, b):
        """Default bias initialization function.

        Args:
            b (torch.Tensor): The bias tensor.

        Returns:
            torch.Tensor: The bias tensor itself.

        """
        if self._std_parameterization not in ['softplus']:
            return nn.init.constant_(b, self._init_std.item())
        else:
            return nn.init.constant_(b, self._init_std.exp().add(-1.0).log().item())

    def _get_mean_and_log_std(self, inp):
        """Get mean and std of Gaussian distribution given inputs.

        Args:
            *inputs: Input to the module.

        Returns:
            torch.Tensor: The mean of Gaussian distribution.
            torch.Tensor: The variance of Gaussian distribution.

        """
        return self._mean_module(inp), self._log_std_module(inp)
    
class GaussianMLPTwoHeadedModule(GaussianMLPBaseModule):
    """GaussianMLPModule which has only one mean network.

    Args:
        input_dim (int): Input dimension of the model.
        output_dim (int): Output dimension of the model.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        std_parameterization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        layer_normalization (bool): Bool for using layer normalization or not.
        normal_distribution_cls (torch.distribution): normal distribution class
            to be constructed and returned by a call to forward. By default, is
            `torch.distributions.Normal`.

    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_std=True,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_parameterization='exp',
                 layer_normalization=False,
                 normal_distribution_cls=Normal):
        super(GaussianMLPTwoHeadedModule,
              self).__init__(input_dim=input_dim,
                             output_dim=output_dim,
                             hidden_sizes=hidden_sizes,
                             hidden_nonlinearity=hidden_nonlinearity,
                             hidden_w_init=hidden_w_init,
                             hidden_b_init=hidden_b_init,
                             output_nonlinearity=output_nonlinearity,
                             output_w_init=output_w_init,
                             output_b_init=output_b_init,
                             learn_std=learn_std,
                             init_std=init_std,
                             min_std=min_std,
                             max_std=max_std,
                             std_parameterization=std_parameterization,
                             layer_normalization=layer_normalization,
                             normal_distribution_cls=normal_distribution_cls)

        self._shared_mean_log_std_network = MultiHeadedMLPModule(
            n_heads=2,
            input_dim=self._input_dim,
            output_dims=self._action_dim,
            hidden_sizes=self._hidden_sizes,
            hidden_nonlinearity=self._hidden_nonlinearity,
            hidden_w_init=self._hidden_w_init,
            hidden_b_init=self._hidden_b_init,
            output_nonlinearities=self._output_nonlinearity,
            output_w_inits=self._output_w_init,
            output_b_inits=[
                nn.init.zeros_,
                (lambda x: nn.init.constant_(x, self._init_std.item())
                 if self._std_parameterization not in ['softplus_real']
                 else lambda x: nn.init.constant_(x, self._init_std.exp().exp().add(-1.0).log().item())),
            ],
            layer_normalization=self._layer_normalization)

    def _get_mean_and_log_std(self, inp):
        """Get mean and std of Gaussian distribution given inputs.

        Args:
            *inputs: Input to the module.

        Returns:
            torch.Tensor: The mean of Gaussian distribution.
            torch.Tensor: The variance of Gaussian distribution.

        """
        return self._shared_mean_log_std_network(inp)
    
    def get_parameters_without_std(self):
        params = []
        for name, param in self.named_parameters():
            if not ("_shared_mean_log_std_network" in name):
                params.append(param)
        return iter(params)
    
    def get_std_parameters(self):
        return iter([self._shared_mean_log_std_network._output_layers[1].linear.weight,
                     self._shared_mean_log_std_network._output_layers[1].linear.bias])
    
class GaussianMLPGlobalStdModule(GaussianMLPBaseModule):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_std=True,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_parameterization='exp',
                 layer_normalization=False,
                 normal_distribution_cls=Normal):
        super(GaussianMLPGlobalStdModule,
              self).__init__(input_dim=input_dim,
                             output_dim=output_dim,
                             hidden_sizes=hidden_sizes,
                             hidden_nonlinearity=hidden_nonlinearity,
                             hidden_w_init=hidden_w_init,
                             hidden_b_init=hidden_b_init,
                             output_nonlinearity=output_nonlinearity,
                             output_w_init=output_w_init,
                             output_b_init=output_b_init,
                             learn_std=learn_std,
                             init_std=init_std,
                             min_std=min_std,
                             max_std=max_std,
                             std_parameterization=std_parameterization,
                             layer_normalization=layer_normalization,
                             normal_distribution_cls=normal_distribution_cls)
        
        self.mean_network = MLPModule(input_dim=self._input_dim,
            output_dim=self._action_dim,
            hidden_sizes=self._hidden_sizes,
            hidden_nonlinearity=self._hidden_nonlinearity,
            hidden_w_init=self._hidden_w_init,
            hidden_b_init=self._hidden_b_init,
            output_nonlinearity=self._output_nonlinearity,
            output_w_init=self._output_w_init,
            output_b_init=nn.init.zeros_,
            layer_normalization=self._layer_normalization)
        
        assert len(self._action_dim.shape) == 0
        self._log_std_parameter = torch.nn.Parameter(torch.Tensor([init_std for i in range(self._action_dim)]).log())

    def _get_mean_and_log_std(self, inp):
        """Get mean and std of Gaussian distribution given inputs.

        Args:
            *inputs: Input to the module.

        Returns:
            torch.Tensor: The mean of Gaussian distribution.
            torch.Tensor: The variance of Gaussian distribution.

        """
        return [self.mean_network(inp), self._log_std_parameter]
    
    def get_parameters_without_std(self):
        params = []
        for name, param in self.named_parameters():
            if not ('_log_std_parameter' in name):
                params.append(param)
        return iter(params)
    
    def get_std_parameters(self):
        return iter([self._log_std_parameter])
    
class GaussianMLPIndependendInputStdModule(GaussianMLPBaseModule):
    def __init__(self,
                 input_idx_mean,
                 input_idx_std,
                 output_dim,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_std=True,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_parameterization='exp',
                 layer_normalization=False,
                 normal_distribution_cls=Normal):
        super(GaussianMLPIndependendInputStdModule,
              self).__init__(input_dim=None,
                             output_dim=output_dim,
                             hidden_sizes=hidden_sizes,
                             hidden_nonlinearity=hidden_nonlinearity,
                             hidden_w_init=hidden_w_init,
                             hidden_b_init=hidden_b_init,
                             output_nonlinearity=output_nonlinearity,
                             output_w_init=output_w_init,
                             output_b_init=output_b_init,
                             learn_std=learn_std,
                             init_std=init_std,
                             min_std=min_std,
                             max_std=max_std,
                             std_parameterization=std_parameterization,
                             layer_normalization=layer_normalization,
                             normal_distribution_cls=normal_distribution_cls)
        
        self.input_idx_mean = input_idx_mean
        self.input_idx_std = input_idx_std
        
        self.mean_network = MLPModule(input_dim = input_idx_mean[1] - input_idx_mean[0],
            output_dim=self._action_dim,
            hidden_sizes=self._hidden_sizes,
            hidden_nonlinearity=self._hidden_nonlinearity,
            hidden_w_init=self._hidden_w_init,
            hidden_b_init=self._hidden_b_init,
            output_nonlinearity=self._output_nonlinearity,
            output_w_init=self._output_w_init,
            output_b_init=nn.init.zeros_,
            layer_normalization=self._layer_normalization)
        
        self._logstd_network = MLPModule(input_dim = input_idx_std[1] - input_idx_std[0],
            output_dim=self._action_dim,
            hidden_sizes=self._hidden_sizes,
            hidden_nonlinearity=self._hidden_nonlinearity,
            hidden_w_init=self._hidden_w_init,
            hidden_b_init=self._hidden_b_init,
            output_nonlinearity=self._output_nonlinearity,
            output_w_init=self._output_w_init,
            output_b_init= lambda x: nn.init.constant_(x, self._init_std.item()),
            layer_normalization=self._layer_normalization)
        
        assert len(self._action_dim.shape) == 0

    def _get_mean_and_log_std(self, inp):
        """Get mean and std of Gaussian distribution given inputs.

        Args:
            *inputs: Input to the module.

        Returns:
            torch.Tensor: The mean of Gaussian distribution.
            torch.Tensor: The variance of Gaussian distribution.

        """
        return [self.mean_network(inp[..., self.input_idx_mean[0]: self.input_idx_mean[1]]), 
                self._logstd_network(inp[..., self.input_idx_std[0]: self.input_idx_std[1]])]
    
    def get_parameters_without_std(self):
        params = []
        for name, param in self.named_parameters():
            if not ('_logstd_network' in name):
                params.append(param)
        return iter(params)
    
    def get_std_parameters(self):
        return self._logstd_network.parameters()