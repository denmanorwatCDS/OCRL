import torch
import itertools
import functools

from oc.optimizer.decays import constant_lr, cosine_decay_with_warmup, exponential_decay_with_warmup
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
from torch import nn
from statistics import mean
from copy import deepcopy

# TODO fix LR schedulers, as RL scheduler is much slower than OC one. Maybe, throw them away altogether?
def calculate_weight_norm(params):
    norm = []
    for param in params:
            norm.append(param.detach().flatten().clone())
    if norm:
        return torch.sum(torch.cat(norm)**2)**(1/2)
    return 0

# TODO fix scheduler groups. Now there are many of them, due to implementation changes. 
# Earlier there were 3, encoder, slot, decoder
class OCOptimizer():
    def __init__(self, optim_config, oc_model, policy = None):
        self.oc_model = oc_model
        self.policy = policy
        self.optim_config = dict(optim_config)
        self._second_optim_step_counter = 0
        self._build()

    def _get_grouped_named_params(self):
        oc_modules = self.oc_model.get_grouped_parameters()
        params = {**oc_modules}

        if self.policy is not None:
            params.update({'policy': self.policy.named_parameters()})

        return params
    
    def _build(self):
        named_params_by_module = self._get_grouped_named_params()
        
        # There could be many parameters for same optimizer
        modules_paramwise_lrs = self.oc_model.get_paramwise_lr()
        if self.policy is not None:
            modules_paramwise_lrs = {**modules_paramwise_lrs, **self.policy.get_paramwise_lr()}
        self.oc_global_grad_clip = self.optim_config.pop('global_oc_gradient_clip_val', None)
        self.oc_grad_clip_type = self.optim_config.pop('global_oc_grad_clip_type', 2.0)
        modules_optim_kwargs = deepcopy(self.optim_config)
        self.grad_clips = {key: modules_optim_kwargs[key].pop('gradient_clip_val', None) for key in\
                           modules_optim_kwargs.keys()}
        self.grad_types = {key: modules_optim_kwargs[key].pop('gradient_clip_type', 2.0) for key in\
                           modules_optim_kwargs.keys()}
        scheduler_kwargs = {key: modules_optim_kwargs[key].pop('lr_scheduler', None) for key in\
                            modules_optim_kwargs.keys()}
        
        schedule_fns, optimizer_kwargs = {}, []
        for optim_key in modules_optim_kwargs.keys():
            module_name, _ = optim_key.split('_')
            schedule_fns[module_name] = constant_lr

            if modules_paramwise_lrs[module_name] is not None:
                lr = modules_optim_kwargs[optim_key]['lr']
                for name, entry in modules_paramwise_lrs[module_name].items():
                    param, lr_mult = entry['params'], entry['lr_mult']
                    modules_optim_kwargs_without_lr = {key: val for key, val in modules_optim_kwargs[optim_key].items() if key != 'lr'}
                    optimizer_kwargs.append({'name': '.'.join([module_name, name]), 'params': param, 
                                                 'lr': lr * lr_mult,
                                                 **(modules_optim_kwargs_without_lr)})
            else:
                optimizer_kwargs.append({'name': module_name, 'params': [param for _, param in named_params_by_module[module_name]],
                                             **(modules_optim_kwargs[optim_key])})
        
        # Similar to stable-baselines; although, no scheduling.
        self.all_optimizer_kwargs = optimizer_kwargs
        
        self.rl_optimizer = torch.optim.AdamW(self.all_optimizer_kwargs)
        self.oc_optimizer = torch.optim.AdamW(self.all_optimizer_kwargs)

        for name in schedule_fns.keys():
            if scheduler_kwargs[name + '_optimizer'] is not None:
                if scheduler_kwargs[name + '_optimizer']['type'] == 'cosine':
                    schedule_fns[name] = functools.partial(cosine_decay_with_warmup,
                        warmup_steps = scheduler_kwargs[name + '_optimizer']['warmup_steps'], 
                        decay_steps = scheduler_kwargs[name + '_optimizer']['decay_steps'])
                    
                elif scheduler_kwargs[name + '_optimizer']['type'] == 'exp':
                    schedule_fns[name] = functools.partial(exponential_decay_with_warmup,
                        warmup_steps = scheduler_kwargs[name + '_optimizer']['warmup_steps'], 
                        decay_steps = scheduler_kwargs[name + '_optimizer']['decay_steps'],
                        decay_rate = scheduler_kwargs[name + '_optimizer']['decay_rate'])
        
        self.lr_lambdas = []
        # We suppose that optimizers are identical, thus schedulers must be also identical
        for param_group in self.rl_optimizer.param_groups:
            param_group_name = param_group['name'].split('.')[0]
            self.lr_lambdas.append(schedule_fns[param_group_name])

        self.rl_scheduler = LambdaLR(self.rl_optimizer, self.lr_lambdas)
        self.oc_scheduler = LambdaLR(self.oc_optimizer, self.lr_lambdas)
        
    def reset_optimizers(self):
        self.rl_optimizer = torch.optim.AdamW(self.all_optimizer_kwargs)
        self.oc_optimizer = torch.optim.AdamW(self.all_optimizer_kwargs)
        old_rl_sched = self.rl_scheduler.state_dict()
        old_oc_sched = self.oc_scheduler.state_dict()

        self.rl_scheduler = LambdaLR(self.rl_optimizer, self.lr_lambdas)
        self.oc_scheduler = LambdaLR(self.oc_optimizer, self.lr_lambdas)
        self.rl_scheduler.load_state_dict(old_rl_sched)
        self.oc_scheduler.load_state_dict(old_oc_sched)

    def optimizer_zero_grad(self):
        self.rl_optimizer.zero_grad(), self.oc_optimizer.zero_grad()
        
    def optimizer_step(self, optim_name):
        assert optim_name in ['rl', 'oc'], 'We only have rl and oc optimizers'
        
        if optim_name == 'oc':
            self.oc_model.update_hidden_states(self._second_optim_step_counter)
            self._second_optim_step_counter += 1
        
        metrics = {}
        target_optim = self.rl_optimizer if optim_name == 'rl' else self.oc_optimizer
        
        module_named_params = self._get_grouped_named_params()
        if self.oc_global_grad_clip is not None:
            params = self.oc_model.parameters()
            metrics[f'tr_oc_global_grad_norm/{optim_name}_optim'] = clip_grad_norm_(params, self.oc_global_grad_clip, 
                                                                                    norm_type = self.oc_grad_clip_type)
        for key in self.optim_config.keys():
            module_name = key.split('_')[0]
            params = [outp[1] for outp in module_named_params[module_name]]
            if self.grad_clips[key]:
                metrics[f'tr_grad_norm/{module_name}_{optim_name}_optim'] = clip_grad_norm_(
                    params, self.grad_clips[key], norm_type = self.grad_types[key]
                )
            weight_norm = self.get_weight_norm([module_name])
            metrics[f'tr_weight_norm/{module_name}_{optim_name}_optim'] = weight_norm
        self.rl_scheduler.step(), self.oc_scheduler.step(), target_optim.step()
        mean_lr, mean_decay = {}, {}
        for i in range(len(target_optim.param_groups)):
            name, lr, wd = target_optim.param_groups[i]['name'], target_optim.param_groups[i]['lr'],\
                target_optim.param_groups[i]['weight_decay']
            if name not in mean_lr.keys():
                mean_lr[f'hyperparams/{name}_{optim_name}_lr'] = [lr]
                mean_decay[f'hyperparams/{name}_{optim_name}_weight_decay'] = [wd]
            else:
                mean_lr[f'hyperparams/{name}_{optim_name}_lr'].append(lr)
                mean_decay[f'hyperparams/{name}_{optim_name}_weight_decay'].append(wd)
        metrics.update({key: mean(mean_lr[key]) for key in mean_lr.keys()})
        metrics.update({key: mean(mean_decay[key]) for key in mean_decay.keys()})
        return metrics
    
    def get_weight_norm(self, module_names):
        grouped_named_parameters = self._get_grouped_named_params()
        grouped_named_parameters = {module_name: grouped_named_parameters[module_name] for module_name in module_names}
        listified_named_parameters = itertools.chain.from_iterable(\
            [[outp[1] for outp in grouped_named_parameters[key]] for key in grouped_named_parameters.keys()])
        return calculate_weight_norm(listified_named_parameters)

    # TODO adapt this code for new version of OCRL. For current tests, this code is not needed.
    """
    def backward_with_encoder_renorm(self, ocr_loss, critic_loss, actor_loss, entropy_loss,
                                           ocr_modules, critic_modules, actor_modules, entropy_modules):

        _ocr_modules = self.fetch_loss_modules(ocr_modules)
        _critic_modules = self.fetch_loss_modules(critic_modules)
        _actor_modules = self.fetch_loss_modules(actor_modules)
        _entropy_modules = self.fetch_loss_modules(entropy_modules)
        
        ocr_loss.backward(retain_graph = True, inputs = _ocr_modules)
        ocr_enc_grad = torch.linalg.vector_norm(self.get_gradient(['encoder', 'slot']))
        ocr_derivatives = self.get_derivatives(['encoder', 'slot', 'decoder', 'policy'])
        self.optimizer_zero_grad()

        critic_loss.backward(retain_graph = True, inputs = _critic_modules)
        critic_enc_grad = torch.linalg.vector_norm(self.get_gradient(['encoder', 'slot']))
        critic_derivatives = self.get_derivatives(['encoder', 'slot', 'decoder', 'policy'])
        self.optimizer_zero_grad()

        actor_loss.backward(retain_graph = True, inputs = _actor_modules)
        actor_enc_grad = torch.linalg.vector_norm(self.get_gradient(['encoder', 'slot']))
        actor_derivatives = self.get_derivatives(['encoder', 'slot', 'decoder', 'policy'])
        self.optimizer_zero_grad()

        entropy_loss.backward(inputs = _entropy_modules)
        entropy_enc_grad = torch.linalg.vector_norm(self.get_gradient(['encoder', 'slot']))
        entropy_derivatives = self.get_derivatives(['encoder', 'slot', 'decoder', 'policy'])
        self.optimizer_zero_grad()

        rl_enc_grad = critic_enc_grad + actor_enc_grad + entropy_enc_grad
        normalization_required = 0
        if ocr_enc_grad < rl_enc_grad:
            normalization_required = 1
            normalizer_coefficient = ocr_enc_grad / rl_enc_grad
            self.multiply_derivatives_inplace(critic_derivatives, normalizer_coefficient, ['encoder', 'slot'])
            self.multiply_derivatives_inplace(actor_derivatives, normalizer_coefficient, ['encoder', 'slot'])
            self.multiply_derivatives_inplace(entropy_derivatives, normalizer_coefficient, ['encoder', 'slot'])
        self.sum_derivatives_inplace([ocr_derivatives, critic_derivatives, actor_derivatives, entropy_derivatives],
                                     module_names = ['encoder', 'slot', 'decoder', 'policy'])
        return {'ocr_grad_norm_policy_step': ocr_enc_grad, 'critic_grad_norm_policy_step': critic_enc_grad, 
                'actor_grad_norm_policy_step': actor_enc_grad, 'entropy_grad_norm_policy_step': entropy_enc_grad, 
                'normalization_required': normalization_required}
    
    def get_gradient(self, module_names):
        grouped_named_parameters = self._get_grouped_named_params()
        grouped_named_parameters = {module_name: grouped_named_parameters[module_name] for module_name in module_names}
        listified_named_parameters = itertools.chain.from_iterable(\
            [[outp[1] for outp in grouped_named_parameters[key]] for key in grouped_named_parameters.keys()])
        return calculate_grad(listified_named_parameters)
    
    def get_weight_norm(self, module_names):
        grouped_named_parameters = self._get_grouped_named_params()
        grouped_named_parameters = {module_name: grouped_named_parameters[module_name] for module_name in module_names}
        listified_named_parameters = itertools.chain.from_iterable(\
            [[outp[1] for outp in grouped_named_parameters[key]] for key in grouped_named_parameters.keys()])
        return calculate_weight_norm(listified_named_parameters)
    
    def get_derivatives(self, module_names):
        grouped_named_parameters = self._get_grouped_named_params()
        grouped_named_parameters = {module_name: grouped_named_parameters[module_name] for module_name in module_names}
        derivatives = {}
        for key, iter in grouped_named_parameters.items():
            derivatives[key] = {}
            for param_name, param in iter:
                if param.grad is not None:
                    derivatives[key][param_name] = param.grad.detach().clone()
                else:
                    derivatives[key][param_name] = torch.zeros(param.shape).cuda()
        return derivatives

    def sum_derivatives_inplace(self, derivatives, module_names):
        grouped_named_parameters = self._get_grouped_named_params()
        for module_name in module_names:
            module_iter = grouped_named_parameters[module_name]
            for param_name, param in module_iter:
                start = True
                for derivative in derivatives:
                    if start:
                        param.grad = torch.zeros(param.shape).cuda()
                    param.grad += derivative[module_name][param_name]
                    start = False
    
    def multiply_derivatives_inplace(self, derivatives, coefficient, module_list):
        for module_name in module_list:
            for param_name, param in derivatives[module_name].items():
                derivatives[module_name][param_name] = param * coefficient
    
    def fetch_loss_modules(self, module_names):
        all_modules = self._get_grouped_named_params()
        outp = []
        for key in module_names:
            outp.append(all_modules[key])
        outp = itertools.chain.from_iterable(outp)
        outp = [name_tensor[1] for name_tensor in outp if name_tensor[1].requires_grad]
        return outp
    
    def save_oc_extractor(self, path):
        self.features_extractor.save_oc_extractor(path)

    def calculate_validation_statistics(self, obs, masks):
        return self.features_extractor.calculate_validation_statistics(obs, masks)
    
    def get_slot_decoder_attention(self, obs):
        return self.features_extractor.get_slot_decoder_attention(obs)
    """