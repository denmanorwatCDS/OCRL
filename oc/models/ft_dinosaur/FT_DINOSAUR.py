import torch
from torch import nn
from oc.models.oc_model import OC_model
from oc.models.ft_dinosaur.submodels import FrameEncoder, MLPDecoder, Resizer
from oc.models.utils.slot_attention import SlotAttentionModule
from copy import deepcopy


class FT_DINOSAUR(OC_model):
    def __init__(self, ocr_config: dict, env_config: dict) -> None:
        super(FT_DINOSAUR, self).__init__(ocr_config, env_config)
        self._obs_size = obs_size = env_config.obs_size
        self._obs_channels = env_config.obs_channels
        self.unfrozen_blocks, self.layerwise_decay = ocr_config.encattr.backbone.unfrozen_blocks, ocr_config.encattr.backbone.layerwise_decay

        ## Configs

        self._enc_size = obs_size // 4

        self.num_slots = ocr_config.slotattr.num_slots
        self.rep_dim = ocr_config.slotattr.slot_size

        self.reg_coef = ocr_config.loss.reg_coef
        # build encoder
        self._enc = FrameEncoder(
            backbone_kwargs = ocr_config.encattr.backbone,
            output_transform_kwargs = ocr_config.encattr.output_transform,
            obs_size = (env_config.obs_size, env_config.obs_size)
        )
        self._slot_attention = SlotAttentionModule(
            num_iterations = ocr_config.slotattr.num_iterations,
            num_slots = ocr_config.slotattr.num_slots,
            input_channels = ocr_config.encattr.output_transform.outp_dim if ocr_config.encattr.output_transform is not None else 384,
            slot_size = ocr_config.slotattr.slot_size,
            mlp_hidden_size = ocr_config.slotattr.mlp_hidden_size,
            num_heads = ocr_config.slotattr.num_slot_heads,
            preinit_type = ocr_config.slotattr.preinit_type,
        )
        self._target_enc = FrameEncoder(
            backbone_kwargs = ocr_config.targetencattr.backbone,
            output_transform_kwargs = None,
            obs_size = (env_config.obs_size, env_config.obs_size)
        )
        with torch.no_grad():
            test_tensor = torch.zeros((1, env_config.obs_channels, env_config.obs_size, env_config.obs_size))
            inp_shape = self._target_enc(test_tensor, patch_dropout = False)['features'].shape[1]

        self._dec = MLPDecoder(
            inp_dim = ocr_config.decoder.inp_dim,
            outp_dim = ocr_config.decoder.outp_dim,
            hidden_dims = ocr_config.decoder.hidden_dims,
            n_patches = inp_shape,
            dropout_prob = ocr_config.decoder.dropout_prob,
            spectral_normalisation = ocr_config.decoder.spectral_normalisation
        )
        self._resizer = Resizer(size = (env_config.obs_size, env_config.obs_size), patch_inputs=True)

        self._paramwise_lr = {}
        for name, _ in self.get_grouped_parameters().items():
            self._paramwise_lr[name] = None
        self._prepare_enc()
        self._prepare_enc_paramwise_lr()

    def get_enc_params(self):
        return self._enc.named_parameters()

    def get_slot_params(self):
        return self._slot_attention.named_parameters()
    
    def get_decoder_params(self):
        return self._dec.named_parameters()
    
    def _prepare_enc(self):
        assert self.unfrozen_blocks is None or (self.unfrozen_blocks > 0), 'Unfrozen blocks must be non-negative.'
        blocks_left, prev_layer_name, prev_block = self.unfrozen_blocks, None, -1
        for name, param in reversed(list(self._enc.named_parameters())):
            if 'output_transform' in name:
                continue

            if prev_layer_name is None:
                prev_layer_name = '.'.join(name.split('.')[:-1])

            if ('.blocks.' in name) and (self.unfrozen_blocks is not None):
                backbone, model, blocks, block_idx, *layer_name = name.split('.')
                block_idx = int(block_idx)
                if prev_block == -1:
                    prev_block = block_idx

                if prev_block != block_idx:
                    prev_block, blocks_left = block_idx, blocks_left - 1
                
                if blocks_left > 0:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def _prepare_enc_paramwise_lr(self):
        prev_layer_name, lr_mult_dict, current_lr_mult = None, {}, 1
        for name, param in reversed(list(self._enc.named_parameters())):
            if not param.requires_grad:
                continue

            elif 'output_transform' in name:
                lr_mult_dict[name] = {'name': name, 'param': param, 'lr_mult': 1}
                continue

            if prev_layer_name is None:
                prev_layer_name = '.'.join(name.split('.')[:-1])

            layer_name = '.'.join(name.split('.')[:-1])
            if prev_layer_name != layer_name:
                current_lr_mult *= self.layerwise_decay
            prev_layer_name = layer_name
            lr_mult_dict[name] = {'name': name, 'params': param, 'lr_mult': current_lr_mult}

        self._paramwise_lr['encoder'] = lr_mult_dict

    def training_mode(self):
        super().training_mode()
        self._prepare_enc()
    
    def get_grouped_parameters(self):
        return {'encoder': self.get_enc_params(),
                'slot': self.get_slot_params(),
                'decoder': self.get_decoder_params()}
    
    def get_paramwise_lr(self):
        return deepcopy(self._paramwise_lr)
    
    def _get_slots(self, obs, do_dropout, training):
        encoder_output = self._enc(obs, patch_dropout = do_dropout)
        features = encoder_output['features']
        
        if training:
            with torch.no_grad():
                _feat = self._enc(obs, patch_dropout = do_dropout)['features']
                self._slot_attention.update_statistics(_feat)

        slots, attns = self._slot_attention(features)
        return slots, attns
    
    def get_loss(self, obs, do_dropout):
        mse = torch.nn.MSELoss(reduction = "mean")

        slots, slot_attns = self._get_slots(obs, do_dropout = do_dropout, training = True)
        dec_out = self._dec(slots)
        decoder_output, decoder_attns = dec_out['reconstruction'], dec_out['masks']
        with torch.no_grad():
            target_encoder_output = self._target_enc(obs, patch_dropout = False)['features']
        assert decoder_output.shape == target_encoder_output.shape,\
            'Decoder shape and target shape must coincide!'
        dino_loss = mse(decoder_output, target_encoder_output)
        return dino_loss
    
    def calculate_validation_data(self, obs):
        mse = torch.nn.MSELoss(reduction = "mean")

        with torch.no_grad():
            target_encoder_output = self._target_enc(obs, patch_dropout = False)['features']

            slots, enc_attns = self._get_slots(obs, do_dropout = False, training = False)
            enc_attns = enc_attns.transpose(-1, -2)
            out = self._dec(slots)
            dec_output, dec_attns = out['reconstruction'], out['masks']
            dino_loss = mse(dec_output, target_encoder_output)
            
            drop_slots, _ = self._get_slots(obs, do_dropout = True, training = False)
            out = self._dec(drop_slots)
            drop_dec_output, drop_dec_attns = out['reconstruction'], out['masks']
            drop_dino_loss = mse(drop_dec_output, target_encoder_output)

            enc_attns, dec_attns, drop_dec_attns =\
                self._resizer(enc_attns), self._resizer(dec_attns),\
                self._resizer(drop_dec_attns)
            
            # Hack to draw correct masks.
            enc_masked_imgs, enc_masks = self.convert_attns_to_masks(obs, enc_attns)
            dec_masked_imgs, dec_masks = self.convert_attns_to_masks(obs, dec_attns)
            drop_dec_masked_imgs, drop_dec_masks = self.convert_attns_to_masks(obs, drop_dec_attns)

            slot_mean, slot_std = self._slot_attention.log_slot_mean_std()
            feat_mean, feat_std = self._slot_attention.log_feat_mean_std()

        metrics = {
                'total_loss': dino_loss.cpu().numpy(), 'drop_dino_loss': drop_dino_loss.cpu().numpy(), 'dino_loss': dino_loss.cpu().numpy(),
                'slot_mean': slot_mean.cpu().numpy(), 'slot_std': slot_std.cpu().numpy(), 
                'feat_mean': feat_mean.cpu().numpy(), 'feat_std': feat_std.cpu().numpy(),
                'reconstructions': {},
                'masks': {'no_dropout_enc': enc_masks.cpu(), 'no_dropout_dec': dec_masks.cpu(),
                          'dropout_dec': drop_dec_masks.cpu()},
                'masked_imgs': {'no_dropout_enc': self.convert_tensor_to_img(enc_masked_imgs),
                                'no_dropout_dec': self.convert_tensor_to_img(dec_masked_imgs),
                                'dropout_dec': self.convert_tensor_to_img(drop_dec_masked_imgs)}
            }
        return metrics
    
    def update_hidden_states(self, step: int) -> None:
        pass
