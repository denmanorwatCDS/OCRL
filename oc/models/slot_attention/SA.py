import torch
import numpy as np
from torch import nn
from oc.models.utils.networks import CNNEncoder, PositionalEmbedding
from oc.models.utils.slot_attention import SlotAttentionModule
from oc.models.utils.dropouts import CnnPatchDropout, FeatureDropout
from oc.models.slot_attention.submodels import BroadCastDecoder
from oc.models.oc_model import OC_model
from oc.models.utils.losses import hungarian_loss
from itertools import chain

# TODO make dict parameters as mixin into classes, not as method implemented in 100500 models
class Slot_Attention(OC_model):
    def __init__(self, ocr_config: dict, obs_size, obs_channels) -> None:
        super(Slot_Attention, self).__init__(ocr_config, obs_size = obs_size, 
                                             obs_channels = obs_channels)
        self._obs_size = obs_size
        self._obs_channels = obs_channels
        self.num_slots = ocr_config.slotattr.num_slots
        self.rep_dim = ocr_config.slotattr.slot_size

        # build encoder
        # TODO change obs_size so it is a tuple, not a single integer
        self._enc = CNNEncoder(obs_channels, ocr_config.cnn_hsize)
        self._enc_pos = PositionalEmbedding(int(obs_size), ocr_config.cnn_hsize)
        self._patch_dropout = CnnPatchDropout(patch_dropout_proba = ocr_config.cnn_patch_dropout.patch_dropout_proba,
                                              min_patches_dropped = ocr_config.cnn_patch_dropout.min_patches_dropped,
                                              max_patches_dropped = ocr_config.cnn_patch_dropout.max_patches_dropped,
                                              img_size = (obs_size, obs_size), fmap_height = obs_channels, next_kernel_size = self._enc.kernel_size,
                                              device = 'cuda:0') # TODO remove hardcoded device
        self._feature_dropout = FeatureDropout(feature_dropout_proba = ocr_config.feature_dropout.feature_dropout_proba,
                                               min_features_dropped = ocr_config.feature_dropout.min_features_dropped,
                                               max_features_dropped = ocr_config.feature_dropout.max_features_dropped,
                                               fmap_size = self._enc.hidden_size, device = 'cuda:0') # TODO remove hardcoded device
        self._slot_attention = SlotAttentionModule(
            num_iterations = ocr_config.slotattr.num_iterations,
            num_slots = ocr_config.slotattr.num_slots,
            input_channels = ocr_config.cnn_hsize,
            slot_size = ocr_config.slotattr.slot_size,
            mlp_hidden_size = ocr_config.slotattr.mlp_hidden_size,
            num_heads = ocr_config.slotattr.num_slot_heads,
            preinit_type = ocr_config.slotattr.preinit_type,
            normalize_keys = ocr_config.slotattr.normalize_keys
        )
        self._dec = BroadCastDecoder(obs_size = obs_size, obs_channels = obs_channels, 
                                     hidden_size = ocr_config.cnn_hsize, slot_size = ocr_config.slotattr.slot_size,
                                     initial_size = ocr_config.initial_size)
        self.use_hungarian_loss = ocr_config.slotattr.matching_loss.use
        self.hungarian_coef = ocr_config.slotattr.matching_loss.coef

    def _get_slot_params(self):
        return self._slot_attention.named_parameters()
    
    def _get_decoder_params(self):
        return self._dec.named_parameters()

    def _get_enc_params(self):
        return chain(
            self._enc.named_parameters(), self._enc_pos.named_parameters(), 
            self._patch_dropout.named_parameters(), self._feature_dropout.named_parameters())
    
    def get_grouped_parameters(self):
        return {'encoder': self._get_enc_params(),
                'slot': self._get_slot_params(),
                'decoder': self._get_decoder_params()}
    
    def _get_slots(self, obs, do_dropout, training):
        if do_dropout:
            self._patch_dropout.turn_on_dropout(), self._feature_dropout.turn_on_dropout()
        else:
            self._patch_dropout.turn_off_dropout(), self._feature_dropout.turn_off_dropout()
        encoder_output = self._enc_pos(self._feature_dropout(self._enc(self._patch_dropout(obs))))
        encoder_output = torch.permute(encoder_output, dims = [0, 2, 3, 1])
        features = torch.reshape(encoder_output, (encoder_output.shape[0], -1, encoder_output.shape[-1]))

        if training:
            with torch.no_grad():
                _feat = self._enc(obs)
                self._slot_attention.update_statistics(_feat)

        slots, enc_attns = self._slot_attention(features)
        return slots, enc_attns
    
    def get_slots(self, obs, training):
        return self._get_slots(obs, do_dropout = False, training = training)[0]
    
    def get_loss(self, obs, future_obs, do_dropout):
        slots, _ = self._get_slots(obs, do_dropout = do_dropout, training = True)
        recon, _ = self._dec(slots)
        mse = torch.nn.MSELoss(reduction = "mean")
        hung_loss = 0
        SA_loss = mse(obs, recon)
        mets = {'total_loss': SA_loss.detach().cpu(),
                'SA_loss': SA_loss.detach().cpu()}
        if self.use_hungarian_loss:
            hung_loss = hungarian_loss(slots, self._get_slots(future_obs, do_dropout = do_dropout, training = True)[0]) * self.hungarian_coef
            mets.update({'hungarian_loss': hung_loss.detach().cpu()})
        total_loss = SA_loss + hung_loss
        return total_loss, mets
    
    def calculate_validation_data(self, obs):
        with torch.no_grad():
            mse = torch.nn.MSELoss(reduction = "mean")

            slots, enc_attns = self._get_slots(obs, do_dropout = False, training = False)
            enc_attns = enc_attns.transpose(-1, -2)
            
            recon, dec_attns = self._dec(slots)
            
            SA_loss = mse(recon, obs)
            enc_masked_imgs, enc_masks = self.convert_attns_to_masks(obs, enc_attns)
            dec_masked_imgs, dec_masks = self.convert_attns_to_masks(obs, dec_attns)

            drop_slots, drop_enc_attns = self._get_slots(obs, do_dropout = True, training = False)
            drop_enc_attns = drop_enc_attns.transpose(-1, -2)
            
            drop_recon, drop_dec_attns = self._dec(drop_slots)

            drop_SA_loss = mse(drop_recon, obs)
            drop_enc_masked_imgs, drop_enc_masks = self.convert_attns_to_masks(obs, drop_enc_attns)
            drop_dec_masked_imgs, drop_dec_masks = self.convert_attns_to_masks(obs, drop_dec_attns) 

            slot_mean, slot_std = self._slot_attention.log_slot_mean_std()
            feat_mean, feat_std = self._slot_attention.log_feat_mean_std()

        metrics = {
            'total_loss': SA_loss.cpu().numpy(), 'recon_loss': SA_loss.cpu().numpy(), 'drop_recon_loss': drop_SA_loss.cpu().numpy(),
            'slot_mean': slot_mean.cpu().numpy(), 'slot_std': slot_std.cpu().numpy(), 
            'feat_mean': feat_mean.cpu().numpy(), 'feat_std': feat_std.cpu().numpy(),
            'reconstructions': {'no_dropout': self.convert_tensor_to_img(recon),
                                'dropout': self.convert_tensor_to_img(drop_recon)},
            'masked_imgs': {'no_dropout_enc': self.convert_tensor_to_img(enc_masked_imgs), 
                            'no_dropout_dec': self.convert_tensor_to_img(dec_masked_imgs),
                            'dropout_enc': self.convert_tensor_to_img(drop_enc_masked_imgs), 
                            'dropout_dec': self.convert_tensor_to_img(drop_dec_masked_imgs), 
                            },
            'masks' : {'no_dropout_enc': enc_masks.cpu(), 'no_dropout_dec': dec_masks.cpu(),
                       'dropout_enc': drop_enc_masks.cpu(), 'dropout_dec': drop_dec_masks.cpu()}
        }
        return metrics
    
    def update_hidden_states(self, step: int) -> None:
        pass

    def save_oc_extractor(self):
        pass