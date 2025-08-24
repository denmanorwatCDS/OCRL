import torch
from torch import nn
from oc.models.oc_model import OC_model
from torch.nn import functional as F
from itertools import chain
from oc.models.utils.networks import CNNEncoder, PositionalEmbedding, linear
from oc.models.utils.dropouts import CnnPatchDropout, FeatureDropout
from oc.models.utils.slot_attention import SlotAttentionModule
from oc.models.slate.submodels import dVAE, LearnedPositionalEncoding, TransformerDecoder, gumbel_softmax, cosine_anneal

# TODO make dict parameters as mixin into classes, not as method implemented in 100500 models
class SLATE(OC_model):
    def __init__(self, ocr_config: dict, env_config: dict) -> None:
        super(SLATE, self).__init__(ocr_config = ocr_config,
                                    env_config = env_config)
        self._obs_size = obs_size = env_config.obs_size
        self._obs_channels = obs_channels = env_config.obs_channels

        ## Configs
        # dvae config
        self._vocab_size = vocab_size = ocr_config.dvae.vocab_size
        self._d_model = d_model = ocr_config.dvae.d_model
        # SA config
        cnn_hsize = ocr_config.cnn.hidden_size
        num_iterations = ocr_config.slotattr.num_iterations
        slot_size = ocr_config.slotattr.slot_size
        mlp_hidden_size = ocr_config.slotattr.mlp_hidden_size
        pos_channels = ocr_config.slotattr.pos_channels
        num_slot_heads = ocr_config.slotattr.num_slot_heads
        preinit_type = ocr_config.slotattr.preinit_type
        # tfdec config
        num_dec_blocks = ocr_config.tfdec.num_dec_blocks
        num_dec_heads = ocr_config.tfdec.num_dec_heads
        # learning config
        dropout = ocr_config.dropout
        self._tau_start = ocr_config.tau_start
        self._tau_final = ocr_config.tau_final
        self._tau_steps = ocr_config.tau_steps
        self._tau = 1.0
        self.num_slots = num_slots = ocr_config.slotattr.num_slots
        self.rep_dim = slot_size

        self._hard = ocr_config.hard

        # build Discrete VAE
        self._dvae = dVAE(self._vocab_size, obs_channels, img_size = obs_size)
        self._enc_size = enc_size = obs_size // 4

        # build encoder
        self._enc = CNNEncoder(obs_channels, cnn_hsize)
        self._enc_pos = PositionalEmbedding(obs_size, cnn_hsize)
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
            num_iterations,
            num_slots,
            cnn_hsize,
            slot_size,
            mlp_hidden_size,
            num_slot_heads,
            preinit_type = preinit_type,
        )

        self._slotproj = linear(slot_size, d_model, bias=False)

        # build decoder
        self._dict = OneHotDictionary(vocab_size, d_model)
        self._z_pos = LearnedPositionalEncoding(1 + enc_size**2, d_model, dropout)
        self._bos_token = BosToken(d_model)
        self._tfdec = TransformerDecoder(
            num_dec_blocks, enc_size**2, d_model, num_dec_heads, dropout
        )
        # intermediate layer between TF decoder and dVAE decoder
        self._out = linear(d_model, vocab_size, bias=False)

    def get_enc_params(self):
        return chain(self._enc.named_parameters(),
                     self._enc_pos.named_parameters(),
                     self._patch_dropout.named_parameters(),
                     self._feature_dropout.named_parameters())

    def get_slot_params(self):
        return chain(self._slot_attention.named_parameters(),
                     self._slotproj.named_parameters())
    
    def get_dvae_params(self):
        return chain(self._dvae.named_parameters(),
                     self._dict.named_parameters())

    def get_decoder_params(self):
        return chain(
            self._z_pos.named_parameters(),
            self._bos_token.named_parameters(),
            self._tfdec.named_parameters(),
            self._out.named_parameters())
    
    def get_grouped_parameters(self):
        return {'encoder': self.get_enc_params(),
                'slot': self.get_slot_params(),
                'decoder': self.get_decoder_params(),
                'dvae': self.get_dvae_params()}

    def _get_z(self, obs):
        # dvae encode
        z, z_logits = self._dvae(obs, tau = self._tau, hard = self._hard)
        # hard z
        z_hard = gumbel_softmax(z_logits, self._tau, True, dim=1).detach()
        return z, z_hard
    
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
    
    def _calculate_CE(self, obs, slots, z_hard):
        z_hard = z_hard.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)
        z_emb = self._dict(z_hard)
        z_emb = torch.cat(
                [self._bos_token().expand(obs.shape[0], -1, -1), z_emb], dim=1
            )
        z_emb = self._z_pos(z_emb)
        projected_slots = self._slotproj(slots)
        decoder_output = self._tfdec(z_emb[:, :-1], projected_slots)
        pred = self._out(decoder_output)
        cross_entropy = (
            -(z_hard * torch.log_softmax(pred, dim=-1))
                .flatten(start_dim=1)
                .sum(-1)
                .mean())
        return cross_entropy
    
    def get_loss(self, obs, do_dropout):
        # DVAE component of the loss
        z, z_hard = self._get_z(obs)
        dvae_recon = self._dvae.decode(z)
        dvae_mse = ((obs - dvae_recon) ** 2).sum() / obs.shape[0]

        # SLATE component of the loss
        slots, _ = self._get_slots(obs, do_dropout = do_dropout, training = True)
        cross_entropy = self._calculate_CE(obs, slots, z_hard = z_hard)
        
        total_loss = dvae_mse + cross_entropy
        mets = {'total_loss': total_loss.detach().cpu(),
                'dvae_mse': dvae_mse.detach().cpu(),
                'cross_entropy': cross_entropy.detach().cpu()}
        return total_loss, mets
    
    def calculate_validation_data(self, obs):
        with torch.no_grad():
            z, z_hard = self._get_z(obs)
            dvae_recon = self._dvae.decode(z)
            dvae_mse = ((obs - dvae_recon) ** 2).sum() / obs.shape[0]

            # SLATE component of the loss
            drop_slots, drop_enc_attns = self._get_slots(obs, do_dropout = True, training = False)
            drop_enc_attns = drop_enc_attns.transpose(-1, -2)
            drop_cross_entropy = self._calculate_CE(obs, drop_slots, z_hard = z_hard)
            drop_tr_recon = self._gen_imgs(drop_slots)
            drop_enc_masked_imgs, drop_enc_masks = self.convert_attns_to_masks(obs, drop_enc_attns)

            slots, enc_attns = self._get_slots(obs, do_dropout = False, training = False)
            enc_attns = enc_attns.transpose(-1, -2)
            
            cross_entropy = self._calculate_CE(obs, slots, z_hard = z_hard)
            tr_recon = self._gen_imgs(slots)
            enc_masked_imgs, enc_masks = self.convert_attns_to_masks(obs, enc_attns)
            
            slot_mean, slot_std = self._slot_attention.log_slot_mean_std()
            feat_mean, feat_std = self._slot_attention.log_feat_mean_std()
            total_loss = dvae_mse + cross_entropy

        metrics = {
                'total_loss': total_loss.cpu().numpy(), 'DVAE_loss': dvae_mse.cpu().numpy(), 
                'drop_CE_loss': drop_cross_entropy.cpu().numpy(), 'CE_loss': cross_entropy.cpu().numpy(),
                'tau': self._tau,
                'slot_mean': slot_mean.cpu().numpy(), 'slot_std': slot_std.cpu().numpy(), 
                'feat_mean': feat_mean.cpu().numpy(), 'feat_std': feat_std.cpu().numpy(),
                'reconstructions': {'DVAE': self.convert_tensor_to_img(dvae_recon), 
                                    'SLATE': self.convert_tensor_to_img(tr_recon),
                                    'dropout_SLATE': self.convert_tensor_to_img(drop_tr_recon)},
                'masked_imgs': {'no_dropout_enc': self.convert_tensor_to_img(enc_masked_imgs),
                                'dropout_enc': self.convert_tensor_to_img(drop_enc_masked_imgs)},
                'masks': {'no_dropout_enc': enc_masks.cpu(), 'dropout_enc': drop_enc_masks.cpu()}
            }
        return metrics
    
    def _gen_imgs(self, slots):
        with torch.no_grad():
            slots = self._slotproj(slots)
            z_gen = slots.new_zeros(0)
            _input = self._bos_token().expand(slots.shape[0], 1, -1)
            for t in range(self._enc_size**2):
                decoder_output = self._tfdec(self._z_pos(_input), slots)
                z_next = F.one_hot(
                    self._out(decoder_output)[:, -1:].argmax(dim=-1), self._vocab_size
                )
                z_gen = torch.cat((z_gen, z_next), dim=1)
                _input = torch.cat([_input, self._dict(z_next)], dim=1)
            z_gen = (
                z_gen.transpose(1, 2)
                .float()
                .reshape(slots.shape[0], -1, self._enc_size, self._enc_size)
            )
        return self._dvae.decode(z_gen)
    
    def update_hidden_states(self, step: int) -> None:
        # update tau
        self._tau = cosine_anneal(
            step, self._tau_start, self._tau_final, 0, self._tau_steps
        )


class OneHotDictionary(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.dictionary = nn.Embedding(vocab_size, emb_size)

    def forward(self, x):
        """
        x: B, N, vocab_size
        """
        tokens = torch.argmax(x, dim=-1)  # batch_size x N
        token_embs = self.dictionary(tokens)  # batch_size x N x emb_size
        return token_embs


class BosToken(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self._bos_token = nn.Parameter(torch.Tensor(1, 1, d_model))
        nn.init.xavier_uniform_(self._bos_token)

    def forward(self):
        return self._bos_token