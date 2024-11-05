import torch
import unittest
from model import MultimodalCaptionGenerator

class VisionConfig:

    def __init__(self):
        self.patch_sz = 16
        self.embed_dim = 768
        self.img_sz = 224
        self.inter_size = 3072
        self.num_heads = 12
        self.num_encoder_layers = 12
        self.layer_norm_eps = 1e-6
        self.in_channels = 3

class MultimodalConfig:

    def __init__(self):
        self.eps = 1e-6
        self.hidden_size = 2048
        self.seq_len = 392
        self.inter_dim = 4096
        self.num_heads = 8
        self.vocab_size = 50259
        self.padding_idx = 50258
        self.num_decoder_layers = 8
        self.emb_dim = 768
        self.proj_dim = 2048
        self.num_image_tokens = 196
        self.num_text_tokens = 196



class TestMultimodalCaptionGenerator():
    
    def test_forward_shapes(self, encoder_config, decoder_config):

        model = MultimodalCaptionGenerator(encoder_config, decoder_config)
        batch_size = 2
        seq_len = decoder_config.seq_len//2
        img = torch.randn(batch_size, 3, 224, 224)
        text = torch.randint(0, 1000, (batch_size, seq_len))
        
        output, attention_scores = model(text, img)

        
        self.assertEqual(output.shape, (batch_size, seq_len, decoder_config.vocab_size))

encoder_config = VisionConfig()
decoder_config = MultimodalConfig()

unit_test = TestMultimodalCaptionGenerator()
unit_test.test_forward_shapes(encoder_config, decoder_config)
