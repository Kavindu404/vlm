class VisionConfig:

    def __init__(self):
        self.patch_sz = 16
        self.embed_dim = 512
        self.img_sz = 224
        self.inter_size = 768
        self.num_heads = 4
        self.num_encoder_layers = 6
        self.layer_norm_eps = 1e-6
        self.in_channels = 3
        self.dropout_prob = 0.3
        self.attention_dropout = 0.3

class MultimodalConfig:

    def __init__(self):

        self.project_name = "base_vlm_self_attn_sinPos_val_v33"
        self.eps = 1e-6
        self.hidden_size = 768
        self.seq_len = 392
        self.max_len = 196
        self.inter_dim = 1024
        self.num_heads = 4
        self.vocab_size = 50259
        self.padding_idx = 50258
        self.sos_token_id = 50257
        self.eos_token_id = 50256
        self.num_decoder_layers = 8
        self.emb_dim = 512
        self.proj_dim = self.hidden_size
        self.num_image_tokens = 196
        self.num_text_tokens = 196

        self.learning_rate = 1e-3
        self.weight_decay = 0.1
        self.warmup_ratio = 0.1
        self.num_epochs = 2000
        self.dropout_prob = 0.3
        self.attention_dropout = 0.3

        self.fixed_batch_idx = 1
        self.fixed_sample_idx = 25
        
        self.train_image_dir = "/unity/g1/kgalla/datasets/train2014/"
        self.train_h5_path = "/unity/g1/kgalla/datasets/annotations/train/coco_train2014_captions.h5"
        self.batch_size = 128
        self.num_workers = 96

        self.val_image_dir = "/unity/g1/kgalla/datasets/val2014/"
        self.val_h5_path = "/unity/g1/kgalla/datasets/annotations/val/coco_val2014_captions.h5"

        self.gpu_ids = [0, 1, 2, 3]
        self.cuda_visible_devices = "0,1,2,3"