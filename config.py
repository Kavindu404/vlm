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
        self.max_len = 196
        self.inter_dim = 4096
        self.num_heads = 8
        self.vocab_size = 50259
        self.padding_idx = 50258
        self.sos_token_id = 50257
        self.eos_token_id = 50256
        self.num_decoder_layers = 8
        self.emb_dim = 768
        self.proj_dim = 2048
        self.num_image_tokens = 196
        self.num_text_tokens = 196

        self.learning_rate = 1e-4
        self.weight_decay = 0.5
        self.num_epochs = 100000
        
        self.train_image_dir = "/unity/g1/kgalla/datasets/train2014/"
        self.train_h5_path = "/unity/g1/kgalla/datasets/annotations/train/coco_train2014_captions.h5"
        self.batch_size = 128
        self.num_workers = 64

        self.val_image_dir = "/unity/g1/kgalla/datasets/val2014/"
        self.val_h5_path = "/unity/g1/kgalla/datasets/annotations/val/coco_val2014_captions.h5"