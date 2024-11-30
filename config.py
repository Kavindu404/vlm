class VisionConfig:

    def __init__(self):
        self.patch_sz = 16
        self.embed_dim = 384
        self.img_sz = 224
        self.inter_size = 512
        self.num_heads = 4
        self.num_encoder_layers = 6
        self.layer_norm_eps = 1e-6
        self.in_channels = 3
        self.dropout_prob = 0.3
        self.attention_dropout = 0.3

class MultimodalConfig:

    def __init__(self):

        self.project_name = "base_vqa_self_attn_sinPos_gqa_v1"
        self.eps = 1e-6
        self.hidden_size = 512
        self.seq_len = 392
        self.max_len = 196
        self.inter_dim = 1024
        self.num_heads = 4
        self.vocab_size = 50261
        self.padding_idx = 50258
        self.sos_token_id = 50257
        self.eos_token_id = 50256
        self.q_start_token_id = 50259
        self.q_end_token_id = 50260
        self.num_decoder_layers = 8
        self.emb_dim = 384
        self.proj_dim = self.hidden_size
        self.num_image_tokens = 196
        self.num_q_tokens = 48
        self.num_a_tokens = 48
        self.max_question_len = 48

        self.learning_rate = 1e-5
        self.weight_decay = 0.1
        self.warmup_ratio = 0.1
        self.num_epochs = 2000
        self.dropout_prob = 0.3
        self.attention_dropout = 0.3

        self.fixed_batch_idx = 1
        self.fixed_sample_idx = 2
        
        self.train_image_dir = "/unity/g1/kgalla/datasets/gqa/images"
        self.train_h5_path = "/unity/g1/kgalla/datasets/gqa/gqa_train.h5"
        self.batch_size = 196
        self.num_workers = 164

        self.val_image_dir = "/unity/g1/kgalla/datasets/gqa/images"
        self.val_h5_path = "/unity/g1/kgalla/datasets/gqa/gqa_val.h5"

        self.gpu_ids = [0, 1, 2, 3]
        self.cuda_visible_devices = "0,1,2,3"