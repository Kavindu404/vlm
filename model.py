import torch
import torch.nn as nn

from decoder import MultimodalDecoder, MultimodalProjector, PosEmbeddings
from encoder import VisionTransformer

class MultimodalCaptionGenerator(nn.Module):
    def __init__(self, encoder_config, decoder_config):
        super().__init__()
        self.image_encoder = VisionTransformer(encoder_config)
        self.multimodal_projector = MultimodalProjector(decoder_config)
        self.decoder = MultimodalDecoder(decoder_config)
        self.decoder_pos_embeds = PosEmbeddings(decoder_config)
        self.output_projection = nn.Linear(decoder_config.hidden_size, decoder_config.vocab_size)
    
    def _merge_input_embeds(self, image_embeds, text_embeds):
        batch_size = image_embeds.shape[0]
        num_image_tokens = image_embeds.shape[1]
        num_text_tokens = text_embeds.shape[1]
        seq_len = num_image_tokens + num_text_tokens

        merged_embeds = torch.concat([image_embeds, text_embeds], dim=1)

        attention_mask = torch.ones((batch_size, seq_len, seq_len), device=image_embeds.device)

        # Note that all the features of the padding token is 0 from the Embedding look-up table
        # This mask is of shape [batch_size, num_text_tokens]
        text_pad_mask = (text_embeds.sum(dim=-1) != 0).bool()
        # Now we need the causal mask for the text generation part
        text_causal_mask = torch.tril(torch.ones((num_text_tokens, num_text_tokens))).bool()

        attention_mask[:, num_image_tokens:, num_image_tokens:] = text_pad_mask.unsqueeze(-1) & text_causal_mask
        attention_mask = attention_mask.float()
        if attention_mask.size() != (batch_size, seq_len, seq_len):

            raise ValueError(
                f"Size mismatch: Attention Mask: {attention_mask.size()}, expected: {(batch_size, seq_len, seq_len)}"
            )
  
        return merged_embeds, attention_mask
    
    def forward(self, text_tokens, imgs):

        text_embeds = self.decoder.get_input_embeddings()(text_tokens)
        img_patches = self.image_encoder(imgs)

        img_embeds = self.multimodal_projector(img_patches)

        input_embeds, attention_mask = self._merge_input_embeds(img_embeds, text_embeds)

        input_embeds = self.decoder_pos_embeds(input_embeds)

        outputs, attention_scores = self.decoder(
            input_embeds,
            attention_mask
        )

        logits = self.output_projection(outputs)

        return logits, attention_scores

    

