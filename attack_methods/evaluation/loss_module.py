import torch
from torch.nn import CrossEntropyLoss

IGNORE_INDEX = -100

class FeatureSimilarityLoss(torch.nn.Module):
    def __init__(self, model, is_targeted=True):
        super(FeatureSimilarityLoss, self).__init__()
        self.model = model
        self.is_targeted = is_targeted
        
    def forward(self, adv_pc, tgt_feature):
        
        if adv_pc.shape[-1] > 6:
            adv_pc = adv_pc[..., :6]
        # adv_pc = adv_pc.to(device)
        adv_feature = self.model(adv_pc)
        #tgt_feature = self.model(tgt_pc)
        
        # Normalize the features
        adv_feature = adv_feature / adv_feature.norm(dim=2, keepdim=True)
        #print(f"adv_feature: {adv_feature.shape}")
        
        # Compute the embedding similarity
        embedding_sim = torch.mean(torch.sum(adv_feature * tgt_feature, dim=2), dim=1)
        
        # Add L2 distance loss to improve feature similarity loss
        l2_loss = torch.mean(torch.norm(adv_feature - tgt_feature, dim=2))
        # Combine cosine similarity loss and L2 loss
        loss = (1 - embedding_sim) + l2_loss if self.is_targeted else embedding_sim - l2_loss
        return loss


class GPT4PointCaptionLoss(torch.nn.Module):
    def __init__(self, model, text_processor=None, is_targeted=True):
        super(GPT4PointCaptionLoss, self).__init__()
        self.model = model
        # self.dtype = dtype
        # self.allowed_class = allowed_class
        # self.vis_processors = vis_processors
        self.text_processor = text_processor
        # print("vis_processors:", vis_processors)
        # print("txt_processors:", txt_processors)
        self.is_targeted = is_targeted
    
        
    def forward(self, adv_pc, tgt_labels ):
        
        if adv_pc.shape[-1] > 6:
            adv_pc = adv_pc[..., :6]
        
        # adv_pc = adv_pc.to(dtype=self.dtype)
        captions = []
        for label in tgt_labels:
            caption = label
            if self.text_processor is not None:
                caption = self.text_processor(caption)
            captions.append(caption)

        sample = {
            "point": adv_pc,
            "text_input": captions,
        }
        
        outputs, targets = self.model(
            sample
        )
        # print(outputs["logits"].shape)
        
        loss = self.compute_loss(outputs["logits"], targets)
        
        if not self.is_targeted:
            loss = -1 * loss  # * for untargeted attack, we want to maximize the loss
        
        return loss
        
     
    def compute_loss(self, logits, labels):
        """
        Compute the loss for the adversarial point clouds.
        """
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()  # * B, L, V
        vocab_size = shift_logits.shape[-1]
        shift_labels = labels[..., 1:].contiguous()  # * B, L
        B = shift_labels.shape[0]
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction='none')  # Compute loss per token, ignoring -100
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model/pipeline parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        loss = loss.view(B, -1)
        loss = loss.sum(dim=1) / (loss != 0).sum(dim=1).clamp(min=1)  # Compute mean loss per batch, ignoring padding
        
        return loss   


    def multi_target_ce(
        self,
        logits: torch.Tensor,
        targets: torch.LongTensor,
        # allowed: torch.Tensor,
        ignore_index: int = -100,
        kappa: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute per-sample multi-target margin loss over time (or spatial) dimension.

        Args:
            logits:     (B, L, C) float tensor of model outputs.
            targets:    (B, L)   long tensor of true labels, or ignore_index.
            allowed:    bool tensor of shape (B, L, C) or (C,),
                        where True indicates a valid target class.
            ignore_index: int label in `targets` to skip.
            kappa:      margin constant; loss = ReLU(true_logit - lse_allowed + kappa).

        Returns:
            loss: Tensor of shape (B,) giving the mean loss over valid positions for each sample.
        """
        B, L, C = logits.shape
        device = logits.device

        allowed = self.allowed_class
        # Broadcast allowed mask if needed
        if allowed.dim() == 1:
            allowed = allowed.view(1, 1, C).expand(B, L, C)
        allowed = allowed.to(device)

        # Compute per-position loss
        # Gather true-logit: shape (B, L)
        idx = torch.arange(B, device=device)[:, None]
        pos = torch.arange(L, device=device)[None, :]
        true_logit = logits[idx, pos, targets.clamp(min=0)]  # we’ll mask out ignore_index next

        # Mask out invalid classes to -inf, then logsumexp over C
        inf_logits = logits.masked_fill(~allowed, float('-inf'))
        lse_allowed = torch.logsumexp(inf_logits, dim=2)    # shape (B, L)

        # Raw margin loss per position
        loss_raw = F.relu(true_logit - lse_allowed + kappa)  # (B, L)

        # Mask out positions where targets == ignore_index
        valid = (targets != ignore_index)                     # (B, L)
        loss_raw = loss_raw * valid.to(loss_raw.dtype)

        # Sum and count per sample
        sum_loss   = loss_raw.sum(dim=1)                      # (B,)
        count_loss = valid.sum(dim=1).clamp(min=1).to(sum_loss.dtype)  # (B,)

        # Per-sample mean (zero if no valid positions)
        loss_per_sample = sum_loss / count_loss               # (B,)

        return loss_per_sample




class PointLLMCaptionLoss(torch.nn.Module):
    def __init__(self, model, tokenizer, conv, is_targeted=True):
        super(PointLLMCaptionLoss, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.conv = conv
        self.is_targeted = is_targeted
    
    def generate_and_mask_target(
        self,
        conversations
    ):

        # Tokenize conversations
        input_ids = self.tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length= self.tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        targets = input_ids.clone()

        # Mask targets
        sep = self.conv.sep + self.conv.roles[1] + ": " 
        # print(f"sep: {sep}")
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(self.tokenizer.pad_token_id).sum())

            rounds = conversation.split(self.conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2: # * can handle padded tokens
                    break
                parts[0] += sep
                round_len = len(self.tokenizer(rou).input_ids)
                instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2

                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX # * this is necessary for padded tokens

            if cur_len < self.tokenizer.model_max_length:
                if cur_len != total_len: # * unk tokens in the dialogue will cause this.
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )
                    
            # # Detokenize targets to text
            # detokenized_targets = tokenizer.batch_decode(targets, skip_special_tokens=True)
            # print(f"Detokenized targets: {detokenized_targets}")

        return input_ids, targets


    def forward(self, adv_pc, tgt_labels, device='cuda'):
        
        if adv_pc.shape[-1] > 6:
            adv_pc = adv_pc[..., :6]
        
        # tgt_captions = generate_tgt_caption( tgt_labels, 1 )
        tgt_captions = tgt_labels
        
        conversations = []
        for tgt_caption in tgt_captions:
            tgt_conv = self.conv.copy()
            tgt_conv.append_message( self.conv.roles[1], tgt_caption )
            conversations.append( tgt_conv.get_prompt() )
        
        input_caption_ids, tgt_caption_ids = self.generate_and_mask_target(
            conversations
        )
        
        outputs = self.model(
                input_ids=input_caption_ids.to(device), 
                point_clouds= adv_pc.to(self.model.dtype),
                labels= tgt_caption_ids.to(device),
                return_dict=True,
                    )
        if self.is_targeted:
            loss = outputs.loss
        else:
            loss = -1 * outputs.loss
        # print(f"loss: {loss}")

        return loss
