
import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGramModel(nn.Module):
    """
    Center word: w_t
      |
      v
    [in_emb]  <- input embedding of center word
      |
      v
    hidden (optional)
      |
      v
    +-----------+-----------+-----------+-----------+
    |           |           |           |           |
    v           v           v           v
    [out_emb] [out_emb] [out_emb] [out_emb]  <- predict context words w_{t-2}, w_{t-1}, w_{t+1}, w_{t+2}
    """
    def __init__(self, vocab_size, embedding_dim, neg_sampler):
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # Input embeddings (center word)
        self.in_embed = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        # Output embeddings (context word)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.neg_sampler=neg_sampler
        self.dummy=nn.Linear(1,1)
        # Initialize embeddings
        self._init_embeddings()

    def _init_embeddings(self):
        initrange = 0.5 / self.embedding_dim
        self.in_embed.weight.data.uniform_(-initrange, initrange)
        self.out_embed.weight.data.zero_()

    def train_forward(self, train_data, device):
        center, context = train_data
        center, context = center.to(device), context.to(device)
        # loss for positives: -log(sigmoid(pos_score)) = softplus(-pos_score)
        v_c = self.in_embed(center)  # [B, D]
        context_embeds = self.out_embed(context)  # [B, D]
        # ---- Positive logits ----
        pos_score = (v_c * context_embeds).sum(dim=-1)  # [B]
        pos_loss = F.softplus(-pos_score).mean()
        
        # ---- Negative samples
        batch_size = context_embeds.size(0)
        neg_context = self.neg_sampler.sample(batch_size, self.neg_sampler.K).to(device)   # [B, K]
        neg_embeds = self.out_embed(neg_context)  # [B, K, D]
        neg_score = torch.bmm(neg_embeds, v_c.unsqueeze(2)).squeeze(2) # bkd x bd1 [B, K]
        neg_loss = F.softplus(neg_score).sum(dim=1).mean()


        return pos_loss + neg_loss

    def get_embeddings(self):
        return self.in_embed.weight.data


class CBOWModel(nn.Module):
    """
    Context words: w_{t-2}, w_{t-1}, w_{t+1}, w_{t+2}
              |        |        |        |
              v        v        v        v
           [in_emb] [in_emb] [in_emb] [in_emb]  <- input embeddings (context)
               \      |      |      /
                \     |      |     /
                 ---> Average / Concatenate ---> hidden layer (optional tanh)
                           |
                           v
                      [out_emb] dot
                           |
                           v
                       Predicted center word
    """
    def __init__(self, vocab_size, embedding_dim,neg_sampler):
        super(CBOWModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.neg_sampler=neg_sampler
        self.in_embed = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.linear = nn.Linear(embedding_dim, vocab_size)

        # Initialize embeddings
        initrange = 0.5 / self.embedding_dim
        self.in_embed.weight.data.uniform_(-initrange, initrange)
        self.out_embed.weight.data.zero_()

    def train_forward(self, train_data, device):
        center, context = train_data # [B, 1], [B, K]
        center, context = center.to(device), context.to(device)
        context_embeds = self.in_embed(context) #[B, K, D]
        

        context_embeds = context_embeds.mean(dim=1) # [B, D]

        # ---- Positive logits ----
        true_embeds = self.out_embed(center.squeeze(1))       # [B, D]
        pos_logits = torch.sum(context_embeds * true_embeds, dim=1)  # [B]
        # if torch.isnan(pos_logits).sum():
        #     1==1
        pos_loss = F.softplus(-pos_logits).mean()   # -F.logsigmoid(pos_score)

        # ---- Negative samples ----
        batch_size = context_embeds.size(0)
        neg_context = self.neg_sampler.sample(batch_size, self.neg_sampler.K).to(device)   # [B, K]
        neg_embeds = self.out_embed(neg_context)  # [B, K, D]
        neg_score = torch.bmm(neg_embeds, context_embeds.unsqueeze(2)).squeeze(2) # bkd x bd1 [B, K]
        neg_loss = F.softplus(neg_score).sum(dim=1).mean()
        return pos_loss + neg_loss
    def get_embeddings(self):
        return self.in_embed.weight.data


class NNLMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, context_size, neg_sampler):
        super(NNLMModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.context_size = context_size
        self.neg_sampler = neg_sampler

        # Input embeddings
        self.in_embed = nn.Embedding(vocab_size, embedding_dim, sparse=True)

        # Projection + hidden layer
        self.linear1 = nn.Linear(embedding_dim * context_size, hidden_dim)
        self.tanh = nn.Tanh()

        # Output embeddings (like word2vec’s out-embedding table)
        self.out_embed = nn.Embedding(vocab_size, hidden_dim, sparse=True)

        # Initialize
        initrange = 0.5 / self.embedding_dim
        self.in_embed.weight.data.uniform_(-initrange, initrange)
        self.out_embed.weight.data.uniform_(-initrange, initrange)

    def train_forward(self, train_data, device):
        center, context = train_data
        center, context = center.to(device), context.to(device)#, presence.to(device)
        # Average context word embeddings
        batch_size = center.size(0)

        # --- Input pipeline ---
        context_embeds = self.in_embed(context)#*presence.unsqueeze(2)                # [B, context, D]
        context_embeds = context_embeds.view(batch_size, -1)   # [B, context*D]

        hidden = self.tanh(self.linear1(context_embeds))               # [B, H]

        # --- Positive logits ---
        true_embeds = self.out_embed(center)          # [B, H]
        pos_logits = torch.sum(hidden * true_embeds, dim=1)     # [B]
        pos_loss = F.softplus(-pos_logits).mean()               # -log σ(uᵀv)

        # --- Negative logits ---
        neg_context = self.neg_sampler.sample(batch_size, self.neg_sampler.K).to(device)   # [B, K]
        neg_embeds = self.out_embed(neg_context)  # [B, K, H]
        neg_score = torch.bmm(neg_embeds, hidden.unsqueeze(2)).squeeze(2) # bkh x bh1 [B, K]
        neg_loss = F.softplus(neg_score).sum(dim=1).mean()

        return pos_loss + neg_loss
    def get_embeddings(self):
        return self.in_embed.weight.data
