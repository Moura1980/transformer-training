#dependências:
import math, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

#reprodutibilidade
torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#funções utilitárias do lab 4:
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k    = Q.size(-1)
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)  
    if mask is not None:
        scores = scores + mask
    return torch.softmax(scores, dim=-1) @ V               


def add_norm(x, sublayer_output, norm):
    return norm(x + sublayer_output)


def causal_mask(seq_len, device):
    m = torch.zeros(1, seq_len, seq_len, device=device)
    idx = torch.triu_indices(seq_len, seq_len, offset=1)
    m[0, idx[0], idx[1]] = float("-inf")
    return m


#sub-módulos
class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class MecanismoDeAtencao(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.WQ = nn.Linear(d_model, d_model, bias=False)
        self.WK = nn.Linear(d_model, d_model, bias=False)
        self.WV = nn.Linear(d_model, d_model, bias=False)

    def forward(self, Q_in, K_in, V_in, mask=None):
        Q = self.WQ(Q_in)
        K = self.WK(K_in)
        V = self.WV(V_in)
        return scaled_dot_product_attention(Q, K, V, mask)


#Encoder
class CamadaEncoder(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.atencao = MecanismoDeAtencao(d_model)
        self.ffn     = FFN(d_model, d_ff)
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)

    def processar(self, x):
        x = add_norm(x, self.atencao(x, x, x),  self.norm1)
        x = add_norm(x, self.ffn(x),              self.norm2)
        return x

    def forward(self, x):          
        return self.processar(x)


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, n_camadas):
        super().__init__()
        self.camadas = nn.ModuleList(
            [CamadaEncoder(d_model, d_ff) for _ in range(n_camadas)]
        )

    def forward(self, x):
        for camada in self.camadas:
            x = camada.processar(x)
        return x



#Decoder
class CamadaDecoder(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.masked_attn = MecanismoDeAtencao(d_model)
        self.cross_attn  = MecanismoDeAtencao(d_model)
        self.ffn         = FFN(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def processar(self, y, Z):
        seq_len = y.size(1)
        mask    = causal_mask(seq_len, y.device)

        y = add_norm(y, self.masked_attn(y, y, y, mask), self.norm1)
        y = add_norm(y, self.cross_attn(y, Z, Z),        self.norm2)
        y = add_norm(y, self.ffn(y),                      self.norm3)
        return y

    def forward(self, y, Z):
        return self.processar(y, Z)


class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, n_camadas, vocab_size):
        super().__init__()
        self.camadas = nn.ModuleList(
            [CamadaDecoder(d_model, d_ff) for _ in range(n_camadas)]
        )
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, y, Z):
        for camada in self.camadas:
            y = camada.processar(y, Z)
        return self.proj(y)          


#Modelo Completo com Embeddings
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model, d_ff, n_camadas):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, d_model, padding_idx=0)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model)
        self.encoder = Encoder(d_model, d_ff, n_camadas)
        self.decoder = Decoder(d_model, d_ff, n_camadas, tgt_vocab)

    def encode(self, src):
        return self.encoder(self.pos_enc(self.src_emb(src)))

    def decode(self, tgt, memory):
        return self.decoder(self.pos_enc(self.tgt_emb(tgt)), memory)

    def forward(self, src, tgt):
        return self.decode(tgt, self.encode(src))


#Datase Hugging Face
print("Carregando dataset")

raw = load_dataset("Helsinki-NLP/opus_books", "en-pt", split="train")
SUBSET_SIZE = 1000
raw = raw.select(range(SUBSET_SIZE))

src_texts = [ex["translation"]["en"] for ex in raw]
tgt_texts = [ex["translation"]["pt"] for ex in raw]

print(f"Exemplos carregados : {len(src_texts)}")
print(f"Exemplo SRC[0]: {src_texts[0]}")
print(f"Exemplo TGT[0]: {tgt_texts[0]}")


#Tokenização Básica
print("Tokenização")

TOKENIZER_NAME = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

#tokens especiais mapeados para os IDs do tokenizador
PAD_ID   = tokenizer.pad_token_id          
START_ID = tokenizer.cls_token_id          
EOS_ID   = tokenizer.sep_token_id         
VOCAB_SIZE = tokenizer.vocab_size

MAX_LEN = 64   

def tokenize_pair(src_list, tgt_list, max_len=MAX_LEN):
    src_ids_list, tgt_ids_list, lbl_ids_list = [], [], []

    for src, tgt in zip(src_list, tgt_list):
        src_enc = tokenizer(
            src, max_length=max_len, truncation=True, add_special_tokens=False
        )["input_ids"]
        src_padded = src_enc[:max_len] + [PAD_ID] * max(0, max_len - len(src_enc))

        tgt_enc = tokenizer(
            tgt, max_length=max_len - 1, truncation=True, add_special_tokens=False
        )["input_ids"]
        dec_in  = [START_ID] + tgt_enc
        dec_in  = dec_in[:max_len] + [PAD_ID] * max(0, max_len - len(dec_in))

        label   = tgt_enc + [EOS_ID]
        label   = label[:max_len] + [PAD_ID] * max(0, max_len - len(label))

        src_ids_list.append(src_padded)
        tgt_ids_list.append(dec_in)
        lbl_ids_list.append(label)

    return (
        torch.tensor(src_ids_list, dtype=torch.long),
        torch.tensor(tgt_ids_list, dtype=torch.long),
        torch.tensor(lbl_ids_list, dtype=torch.long),
    )

print("Tokenizando 1 000 pares")
SRC_T, TGT_T, LBL_T = tokenize_pair(src_texts, tgt_texts)
print(f"SRC shape : {SRC_T.shape}")   
print(f"TGT shape : {TGT_T.shape}")   
print(f"LBL shape : {LBL_T.shape}")   


class TranslationDataset(Dataset):
    def __init__(self, src, tgt, lbl):
        self.src, self.tgt, self.lbl = src, tgt, lbl

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        return self.src[i], self.tgt[i], self.lbl[i]


dataset    = TranslationDataset(SRC_T, TGT_T, LBL_T)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


#Training Loop
print("Training Loop")

#Hiperparâmetros
D_MODEL   = 128
D_FF      = 256
N_CAMADAS = 2
EPOCHS    = 15
LR        = 1e-3

model = Transformer(
    src_vocab  = VOCAB_SIZE,
    tgt_vocab  = VOCAB_SIZE,
    d_model    = D_MODEL,
    d_ff       = D_FF,
    n_camadas  = N_CAMADAS,
).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)   
optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)

loss_history = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0

    for src_batch, tgt_batch, lbl_batch in dataloader:
        src_batch = src_batch.to(DEVICE)   
        tgt_batch = tgt_batch.to(DEVICE)   
        lbl_batch = lbl_batch.to(DEVICE)   

        logits = model(src_batch, tgt_batch)   
        
        loss = criterion(logits.permute(0, 2, 1), lbl_batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    loss_history.append(avg_loss)
    print(f"Época {epoch:2d}/{EPOCHS}  |  Loss médio: {avg_loss:.4f}")

print("\nCurva de Loss:")
for i, l in enumerate(loss_history, 1):
    bar = "█" * int(l * 10)
    print(f"  Época {i:2d}: {l:.4f}  {bar}")


#Overfitting Test
print("TAREFA 4 – Overfitting Test (uma frase do conjunto de treino)")


#Frase de treino escolhida para o teste
FRASE_TESTE_EN = src_texts[0]
FRASE_TESTE_PT = tgt_texts[0]
print(f"Frase de entrada  (EN): {FRASE_TESTE_EN}")
print(f"Tradução esperada (PT): {FRASE_TESTE_PT}\n")

model.eval()

#Tokeniza a fonte
src_ids = tokenizer(
    FRASE_TESTE_EN, max_length=MAX_LEN, truncation=True, add_special_tokens=False
)["input_ids"]
src_ids += [PAD_ID] * max(0, MAX_LEN - len(src_ids))
src_tensor = torch.tensor([src_ids], dtype=torch.long, device=DEVICE)

with torch.no_grad():
    memory = model.encode(src_tensor)


#Loop auto-regressivo
dec_ids       = [START_ID]
palavras_ids  = []
MAX_DECODE    = 50

print("Geração auto-regressiva:")
for step in range(MAX_DECODE):
    dec_tensor = torch.tensor([dec_ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        logits = model.decode(dec_tensor, memory)  
    next_id = int(logits[0, -1].argmax())          
    if next_id == EOS_ID:
        print(f"  Passo {step+1:2d}: <EOS> → fim da geração")
        break
    dec_ids.append(next_id)
    palavras_ids.append(next_id)
    token_str = tokenizer.decode([next_id], skip_special_tokens=True)
    print(f"  Passo {step+1:2d}: '{token_str}' (id={next_id})")

traducao = tokenizer.decode(palavras_ids, skip_special_tokens=True)
print(f"\nTradução gerada : {traducao}")
print(f"Tradução esperada: {FRASE_TESTE_PT}")


#VERIFICAÇÕES DE SANIDADE
print("VERIFICAÇÕES DE SANIDADE")
with torch.no_grad():
    sample_src = SRC_T[:2].to(DEVICE)
    sample_tgt = TGT_T[:2].to(DEVICE)
    out = model(sample_src, sample_tgt)

print(f"  Entrada  SRC : {sample_src.shape}")
print(f"  Entrada  TGT : {sample_tgt.shape}")
print(f"  Saída logits : {out.shape}   (esperado: [2, {MAX_LEN}, {VOCAB_SIZE}])")
assert out.shape == (2, MAX_LEN, VOCAB_SIZE)
print(f"Loss final ({loss_history[-1]:.4f}) < Loss inicial ({loss_history[0]:.4f})")
assert loss_history[-1] < loss_history[0]