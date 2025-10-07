'''
=======================================
TRANSFORMER: Traductor Java ↔ Python
Autores:
Axel Octavio Alcantara Gomez
Jose Damian Herrera Zepda
=======================================
'''

# ======================================
# PASO 1: Importar librerías y cargar dataset
# ======================================

import torch
import torch.nn as nn
import torch.optim as optim
import math
import pandas as pd
import re
from typing import List
from torch.utils.data import Dataset, DataLoader
import time



# Cargar dataset desde Excel (debe tener columnas: java_code, python_code)
df = pd.read_excel('java_python_dataset.xlsx')

print("✅ Dataset cargado correctamente:")
print(df.head())


# =========================
# PASO 2: Tokenizador (mejor para código)
# =========================
# Patrón: strings, identificadores, números, operadores multi-char y símbolos individuales
TOKEN_PATTERN = re.compile(
    r'\".*?\"|\'.*?\'|\b[_A-Za-z]\w*\b|\d+\.\d+|\d+|==|!=|<=|>=|->|\+\+|--|&&|\|\||[+\-*/%<>]=?|[{}()\[\];,.:=]'
)

def tokenize(code: str) -> List[str]:
    if not isinstance(code, str):
        code = str(code)
    tokens = TOKEN_PATTERN.findall(code)
    # si no encuentra tokens, usa cada caracter (fallback)
    return tokens if tokens else list(code)

def detokenize(tokens: List[str]) -> str:
    s = ' '.join(tokens)
    # quitar espacios antes de: , ; . : ) ] }
    s = re.sub(r'\s+([,;.\:\)\]\}])', r'\1', s)
    # quitar espacio después de: ( [ {
    s = re.sub(r'([\(\[\{])\s+', r'\1', s)
    # arreglar ": '" inside strings (no tocar strings)
    return s.strip()

# =========================
# PASO 3: Construir vocabulario de tokens
# =========================
all_tokens = []
for code in df['java_code'].tolist() + df['python_code'].tolist():
    all_tokens.extend(tokenize(code))

unique_tokens = sorted(set(all_tokens))
SPECIAL_TOKENS = ['<pad>', '<sos>', '<eos>']
vocab = SPECIAL_TOKENS + unique_tokens

token2idx = {tok: i for i, tok in enumerate(vocab)}
idx2token = {i: tok for tok, i in token2idx.items()}

PAD_IDX = token2idx['<pad>']
SOS_IDX = token2idx['<sos>']
EOS_IDX = token2idx['<eos>']

print(f"Vocab size: {len(vocab)} (incluye tokens especiales)")

# =========================
# PASO 4: Funciones encode/decode para tokens
# =========================
def encode_tokens(tokens: List[str]) -> List[int]:
    return [token2idx.get(t, token2idx['<pad>']) for t in tokens]

def encode_text(text: str) -> List[int]:
    return encode_tokens(tokenize(text))

def decode_indices(indices: List[int]) -> List[str]:
    return [idx2token[i] for i in indices if i in idx2token]

def add_sos_eos(indices: List[int]) -> List[int]:
    return [SOS_IDX] + indices + [EOS_IDX]

# =========================
# PASO 5: Preparar secuencias y dataset
# =========================
src_seqs = [encode_text(code) for code in df['java_code'].tolist()]
tgt_seqs = [add_sos_eos(encode_text(code)) for code in df['python_code'].tolist()]

# Si dataset muy pequeño, avisar
if len(src_seqs) < 50:
    print("⚠️ Aviso: tu dataset es pequeño (<50). Considera añadir más ejemplos para mejores resultados.")

class CodeDataset(Dataset):
    def __init__(self, src_seqs, tgt_seqs):
        self.src = src_seqs
        self.tgt = tgt_seqs
    def __len__(self):
        return len(self.src)
    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_lens = [len(x) for x in src_batch]
    tgt_lens = [len(x) for x in tgt_batch]
    max_src = max(src_lens)
    max_tgt = max(tgt_lens)
    src_padded = [x + [PAD_IDX] * (max_src - len(x)) for x in src_batch]
    tgt_padded = [x + [PAD_IDX] * (max_tgt - len(x)) for x in tgt_batch]
    return torch.tensor(src_padded, dtype=torch.long), torch.tensor(tgt_padded, dtype=torch.long)

dataset = CodeDataset(src_seqs, tgt_seqs)
loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

# =========================
# PASO 6: Modelo Transformer (estilo de la maestra, con soporte padding masks)
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=256, nhead=8, num_encoder_layers=3,
                 num_decoder_layers=3, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_IDX)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_encoder = PositionalEncoding(d_model)
        self.pos_decoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model

    def forward(self, src, tgt, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        # src: (batch, src_len), tgt: (batch, tgt_len)
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_decoder(tgt_emb)
        # transpose to (seq_len, batch, d_model)
        src_emb = src_emb.transpose(0, 1)
        tgt_emb = tgt_emb.transpose(0, 1)
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=src_key_padding_mask)
        output = output.transpose(0, 1)  # back to (batch, seq_len, d_model)
        return self.output_layer(output)

# =========================
# PASO 7: Setup entrenamiento
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Entrenando en:", device)

model = TransformerModel(len(vocab), len(vocab)).to(device)

def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones((sz, sz), device=device) * float('-inf'), diagonal=1)
    return mask

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# =========================
# PASO 8: Entrenar (batches)
# =========================
NUM_EPOCHS = 30  # aumenta si tienes más datos y tiempo
print("\n🚀 Entrenamiento iniciando...\n")
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    t0 = time.time()
    for src_batch, tgt_batch in loader:
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)
        tgt_input = tgt_batch[:, :-1]
        tgt_output = tgt_batch[:, 1:]

        tgt_mask = generate_square_subsequent_mask(tgt_input.size(1))
        src_key_padding_mask = (src_batch == PAD_IDX)  # shape (batch, src_len)
        tgt_key_padding_mask = (tgt_input == PAD_IDX)

        optimizer.zero_grad()
        logits = model(src_batch, tgt_input, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {epoch_loss/len(loader):.4f} | time: {time.time()-t0:.1f}s")

# =========================
# PASO 9: Inferencia (greedy con detokenize)
# =========================
model.eval()
MAX_TGT_LEN = 200

def predict(code: str, max_len=MAX_TGT_LEN):
    tokens = encode_text(code)
    src = torch.tensor([tokens + [PAD_IDX] * 0], dtype=torch.long).to(device)  # shape (1, src_len)
    src_key_padding_mask = (src == PAD_IDX)
    # encode memory
    with torch.no_grad():
        src_emb = model.src_embedding(src) * math.sqrt(model.d_model)
        src_emb = model.pos_encoder(src_emb)
        memory = model.encoder(src_emb.transpose(0,1), src_key_padding_mask=src_key_padding_mask)

        ys = torch.tensor([[SOS_IDX]], dtype=torch.long).to(device)  # (1,1)
        for i in range(max_len):
            tgt_emb = model.tgt_embedding(ys) * math.sqrt(model.d_model)
            tgt_emb = model.pos_decoder(tgt_emb)
            out = model.decoder(tgt_emb.transpose(0,1), memory)  # (tgt_len, batch, d_model)
            out = out.transpose(0,1)  # (batch, tgt_len, d_model)
            logits = model.output_layer(out)  # (batch, tgt_len, vocab)
            next_token_logits = logits[0, -1, :]
            next_token = next_token_logits.argmax().unsqueeze(0).unsqueeze(0)  # (1,1)
            ys = torch.cat([ys, next_token], dim=1)
            if next_token.item() == EOS_IDX:
                break
        pred_indices = ys.squeeze(0).cpu().tolist()
        # quitar primer <sos>
        if pred_indices and pred_indices[0] == SOS_IDX:
            pred_indices = pred_indices[1:]
        # eliminar todo despues del <eos>
        if EOS_IDX in pred_indices:
            pred_indices = pred_indices[:pred_indices.index(EOS_IDX)]
        pred_tokens = decode_indices(pred_indices)
        return detokenize(pred_tokens)

# =========================
# PASO 10: Probar con ejemplos del dataset
# =========================
print("\n💡 Traducciones de prueba (Java -> Python):\n")
for i, row in df.iterrows():
    java = row['java_code']
    gold_py = row['python_code']
    pred = predict(java)
    print("Java:  ", java)
    print("Esperado (gold):", gold_py)
    print("Predicho:        ", pred)
    print("-" * 60)
    # mostrar solo los primeros 10 ejemplos
    if i >= 9:
        break