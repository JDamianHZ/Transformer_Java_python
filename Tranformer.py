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

# Cargar dataset desde Excel (debe tener columnas: java_code, python_code)
df = pd.read_excel('java_python_dataset.xlsx')

print("✅ Dataset cargado correctamente:")
print(df.head())


# ======================================
# PASO 2: Crear vocabulario
# ======================================

# Unir todos los textos para crear un vocabulario de caracteres únicos
all_text = ''.join(df['java_code'].tolist() + df['python_code'].tolist())
vocab = sorted(list(set(all_text)))
vocab = ['<pad>', '<sos>', '<eos>'] + vocab

# Diccionarios para codificar y decodificar
char2idx = {ch: idx for idx, ch in enumerate(vocab)}
idx2char = {idx: ch for ch, idx in char2idx.items()}


# ======================================
# PASO 3: Funciones de utilidad (encode, decode, pad)
# ======================================

def encode(seq):
    """Convierte texto a una lista de índices"""
    return [char2idx[c] for c in seq if c in char2idx]


def decode(seq):
    """Convierte índices de nuevo a texto"""
    return ''.join([
        idx2char[i] for i in seq
        if i not in [char2idx['<pad>'], char2idx['<sos>'], char2idx['<eos>']]
    ])


def add_tokens(seq):
    """Agrega tokens de inicio y fin"""
    return [char2idx['<sos>']] + encode(seq) + [char2idx['<eos>']]


def pad(seq, max_len):
    """Rellena secuencias al mismo tamaño"""
    return seq + [char2idx['<pad>']] * (max_len - len(seq))


# ======================================
# PASO 4: Preparar dataset (Java → Python)
# ======================================

src_seqs = [encode(code) for code in df['java_code']]
tgt_seqs = [add_tokens(code) for code in df['python_code']]

max_src_len = max(len(seq) for seq in src_seqs)
max_tgt_len = max(len(seq) for seq in tgt_seqs)

src_padded = torch.tensor([pad(seq, max_src_len) for seq in src_seqs])
tgt_padded = torch.tensor([pad(seq, max_tgt_len) for seq in tgt_seqs])


# ======================================
# PASO 5: Definir el modelo Transformer
# ======================================

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
                 d_model=128, nhead=4, num_encoder_layers=2,
                 num_decoder_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        self.pos_decoder = PositionalEncoding(d_model)
        # Encoder y decoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        # Capa final
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model

    def forward(self, src, tgt, tgt_mask=None):
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_decoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        src_emb = src_emb.transpose(0, 1)
        tgt_emb = tgt_emb.transpose(0, 1)
        memory = self.encoder(src_emb)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        output = output.transpose(0, 1)
        return self.output_layer(output)


# ======================================
# PASO 6: Inicializar modelo, máscara y optimizador
# ======================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"⚙️ Entrenando en: {device}")

model = TransformerModel(
    src_vocab_size=len(vocab),
    tgt_vocab_size=len(vocab),
    d_model=128,
    nhead=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=256,
    dropout=0.1
).to(device)


def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones((sz, sz)) * float('-inf'), diagonal=1)
    return mask


criterion = nn.CrossEntropyLoss(ignore_index=char2idx['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ======================================
# PASO 7: Entrenamiento
# ======================================

print("\n🚀 Iniciando entrenamiento...\n")

for epoch in range(50):
    model.train()
    optimizer.zero_grad()

    src = src_padded.to(device)
    tgt_input = tgt_padded[:, :-1].to(device)
    tgt_output = tgt_padded[:, 1:].to(device)

    tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)
    output = model(src, tgt_input, tgt_mask=tgt_mask)

    output = output.reshape(-1, output.shape[-1])
    tgt_output = tgt_output.reshape(-1)

    loss = criterion(output, tgt_output)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")


# ======================================
# PASO 8: Inferencia (Predicción)
# ======================================

model.eval()

def predict(code):
    src = torch.tensor([pad(encode(code), max_src_len)]).to(device)
    memory = model.encoder(model.pos_encoder(model.src_embedding(src).transpose(0, 1)))
    ys = torch.ones(1, 1).fill_(char2idx['<sos>']).type(torch.long).to(device)
    for i in range(max_tgt_len - 1):
        tgt_emb = model.pos_decoder(model.tgt_embedding(ys).transpose(0, 1))
        out = model.decoder(tgt_emb, memory)
        out = model.output_layer(out)
        next_char = out[-1, :, :].argmax(dim=1)
        ys = torch.cat([ys, next_char.unsqueeze(0).transpose(0, 1)], dim=1)
    return decode(ys.squeeze(0).tolist())


print("\n💡 Ejemplos de traducción Java → Python:\n")
for code in df['java_code']:
    print(f"Java: {code}\nPython traducido: {predict(code)}\n")
