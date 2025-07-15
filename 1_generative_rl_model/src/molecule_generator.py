# src/molecule_generator.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.optim.lr_scheduler import ReduceLROnPlateau
import selfies as sf
import random
import logging
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

class RNN(nn.Module):
    """Архитектура рекуррентной нейронной сети (LSTM)."""
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=512, num_layers=3):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, h=None):
        x = self.embedding(x)
        output, h = self.lstm(x, h)
        logits = self.fc(output)
        return logits, h

class MoleculeGeneratorRNN:
    """Обертка для RNN, управляющая словарем, обучением и генерацией молекул."""
    def __init__(self, all_smiles_for_vocab, max_len=128, device='cpu'):
        self.device = device
        self.max_len = max_len
        self._setup_vocab(all_smiles_for_vocab)
        self.model = RNN(self.vocab_size, embedding_dim=128, hidden_dim=512, num_layers=3).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    def _setup_vocab(self, all_smiles):
        logger.info("Создание словаря SELFIES...")
        all_selfies = [sf.encoder(s) for s in tqdm(all_smiles, desc="Кодирование SMILES в SELFIES") if s]
        self.alphabet = sorted(list(sf.get_alphabet_from_selfies(all_selfies)))
        self.alphabet.extend(['<nop>', '<sos>', '<eos>']) # Специальные токены
        self.vocab = {s: i for i, s in enumerate(self.alphabet)}
        self.inv_vocab = {i: s for s, i in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        logger.info(f"Размер словаря: {self.vocab_size}")
    
    def _selfies_to_tensor(self, selfies):
        tokens = ['<sos>'] + list(sf.split_selfies(selfies)) + ['<eos>']
        if len(tokens) > self.max_len: return None
        return torch.tensor([self.vocab.get(t, self.vocab['<nop>']) for t in tokens], dtype=torch.long, device=self.device)

    def pretrain(self, train_smiles, val_smiles, epochs, batch_size, patience=3):
        criterion = nn.CrossEntropyLoss(ignore_index=self.vocab['<nop>'])
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            self.model.train()
            random.shuffle(train_smiles)
            total_train_loss = 0
            
            pbar = tqdm(range(0, len(train_smiles), batch_size), desc=f"Предобучение. Эпоха {epoch+1}/{epochs}")
            for i in pbar:
                batch_smiles = train_smiles[i:i+batch_size]
                batch_selfies = [sf.encoder(s) for s in batch_smiles]
                batch_tensors = [self._selfies_to_tensor(s) for s in batch_selfies if s]
                if not batch_tensors: continue
                
                padded = nn.utils.rnn.pad_sequence(batch_tensors, batch_first=True, padding_value=self.vocab['<nop>'])
                inputs, targets = padded[:, :-1], padded[:, 1:]
                
                logits, _ = self.model(inputs)
                loss = criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_train_loss += loss.item()
                pbar.set_postfix({'train_loss': loss.item()})

            avg_train_loss = total_train_loss / max(1, len(train_smiles) / batch_size)
            avg_val_loss = self._validate(val_smiles, batch_size, criterion)
            self.scheduler.step(avg_val_loss)

            logger.info(f"Эпоха {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                self.save_model("models/generator_pretrained_best.pth")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(f"Ранняя остановка на эпохе {epoch+1}")
                    break
    
    def _validate(self, val_smiles, batch_size, criterion):
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for i in range(0, len(val_smiles), batch_size):
                batch_smiles = val_smiles[i:i+batch_size]
                batch_selfies = [sf.encoder(s) for s in batch_smiles]
                batch_tensors = [self._selfies_to_tensor(s) for s in batch_selfies if s]
                if not batch_tensors: continue
                
                padded = nn.utils.rnn.pad_sequence(batch_tensors, batch_first=True, padding_value=self.vocab['<nop>'])
                inputs, targets = padded[:, :-1], padded[:, 1:]
                logits, _ = self.model(inputs)
                loss = criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
                total_val_loss += loss.item()

        return total_val_loss / max(1, len(val_smiles) / batch_size)

    @torch.no_grad()
    def sample(self, num_samples, temperature=1.0):
        self.model.eval()
        sampled_selfies = []
        for _ in tqdm(range(num_samples), desc="Генерация молекул (сэмплирование)", leave=False):
            x = torch.tensor([[self.vocab['<sos>']]], device=self.device)
            h = None
            sequence = []
            for _ in range(self.max_len):
                logits, h = self.model(x, h)
                probs = torch.softmax(logits.squeeze(0) / temperature, dim=-1)
                cat_dist = Categorical(probs)
                next_token_idx = cat_dist.sample()
                
                if next_token_idx.item() == self.vocab['<eos>']: break
                sequence.append(self.inv_vocab[next_token_idx.item()])
                x = next_token_idx.unsqueeze(0).unsqueeze(0)
            
            sampled_selfies.append("".join(sequence))
        return sampled_selfies
    
    def train_step(self, smiles_batch, rewards):
        self.model.train()
        total_loss = 0
        for smiles, reward in zip(smiles_batch, rewards):
            try:
                selfies = sf.encoder(smiles)
                tensor = self._selfies_to_tensor(selfies)
                if tensor is None: continue
            except:
                continue

            inputs, targets = tensor[:-1].unsqueeze(0), tensor[1:].unsqueeze(0)
            logits, _ = self.model(inputs)
            log_probs = torch.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(2, targets.unsqueeze(2)).squeeze()
            policy_loss = -torch.sum(action_log_probs) * reward
            total_loss += policy_loss
            
        if total_loss == 0: return 0.0
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item() / max(1, len(smiles_batch))

    def mutate_selfies(self, selfies_str, mutation_rate=0.1):
        tokens = list(sf.split_selfies(selfies_str))
        for i in range(len(tokens)):
            if random.random() < mutation_rate:
                tokens[i] = random.choice(self.alphabet[:-3]) # Исключаем спец. токены
        return "".join(tokens)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        logger.info(f"Модель генератора сохранена в: {path}")
    
    def load_model(self, path):
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            logger.info(f"Модель генератора загружена из: {path}")
        except FileNotFoundError:
            logger.error(f"Файл модели не найден: {path}. Обучение начнется с нуля.")
        except Exception as e:
            logger.error(f"Не удалось загрузить модель {path}: {e}")
