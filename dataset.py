import os
import re
import torch
from datasets import load_dataset

DATA_PATH = 'data/turk_lm_data.txt'


def download_turkish_data(sample_size=100000):
    if os.path.isfile(DATA_PATH):
        print(f'Dataset already exists: {DATA_PATH}')
        return

    os.makedirs('data', exist_ok=True)

    print("Connecting to HuggingFace Stream...")
    # streaming=True diyerek DİSKİ DOLDURMADAN veriyi canlı akışla alıyoruz!
    dataset_stream = load_dataset("vngrs-ai/vngrs-web-corpus", split="train", streaming=True)

    spam_pattern = re.compile(r'(bahis|fenomenbet|casino|tahinpekmez|http|www\.)', re.IGNORECASE)

    saved_count = 0
    print(f"Data is being streamed, cleaned, and written to {DATA_PATH}....")

    with open(DATA_PATH, "w", encoding="utf-8") as f:
        for item in dataset_stream:
            text = item["text"].strip()

            # Filtreleme
            if len(text) >= 40 and not spam_pattern.search(text):
                clean_text = " ".join(text.split())
                f.write(clean_text + "\n")
                saved_count += 1

                # İstenen temiz örnek sayısına ulaşıldığında akışı durdur
                if saved_count >= sample_size:
                    break

                if saved_count % 10000 == 0:
                    print(f"Processed & saved: {saved_count}/{sample_size} samples...")

    print(f"Process Completed! Total clean samples saved: {saved_count}")


class CharecterTokenizer:
    def __init__(self, text: str):
        self.characters = sorted(list(set(text)))
        self.vocab_size = len(self.characters)

        self.char_to_id = {char: i for i, char in enumerate(self.characters)}
        self.id_to_char = {i: char for i, char in enumerate(self.characters)}

        print(f"Vocabulary size: {self.vocab_size} characters")
        print(f"Characters: {repr(''.join(self.characters[:50]))}...")

    def encode(self, text: str) -> list:
        return [self.char_to_id[char] for char in text if char in self.char_to_id]

    def decode(self, ids: list) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return ''.join([self.id_to_char[id] for id in ids])


def get_batch(data: torch.Tensor, block_size: int, batch_size: int):
    max_start = len(data) - block_size - 1
    positions = torch.randint(max_start, (batch_size,))

    x_list = []
    y_list = []

    for pos in positions:
        x_list.append(data[pos:pos + block_size])
        y_list.append(data[pos + 1:pos + block_size + 1])

    x = torch.stack(x_list)
    y = torch.stack(y_list)

    return x, y


def load_data(block_size: int = 256, train_split: float = 0.8):
    download_turkish_data()

    with open(DATA_PATH, 'r', encoding='utf-8') as file:
        text = file.read()

    print(f"\nDataset size: {len(text):,} characters")
    print(f"Sample text:\n{text[:200]}")
    print("..." + "-" * 50)

    tokenizer = CharecterTokenizer(text)
    all_ids = tokenizer.encode(text)

    data = torch.tensor(all_ids, dtype=torch.long)

    split_index = int(train_split * len(data))
    train_data = data[:split_index]
    val_data = data[split_index:]

    print(f"\nTrain size: {len(train_data):,} tokens")
    print(f"Val size: {len(val_data):,} tokens")

    return train_data, val_data, tokenizer


if __name__ == '__main__':
    train_data, val_data, tokenizer = load_data()
    x, y = get_batch(train_data, block_size=128, batch_size=4)

    print(f"\nSample batch:")
    print(f"  Input shape: {x.shape}")
    print(f"  Target shape: {y.shape}")