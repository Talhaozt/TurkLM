import torch

from model import GPT
from dataset import *

def load_model(checkpoint_path: str = 'checkpoints/model.pt'):

    device = torch.device('mps' if not torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint['config']

    download_turkish_data()

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read()

    tokenizer = CharecterTokenizer(text)

    model = GPT(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        block_size=config['block_size'],
        dropout=0.0
    )

    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict'))
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model, tokenizer, device


@torch.no_grad()
def generate(model, tokenizer, device, prompt: str, max_tokens: int = 500, temperature: float = 0.8):
    promt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(promt_ids, dtype=torch.long, device=device).unsqueeze(0)
    output_ids = model.generate(input_ids, max_new_tokens=max_tokens, temperature=temperature)
    generated_text = tokenizer.decode(output_ids[0])
    return generated_text


if __name__ == '__main__':
    model, tokenizer, device = load_model()

    while True:
        try:
            prompt = input("\nYour Prompt: ")
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not prompt:
                continue
            generated_text = generate(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
                max_tokens=500,
                temperature=0.8
            )

            print("\n" + generated_text)
        except KeyboardInterrupt:
            print('Farewell')
            break