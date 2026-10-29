from pathlib import Path
import json
from src.tokenizer.tokenizer_utils import load_tokenizer
class TokenizerAnalyzer:
    def __init__(self,
        tokenizer_path):
        self.tokenizer = load_tokenizer(tokenizer_path)
    def analyze_vocabulary(self):
        vocab = self.tokenizer.get_vocab()
        return {"vocab_size": len(vocab)}

    def analyze_text(self, text):
        encoded = self.tokenizer.encode(text)
        tokens = encoded.tokens
        return {"characters": len(text), "bytes": len(text.encode("utf-8")), "tokens": len(tokens), "compression_ratio":(len(text)/max(len(tokens),1))}

    def analyze_files(self,files):
        total_chars = 0
        total_bytes = 0
        total_tokens = 0
        for file in files:
            print(f"Processing {file}")
            with open(file, "r", encoding="utf-8") as f:
                while True:
                    chunk = f.read(1024*1024)
                    if not chunk:
                        break
                    result = self.analyze_text(chunk)
                    total_chars += (result["characters"])
                    total_bytes += (result["bytes"])
                    total_tokens += (result["tokens"])
        return {"total_characters": total_chars, "total_bytes": total_bytes, "total_tokens": total_tokens, "characters_per_token":(total_chars/max(total_tokens,1)),
            "bytes_per_token": (total_bytes/max(total_tokens,1))}

    def save_report(self, report, output):
        with open(output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
