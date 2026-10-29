from pathlib import Path
import re
from src.tokenizer.tokenizer_analysis import TokenizerAnalyzer

def natural_sort_key(path):
    return [int(text) if text.isdigit() else text for text in re.split(r"(\d+)", path.name)]

def main():
    tokenizer_path = ("tokenizer/tokenizer.json")
    dataset_directory = Path("data/raw")
    files = sorted(dataset_directory.glob("*.txt"), key=natural_sort_key)
    print(f"Total files found: {len(files)}")
    if not files:
        raise FileNotFoundError("No text files found in data/raw")
    print("Files to analyze:")
    for file in files:
        print(file)
    analyzer = TokenizerAnalyzer(tokenizer_path)
    report = {}
    report["vocabulary"] = (analyzer.analyze_vocabulary())
    report["dataset"] = (analyzer.analyze_files(files))
    analyzer.save_report(report, "tokenizer/tokenizer_analysis.json")
    print("\nTokenizer analysis completed.")
    print(report)

if __name__ == "__main__":
    main()
