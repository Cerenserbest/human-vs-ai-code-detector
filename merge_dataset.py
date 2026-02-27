import json

HUMAN_FILES = [
    "human_snippets_clean.jsonl",
    "human_extra_generated.jsonl",
]

AI_FILE = "ai_snippets_generated.jsonl"

OUTPUT_FILE = "dataset_final.jsonl"

def load_jsonl(path, label):
    data = []

    # Önce dosya komple JSON array mi diye deneyelim
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Eğer köşeli parantezle başlıyorsa → JSON array formatı
    if content.startswith("["):
        try:
            arr = json.loads(content)
            for obj in arr:
                obj["label"] = label
                data.append(obj)
            return data
        except:
            pass  # JSON array değilse normal satır-satır okumaya devam edeceğiz

    # --- Normal JSONL okuma ---
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                obj["label"] = label
                data.append(obj)
            except:
                print(f"Bozuk satır atlandı: {line[:40]}")
                continue

    return data


def save_jsonl(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def main():
    print("Human veri okunuyor...")
    humans = []
    for file in HUMAN_FILES:
        humans.extend(load_jsonl(file, 0))

    print("AI veri okunuyor...")
    ais = load_jsonl(AI_FILE, 1)

    print(f"Human: {len(humans)} adet")
    print(f"AI: {len(ais)} adet")

    final = humans + ais
    print(f"Toplam final dataset: {len(final)} satır")

    save_jsonl(OUTPUT_FILE, final)

    print("======================================")
    print("Final dataset oluşturuldu!")
    print("======================================")


if __name__ == "__main__":
    main()
