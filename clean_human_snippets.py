import json
import hashlib

# ==============================
# AYARLAR
# ==============================
INPUT_FILE = "human_code_snippets.jsonl"      # Girdi dosyanın adı
OUTPUT_FILE = "human_snippets_clean.jsonl"    # Çıktı dosyasının adı
MIN_LINES = 5     # Minimum satır sayısı
MAX_LINES = 200   # Maksimum satır sayısı


def clean_code(code: str) -> str:
    """
    Kodun içinden:
    - Tamamen yorum olan satırları
    - Üç tırnaklı (\"\"\" ... \"\"\") docstring bloklarını
    - Baştaki/sondaki boş satırları
    temizler, çoklu boş satırları azaltır.
    """
    lines = code.splitlines()
    cleaned_lines = []
    in_docstring = False
    doc_delim = None

    for line in lines:
        stripped = line.strip()

        # Docstring başlangıcı
        if not in_docstring and (stripped.startswith('"""') or stripped.startswith("'''")):
            # Tek satırlık docstring ise komple atla
            if stripped.count('"""') == 2 or stripped.count("'''") == 2:
                continue
            in_docstring = True
            doc_delim = '"""' if stripped.startswith('"""') else "'''"
            continue

        # Docstring içindeysek
        if in_docstring:
            if doc_delim in stripped:
                in_docstring = False
                doc_delim = None
            continue

        # Tamamen boş satır
        if stripped == "":
            cleaned_lines.append("")
            continue

        # Tamamen yorum satırı (# ile başlayan)
        if stripped.startswith("#"):
            continue

        # Normal kod satırı
        cleaned_lines.append(line.rstrip())

    # Çoklu boş satırları en fazla 2'ye düşür
    final_lines = []
    blank_count = 0
    for line in cleaned_lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 2:
                final_lines.append("")
        else:
            blank_count = 0
            final_lines.append(line)

    # Baştaki ve sondaki boş satırları kırp
    while final_lines and final_lines[0].strip() == "":
        final_lines.pop(0)
    while final_lines and final_lines[-1].strip() == "":
        final_lines.pop()

    return "\n".join(final_lines)


def main():
    total = 0
    kept = 0
    seen_hashes = set()

    # Girdi dosyasını aç
    try:
        fin = open(INPUT_FILE, "r", encoding="utf-8")
    except FileNotFoundError:
        print(f"[HATA] Girdi dosyası bulunamadı: {INPUT_FILE}")
        print("Bu dosyayı, bu python dosyasıyla aynı klasöre koyduğundan emin ol.")
        return

    # Çıktı dosyasını aç
    fout = open(OUTPUT_FILE, "w", encoding="utf-8")

    for line in fin:
        total += 1
        obj = json.loads(line)
        code = obj.get("code_snippet", "")
        cleaned = clean_code(code)

        if not cleaned:
            continue

        line_count = cleaned.count("\n") + 1

        # Satır sayısı filtresi
        if line_count < MIN_LINES or line_count > MAX_LINES:
            continue

        # Duplicate engelleme (aynı kodları at)
        normalized = "\n".join(l.rstrip() for l in cleaned.splitlines())
        h = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        # Yeni format
        new_obj = {
            "id": obj.get("code_id"),
            "text": cleaned,
            "label": "human",
            "lang": "python"
        }

        fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")
        kept += 1

    fin.close()
    fout.close()

    print("======================================")
    print(f"Toplam okunan snippet sayısı : {total}")
    print(f"Filtrelenip kaydedilen sayı : {kept}")
    print(f"Çıktı dosyası               : {OUTPUT_FILE}")
    print("======================================")


if __name__ == "__main__":
    main()
