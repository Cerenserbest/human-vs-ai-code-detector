import requests
import base64
import json
import time
from requests.exceptions import ConnectionError, Timeout, ChunkedEncodingError
import os # Yeni: Dosya işlemlerini kolaylaştırmak için

# !!! BURAYA KENDİ KOPYALADIĞINIZ PAT ANAHTARINIZI YAZIN !!!
# Belirtecinizi tırnak işaretleri ("") içine almayı UNUTMAYIN!
GITHUB_TOKEN = "" 

# --- Ayarlar ---
CODE_LANGUAGE = ".py" 
MAX_SNIPPETS_PER_REPO = 1500 # Her repodan maksimum 1500 dosya çekmeye çalışılacak
MIN_CODE_LENGTH = 50 # Kod parçacığının minimum karakter uzunluğu (50 yaparak filtreyi gevşettik)

# Genişletilmiş ve Çeşitlendirilmiş Python REPOSITORIES Listesi (6000+ hedefine yönelik)
REPOSITORIES = [
    {"owner": "pallets", "repo": "flask"},         
    {"owner": "psf", "repo": "requests"},          
    {"owner": "numpy", "repo": "numpy"},           
    {"owner": "scipy", "repo": "scipy"},           
    {"owner": "django", "repo": "django"},         
    {"owner": "pandas-dev", "repo": "pandas"},     
    {"owner": "celery", "repo": "celery"},          
    {"owner": "tornadoweb", "repo": "tornado"},     
    {"owner": "python", "repo": "cpython"},        
    {"owner": "scikit-learn", "repo": "scikit-learn"},
    {"owner": "pypa", "repo": "pip"},              
    {"owner": "encode", "repo": "httpx"},          
    {"owner": "scrapy", "repo": "scrapy"},         
    {"owner": "tiangolo", "repo": "fastapi"},      
    {"owner": "localstack", "repo": "localstack"}, # YENİ REPO
    {"owner": "encode", "repo": "uvicorn"},        # YENİ REPO
    {"owner": "commaai", "repo": "openpilot"},     # YENİ REPO
    {"owner": "pallets", "repo": "click"},         # YENİ REPO
    {"owner": "marshmallow-code", "repo": "marshmallow"}, # YENİ REPO
    {"owner": "google", "repo": "python-fire"},    # YENİ REPO
    {"owner": "jazzband", "repo": "django-rest-framework"}, # YENİ REPO
    {"owner": "googleapis", "repo": "google-cloud-python"}, # YENİ REPO
    {"owner": "kennethreitz", "repo": "requests"},
]
# --- ---

headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

def get_repo_files(owner, repo):
    """Belirtilen repodaki tüm dosyaların yollarını (path) çeker."""
    print(f"\n--- {owner}/{repo} repodaki dosyalar listeleniyor... ---")
    
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1"
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"HATA: {owner}/{repo} içeriği çekilemedi. Hata kodu: {response.status_code}")
        return []
    
    tree = response.json().get('tree', [])
    
    code_files = [item for item in tree if item['path'].endswith(CODE_LANGUAGE) and item['type'] == 'blob']
    
    print(f"-> Toplam {len(code_files)} adet {CODE_LANGUAGE} dosyası bulundu.")
    return code_files

def get_file_content(owner, repo, file_path):
    """Belirtilen dosya yolunun içeriğini çeker ve Base64'ten çözer."""
    content_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    
    # Bağlantı hataları için 3 kez tekrar deneme mekanizması
    for attempt in range(3):
        try:
            content_response = requests.get(content_url, headers=headers, timeout=15) # Zaman aşımı arttırıldı
            
            if content_response.status_code == 200:
                content_data = content_response.json()
                encoded_content = content_data.get('content')
                
                if encoded_content:
                    try:
                        decoded_bytes = base64.b64decode(encoded_content.replace('\n', ''))
                        return decoded_bytes.decode('utf-8')
                    except:
                        return None 
            
            elif content_response.status_code == 404:
                return None 
            
            # Hata kodu gelirse tekrar dene
            print(f"[{owner}/{repo}] Hata kodu: {content_response.status_code}. Tekrar deniyor...")
            time.sleep(2) 
            
        except (ConnectionError, Timeout, ChunkedEncodingError):
    # ...
            # Bağlantı hatalarında 3 saniye bekle ve tekrar dene
            print(f"[{owner}/{repo}] Bağlantı hatası. {attempt + 1}. deneme...")
            time.sleep(3)
            
        except Exception:
            return None
            
    return None 

def get_current_snippet_count():
    """Mevcut toplanmış kod parçacığı sayısını bulur."""
    if not os.path.exists("human_code_snippets.jsonl"):
        return 0
    try:
        with open("human_code_snippets.jsonl", "r", encoding="utf-8") as f:
            return sum(1 for line in f)
    except Exception:
        return 0

# --- Ana Çalışma Alanı ---

# Mevcut toplanmış kod sayısını al (Veri kaybını önlemek için)
toplanan_sayi = get_current_snippet_count()
print(f"Önceki oturumlardan {toplanan_sayi} adet kod bulundu. Buradan devam ediliyor.")

# 'a' modu (append/ekle) ile açılır, dosya silinmez.
with open("human_code_snippets.jsonl", "a", encoding="utf-8") as f:
    
    for repo_info in REPOSITORIES:
        owner = repo_info['owner']
        repo = repo_info['repo']
        
        dosya_listesi = get_repo_files(owner, repo)
        
        cekilen_sayi = 0
        for dosya in dosya_listesi:
            if cekilen_sayi >= MAX_SNIPPETS_PER_REPO:
                break
                
            path = dosya['path']
            kod_icerigi = get_file_content(owner, repo, path)
            
            # Filtre: Minimum uzunluk ve test dosyası kontrolü
            if kod_icerigi and len(kod_icerigi.strip()) > MIN_CODE_LENGTH and not "test" in path.lower(): 
                toplanan_sayi += 1
                cekilen_sayi += 1
                
                # Kod parçacığını (tek bir satır olarak) dosyaya ekle
                kod = {
                    "code_id": f"HUMAN_{toplanan_sayi}",
                    "code_snippet": kod_icerigi, # Parçacık Ayırma (Snippetization) adımını siz yapmalısınız!
                    "source_type": "human",
                    "repo_path": f"{owner}/{repo}/{path}"
                }
                f.write(json.dumps(kod, ensure_ascii=False) + "\n")
                
                if toplanan_sayi % 100 == 0:
                     print(f"   -> TOPLAM İLERLEME: {toplanan_sayi} kod parçacığı (Repo: {repo})")
            
            # API kısıtlamalarına takılmamak için kısa bir bekleme
            time.sleep(0.05) 

print(f"\nİşlem Tamamlandı veya Durduruldu. Toplam {toplanan_sayi} adet kod 'human_code_snippets.jsonl' dosyasına eklendi.")