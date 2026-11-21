# Fluo-N2DH-GOWT1 hücre segmentasyonu ve sayımı

Üsküdar Üniversitesi Bilgisayar Mühendisliği yüksek lisans Sayısal Görüntü İşleme ara sınav projesi. Klasik yöntemler floresan mikroskopi karelerinde çekirdekleri ayırır ve bireysel etiketleri sayar. Sabit başlangıç yöntemi ham gri seviye → Otsu → 8 bağlantılı bileşen sayımıdır.

Sekans 01'den 10 geliştirme ve 5 doğrulama karesi, Sekans 02'den 5'er karelik 3 test bloğu seçilir. Tohum **29033** olarak kaydedilmiştir. Birbirinden ayrılması gereken gruplar ve bloklar arasında en az beş kullanılmayan kare bırakılır. Referans sayım her TRA TIFF dosyasındaki benzersiz pozitif etiketlerin sayısıdır.

Doğrulamada seçilen **E24**, ham görüntüyü 5×5 Gaussian filtreyle yumuşatır. 121×121 Gaussian komşuluğunda, OpenCV'nin C = −2 parametresi ile eşikler. 5×5 eliptik elemanla açma ve kapama, delik doldurma ve 250 pikselin altındaki bileşenlerin çıkarılmasıyla sayımı elde eder. Üç adayın doğrulama sonuçları `validation_results.csv` dosyasındadır.

## Veri ve kurulum

1. [CTC eğitim arşivini](https://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip) indirin. Yaklaşık 59 MB olan ZIP dosyasını bu deponun `data/` klasörüne koyun. Kod ZIP'i doğrudan okur. ZIP'i önceden açtıysanız `--data` seçeneğine `01`, `02`, `01_GT` ve `02_GT` klasörlerinin bulunduğu dizini verin. [Resmî veri seti sayfası](https://celltrackingchallenge.net/2d-datasets/)
2. Depo klasöründe Python 3.12 ile bir sanal ortam oluşturup bağımlılıkları yükleyin. Aşağıdaki örnekler Windows PowerShell içindir.

```powershell
py -3.12 -m venv .venv
$python = '.\.venv\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
& $python -m pip install -r requirements.txt
```

Python 3.12 varsayılan kurulumunuzsa, `py -3.12` yerine `python` yazabilirsiniz. Jupyter kullanmak için aynı ortama `notebook` paketini yükleyip defteri açın.

```powershell
& $python -m pip install notebook
& $python -m notebook main.ipynb
```

Google Colab için `main.ipynb` dosyasını açın, GitHub deponuzu oturuma klonlayın ve çalışma dizinini depo köküne ayarlayın. İlk kurulum hücresinden sonra veri ve deney hücrelerini sırayla çalıştırın. Depo özelse erişim yetkisi olan bir oturum kullanın veya kodun yerel kopyasını dosya olarak yükleyin.

## Geliştirme ve kilitli tahminleri yeniden üretme

Bu örnek çalışmanın kayıtlı dosyalarına dokunmadan sonuçları ayrı bir `reproduction` klasöründe üretir. Python modülü depo kökünden çalıştırılır. `lock` **Sekans 02 görüntülerini** kullanır ve Sekans 02 anotasyonlarını açmaz. İşlem süreleri donanıma ve sistem yüküne göre değişebilir.

```powershell
$data = '.\data\Fluo-N2DH-GOWT1.zip'
$scratch = '.\reproduction'
& $python -m cellcount.run select  --data $data --out $scratch --seed 29033
& $python -m cellcount.run develop --data $data --out $scratch
& $python -m cellcount.run lock    --data $data --out $scratch
& $python -m cellcount.figures --data $data --results $scratch --output '..\gowt1-private-figures'
```

Başka bir tekrar için farklı bir `--out` klasörü kullanın. Defter, varsayılan olarak her çalıştırmada yeni bir geçici klasör açar. Özel görselleri depo dışında ve yalnızca ders teslimi için saklayın.

## GitHub kilidi ve bağımsız değerlendirme

Teslim edilen protokolün değerlendirme adımı, deponun kökündeki kare seçimini, pipeline parametrelerini, tahminleri ve bağlantılı kodu GitHub'da önceden gönderilmiş kilit commit'i ile karşılaştırır. Kilitli dosyaları yeniden üretim klasöründeki kopyalarla değiştirmeyin.

Bağımsız değerlendirmeyi yeniden çalıştırmak için GitHub deposunu `git clone` ile klonlayın. ZIP kopyası geliştirme deneylerini ve kilitli tahminleri yeniden üretmek için yeterlidir. GitHub doğrulaması depo geçmişi üzerinden yürür.

```powershell
$lock = git log -1 --format=%H -- test_predictions_locked.csv
& $python -m cellcount.run evaluate --data $data --out . --lock-commit $lock --branch main
& $python -m cellcount.figures --data $data --results . --output '..\gowt1-private-figures' --lock-commit $lock --branch main
```

Kilit commit’i, `final_pipeline.json` ve `test_predictions_locked.csv` dosyalarını ilk kez kaydeder. Bu commit GitHub'a gönderilip doğrulandıktan sonra referanslar açılır. Bağımsız metrikler sonraki bir commit'te kaydedilir. Araç, tahmin commit'inin uzak dalın geçmişinde bulunduğunu da doğrulayabilir. Özel bir GitHub deposu kullanıyorsanız öğretim elemanına erişim verin.

## Dosyalar

| Dosya | İçerik |
| --- | --- |
| `main.ipynb` | Kare seçimi, deneyler, doğrulama, kilitli tahmin ve görsel inceleme |
| `cellcount/data.py` | ZIP veya klasörden TIFF okuma ve deterministik seçim |
| `cellcount/pipeline.py` | Baseline, deney ayarları ve görüntü işleme |
| `cellcount/run.py` | Aşamaların yürütülmesi, metrik ve kilit doğrulama |
| `cellcount/figures.py` | Ders için özel karşılaştırma görselleri |
| `selected_frames.csv` | Grup, sekans, blok, kare, dosya adı ve tohum |
| `experiments.csv` | B00 ve 30 deneyin ayarları ile geliştirme metrikleri |
| `development_frame_results.csv` | Geliştirme aşamasında kare bazında sayımlar |
| `validation_results.csv` | Üç adayın sıralaması ve doğrulama metrikleri |
| `validation_frame_results.csv`, `validation_baseline.csv` | Ayrıntılı doğrulama sayımları ve baseline özeti |
| `intensity_statistics.csv` | Geliştirme ve doğrulama karelerinin boyut, tip, yoğunluk ve referans bilgileri |
| `final_pipeline.json` | Sabit ayarlar, doğrulama özeti ve SHA-256 özetleri |
| `test_predictions_locked.csv` | Referanslar okunmadan kaydedilen 15 tahmin çifti |
| `test_results.csv`, `test_metrics.csv` | Bağımsız değerlendirme yapıldığında üretilen sayımlar, hatalar ve özet metrikler |

Özel görsellerde yoğunluk ve histogram panelleri, eşikleme karşılaştırmaları, nihai pipeline adımları, örnek kareler ve zamansal sayım grafiği bulunur. GitHub'a veri ZIP'ini, TIFF dosyalarını, örnek kare görsellerini ve gömülü defter çıktılarını yüklemeyin. Defteri kaydederken bütün hücre çıktılarını temizleyin.

Veri kaynağı Cell Tracking Challenge ve Dr. E. Bártová'dır. [Maška ve diğerleri, Nature Methods, 2023](https://doi.org/10.1038/s41592-023-01879-y) · [CTC veri kullanım koşulları](https://celltrackingchallenge.net/datasets/)
