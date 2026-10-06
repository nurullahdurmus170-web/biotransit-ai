# BioTransit AI

Sevkiyat sırasında kaydedilen sıcaklık ve mekanik maruziyet (IMU) verisinden, taşınan protein/enzim numunesinin gönderisini **LOW / MEDIUM / HIGH** riskli olarak sınıflandıran bir karar destek prototipi.

> **Durum: kavram kanıtı.** Bu depodaki tüm sonuçlar **sentetik** veri üzerindedir (literatür bilgisiyle üretilmiş 600 gönderi, 7 senaryo). Gerçek numune deneyi (HRP aktivite testi) henüz yapılmamıştır. Aşağıdaki metrikler gerçek dünya başarımı değildir.

## Problem
Numuneler sevkiyatta sıcaklık ve mekanik şoka maruz kalır. Alıcı laboratuvar çoğu zaman numunenin hâlâ kullanılabilir olup olmadığını bilemez ya da yalnızca sıcaklık eşiğine bakar. Amaç, sıcaklık ve mekanik maruziyeti birlikte değerlendirip açıklanabilir bir risk sınıfı üretmektir.

## Veri
- 600 sentetik gönderi, 7 senaryo: `stable_cold_chain`, `short_temp_excursion`, `prolonged_temp_excursion`, `high_vibration`, `drone_like_vibration`, `shock_heavy`, `combined_stress`
- 15 özellik (sıcaklık: ortalama/maks./sapma, 8 °C üstü süre ve alan; mekanik: ivme RMS, titreşim dozu, şok sayıları, baskın frekans; süre)
- Sınıf dağılımı dengesiz: LOW 438, MEDIUM 102, HIGH 60 (LOW ≈ %73)
- Gerçek veri planı: tek model enzim (yaban turpu peroksidazı, HRP), kontrollü sıcaklık/titreşim maruziyeti ve kolorimetrik aktivite testi, yaklaşık 100 numune. **Henüz yapılmadı.**

## Sonuçlar (sentetik veri, sınıflandırma)

**Rastgele %20 holdout**

| Model | Accuracy | Balanced acc. | Macro-F1 |
|---|---|---|---|
| Random Forest | 0.907 | 0.899 | 0.870 |
| Çok değişkenli lojistik regresyon | 0.887 | 0.889 | 0.845 |
| Basit eşik | 0.407 | 0.557 | 0.377 |
| Dummy (en sık sınıf) | 0.733 | 0.333 | 0.282 |

**5-katlı çapraz doğrulama (ortalama ± sd)**

| Model | Accuracy | Balanced acc. | Macro-F1 |
|---|---|---|---|
| Çok değişkenli lojistik regresyon | 0.927 ± 0.006 | 0.910 ± 0.022 | 0.882 ± 0.016 |
| Random Forest | 0.915 ± 0.025 | 0.849 ± 0.074 | 0.845 ± 0.060 |
| Karar ağacı (derinlik 3) | 0.813 ± 0.033 | 0.766 ± 0.035 | 0.730 ± 0.038 |
| Dummy (en sık sınıf) | 0.730 ± 0.004 | 0.333 | 0.281 ± 0.001 |

Holdout'ta Random Forest, çapraz doğrulamada lojistik regresyon önde; fark, RF'nin fold'lar arası dalgalanması (sd 0.060) düşünüldüğünde anlamlı sayılmamalı. Bu veri kümesinde basit bir doğrusal model de rekabetçi.

**Leave-One-Scenario-Out (bir senaryoyu tamamen dışarıda bırakma; Random Forest, 350 ağaç, derinlik 8)**

| Dışarıda bırakılan senaryo | n | Test setindeki sınıflar | Accuracy |
|---|---|---|---|
| combined_stress | 90 | HIGH, LOW, MEDIUM | **0.078** |
| prolonged_temp_excursion | 80 | HIGH, LOW, MEDIUM | **0.313** |
| high_vibration | 80 | LOW, MEDIUM | 0.763 |
| short_temp_excursion | 90 | LOW | 0.911 |
| stable_cold_chain | 120 | LOW | 1.000 |
| drone_like_vibration | 60 | LOW | 1.000 |
| shock_heavy | 80 | LOW | 1.000 |

Bu tablo iyi haber değil: Random Forest, eğitimde görmediği **combined_stress** ve **prolonged_temp_excursion** senaryolarında büyük ölçüde başarısız oluyor. HIGH sınıfı test setinde yalnızca bu iki senaryoda var; bu iki senaryodan biri dışarıda kalınca eğitimde HIGH örnekleri azalıyor (nedeni bu veriyle kanıtlanmadı, olası açıklama bu). Sonuç: yüksek CV skoru, modelin yeni koşullara genelleştiği anlamına gelmiyor.

> `results/figures/FIG1_scenario_holdout_collapse.png` grafiği macro-F1 gösteriyor. Tek sınıflı test setlerinde (LOW) macro-F1 yapısal olarak en fazla 0.333 olabilir, bu yüzden bu barlar model başarısızlığı değildir. Yorumlamak için yukarıdaki accuracy tablosunu kullanın.

**Özellik önemi (permutation importance)**

Mekanik özellikler önde (titreşim dozu 0.053, 2 g üstü titreşim süresi 0.044, maks. ivme RMS 0.042), sıcaklık özelliklerinin çoğu daha düşük ve belirsizlik aralıkları geniş (`results/figures/FIG2_permutation_importance.png`). Bu sıralama sentetik verinin üretim mantığını yansıtıyor olabilir; gerçek enzim davranışı hakkında bir bulgu değildir.

## Sınırlılıklar
- Sentetik veri; gerçek numune doğrulaması yok.
- Genelleme testi zayıf (yukarıya bakın).
- Tek enzim (HRP) ile sınırlıdır; diğer proteinlere genellenebilirlik test edilmemiştir.
- Sonuçlar risk sınıflandırmasını kapsar; donanım entegrasyonu yapılmamıştır (planlanan: düşük-g IMU MPU6050 + ±200G ivmeölçer).

## Çalıştırma

**Demo (Streamlit)**
```bash
pip install -r requirements.txt
cd demo
streamlit run demo_app.py
```
Uygulama dosyaları çalışma klasöründen okur; bu yüzden `demo/` içinden çalıştırın.
- **Örnek Gönderi Seç** sekmesi, bu depoda bulunan `shipments_with_features.csv` ile çalışır. Ham sensör zaman serisi dosyası (`BioTransit_sensor_timeseries.csv`) bu depoda **yok**; dosya olmadığında sekme çökmez, yalnızca sıcaklık/şok grafiğini atlar ve risk sınıflandırmasını normal şekilde gösterir (bu davranış test edilmiştir: `streamlit.testing.v1.AppTest` ile hem dosya yokken hem de LOW/HIGH örnekleriyle uçtan uca doğrulanmıştır).
- **Kendi CSV'ni Yükle** sekmesi `data/sample/` içindeki üç örnek dosyayla (LOW / MEDIUM / HIGH) tam olarak çalışır; bu akış ham sensör dosyasına ihtiyaç duymaz. Beklenen kolonlar: `shipment_id, elapsed_min, temperature_c, accel_rms_g, vibration_freq_hz, shock_peak_g`.

Kısa bir ekran kaydı: [`docs/biotransit_demo.mp4`](docs/biotransit_demo.mp4)

**Notebook**
`notebooks/BioTransit_CDR_MVP_Colab_v.ipynb`, çalışma klasöründe `BioTransit_sensor_timeseries.csv` ve `BioTransit_literature_informed_shipments.csv` dosyalarını bekler. **Bu iki ham veri dosyası depoda yer almıyor**, bu yüzden notebook sıfırdan yeniden çalıştırılamaz; notebook'un kendi hücre çıktıları (gömülü) `results/` klasöründeki CSV'lerle birebir tutarlıdır ve statik olarak incelenebilir. Notebook öznitelik çıkarımını, baseline karşılaştırmasını, 5-katlı CV'yi ve senaryo bazlı genelleme testini içerir.

**Not:** `rf_model.joblib` demo için eğitilmiş modeldir. Bu depodaki `requirements.txt`, modelin kaydedildiği scikit-learn sürümüne sabitlenmiştir (joblib/pickle dosyaları sürüm ve güvenlik açısından yalnızca güvendiğiniz kaynaktan yüklenmelidir). Demodaki örnek gönderiler (BT0001, BT0214, BT0220) veri setindendir; model bu örnekler üzerinde başarılı görünse de bu, genelleme kanıtı değildir.

## Yapı
```
results/                 Metrik CSV'leri, teknik özet, şekiller
notebooks/               Öznitelik çıkarımı ve model karşılaştırma (Colab)
demo/                    Streamlit uygulaması, model ve veri
data/sample/             Tek gönderilik ham sensör örnekleri (LOW / MEDIUM / HIGH)
docs/                    Ekran kaydı
```

## Ekip
Nurullah Durmuş, Elif Oflaz, Yusuf Emre Arı
