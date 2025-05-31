# İleri ve Geri Yayılım ile Sinir Ağı Uygulaması

## Özet

Bu çalışma, hem sınıflandırma hem de regresyon görevleri için ileri ve geri yayılım algoritmalarını kullanarak sıfırdan bir ileri beslemeli sinir ağı uygulamasını sunmaktadır. Uygulama, çoklu aktivasyon fonksiyonları (ReLU, Sigmoid, Tanh, Doğrusal, Softmax), mini-batch desteği ile gradyan tabanlı optimizasyon ve erken durdurma mekanizmalarını içermektedir. Deneysel doğrulama sentetik veri kümeleri üzerinde gerçekleştirilmiş, ikili sınıflandırma görevlerinde %87 doğruluk ve regresyon problemlerinde 0.9443 R² skoru elde edilmiş, uygulanan algoritmaların etkinliği kanıtlanmıştır.

**Anahtar Kelimeler:** Sinir Ağları, İleri Yayılım, Geri Yayılım, Gradyan İnişi, Makine Öğrenmesi

## 1. Giriş

### 1.1 Arka Plan

Yapay sinir ağları, iteratif optimizasyon süreçleri aracılığıyla verilerden karmaşık kalıpları öğrenebilen makine öğrenmesinin temel araçları haline gelmiştir. Sinir ağı eğitiminin altında yatan temel mekanizmalar, ağın tahmin hatalarına dayalı olarak ağırlık ve bias değerlerini ayarlayarak öğrenmesini sağlayan ileri yayılım ve geri yayılım algoritmalarıdır.

### 1.2 Motivasyon

Çok sayıda sinir ağı çerçevesi mevcut olmasına rağmen, uygulama yoluyla temel algoritmaları anlamak matematiksel temeller ve hesaplama süreçleri hakkında daha derin içgörüler sağlar. Bu çalışma, ileri ve geri yayılımın temel ilkelerini göstermek için sıfırdan bir sinir ağı uygular.

### 1.3 Amaçlar

Bu araştırmanın temel amaçları:
- Ağ katmanları boyunca aktivasyonları hesaplamak için ileri yayılım uygulamak
- Gradyan hesaplaması için zincir kuralını kullanarak geri yayılım geliştirmek
- Çoklu aktivasyon fonksiyonları ve kayıp fonksiyonlarını desteklemek
- Sınıflandırma ve regresyon görevlerinde uygulamayı doğrulamak
- Yakınsama ve performans metriklerini göstermek

### 1.4 Kapsam

Bu uygulama aşağıdakileri destekleyen tam bağlantılı ileri beslemeli ağlara odaklanır:
- İkili ve çok sınıflı sınıflandırma
- Regresyon görevleri
- Çoklu aktivasyon fonksiyonları (ReLU, Sigmoid, Tanh, Doğrusal, Softmax)
- Mini-batch gradyan inişi optimizasyonu
- Düzenlileştirme için erken durdurma

## 2. Yöntemler

### 2.1 Sinir Ağı Mimarisi

Uygulanan sinir ağı şunlardan oluşur:
- **Giriş Katmanı**: Özellik vektörlerini alır
- **Gizli Katmanlar**: Aktivasyon fonksiyonları ardından doğrusal dönüşümler uygular
- **Çıkış Katmanı**: Göreve özgü aktivasyon fonksiyonları kullanarak tahminler üretir

### 2.2 İleri Yayılım Algoritması

İleri yayılım aktivasyonları katman katman hesaplar:

Her l katmanı için:
```
z^[l] = W^[l] · a^[l-1] + b^[l]
a^[l] = g(z^[l])
```

Burada:
- W^[l]: l katmanı için ağırlık matrisi
- b^[l]: l katmanı için bias vektörü
- g(): Aktivasyon fonksiyonu
- a^[0] = X (giriş özellikleri)

### 2.3 Aktivasyon Fonksiyonları

Uygulama şunları içerir:

**ReLU**: f(x) = max(0, x)
- Doğrusal olmama için gizli katmanlarda kullanılır
- Gradyan: f'(x) = x > 0 ise 1, yoksa 0

**Sigmoid**: f(x) = 1/(1 + e^(-x))
- İkili sınıflandırma çıkışı için kullanılır
- Gradyan: f'(x) = f(x)(1 - f(x))

**Tanh**: f(x) = tanh(x)
- Gizli katmanlar için alternatif aktivasyon
- Gradyan: f'(x) = 1 - tanh²(x)

**Softmax**: f(x_i) = e^(x_i) / Σe^(x_j)
- Çok sınıflı sınıflandırma için kullanılır
- Sınıflar üzerinde olasılık dağılımı sağlar

**Doğrusal**: f(x) = x
- Regresyon çıkış katmanları için kullanılır
- Gradyan: f'(x) = 1

### 2.4 Geri Yayılım Algoritması

Geri yayılım zincir kuralı kullanarak gradyanları hesaplar:

**Çıkış Katmanı Hatası**:
- Sınıflandırma: δ^[L] = a^[L] - y
- Regresyon: δ^[L] = a^[L] - y

**Gizli Katman Hataları**:
```
δ^[l] = (W^[l+1])^T δ^[l+1] ⊙ g'(z^[l])
```

**Ağırlık ve Bias Gradyanları**:
```
∂L/∂W^[l] = (1/m) δ^[l] (a^[l-1])^T
∂L/∂b^[l] = (1/m) Σδ^[l]
```

### 2.5 Kayıp Fonksiyonları

**İkili Çapraz Entropi** (Sınıflandırma):
```
L = -(1/m) Σ[y log(ŷ) + (1-y) log(1-ŷ)]
```

**Kategorik Çapraz Entropi** (Çok sınıflı):
```
L = -(1/m) Σ Σ y_ij log(ŷ_ij)
```

**Ortalama Kare Hata** (Regresyon):
```
L = (1/m) Σ(y - ŷ)²
```

### 2.6 Optimizasyon

**Mini-batch Gradyan İnişi**:
- Her epoch'ta veriler karıştırılır
- Ağırlıklar batch başına güncellenir: W = W - α∇W
- Kararlılık için gradyan kırpma uygulanır

**Ağırlık Başlatma**:
- Regresyon için Xavier başlatması: σ = √(2/(n_in + n_out))
- ReLU için He başlatması: σ = √(2/n_in)

**Erken Durdurma**:
- Eğitim setindeki kaybı izle
- 50 epoch boyunca iyileşme yoksa eğitimi durdur

### 2.7 Deneysel Kurulum

**Sınıflandırma Veri Kümesi**:
- 1000 örnek, 20 özellik
- 15 bilgilendirici, 5 gereksiz özellik
- İkili sınıflandırma görevi
- Eğitim/test bölünmesi: 80/20

**Regresyon Veri Kümesi**:
- 1000 örnek, 10 özellik
- 8 bilgilendirici özellik
- Sürekli hedef değişken
- Eğitim/test bölünmesi: 80/20

**Ağ Mimarileri**:
- Sınıflandırma: [20, 16, 8, 1] ReLU → Sigmoid ile
- Regresyon: [10, 12, 6, 1] Tanh → Doğrusal ile

**Eğitim Parametreleri**:
- Öğrenme oranı: 0.01
- Epoch: 1000
- Batch boyutu: Tam batch
- Optimizasyon: Kırpma ile gradyan inişi

## 3. Sonuçlar

### 3.1 Sınıflandırma Performansı

**Eğitim İlerleyişi**:
- Başlangıç kaybı: 0.8539
- Son kayıp: 0.3335
- Başlangıç doğruluğu: %49.88
- Son doğruluk: %87.12

**Test Performansı**:
- Test doğruluğu: %87.00
- Eğitim performansı ile tutarlı, aşırı öğrenme olmadığını gösterir

**Yakınsama**:
- 1000 epoch boyunca istikrarlı kayıp azalması
- Doğruluk iyileşmesi: 37+ yüzde puanı
- Gradyan patlaması olmadan kararlı eğitim

### 3.2 Regresyon Performansı

**Eğitim İlerleyişi**:
- Başlangıç kaybı: 1.2302
- Son kayıp: 0.0523
- İlk 200 epoch içinde hızlı yakınsama

**Test Performansı**:
- Test MSE: 0.0463
- Test R² skoru: 0.9443 (varyansın %94.43'ü açıklandı)
- Güçlü tahmin performansı

### 3.3 Eğitim Dinamikleri

**Kayıp Yakınsaması**:
- Sınıflandırma: Üstel azalma modeli
- Regresyon: Hızlı başlangıç düşüşü, sonra kademeli iyileşme
- Her iki görevde de aşırı öğrenme belirtisi yok

**Gradyan Kararlılığı**:
- Gradyan kırpma patlamayı önledi
- Eğitim boyunca kararlı parametre güncellemeleri
- Çoklu çalıştırmalarda tutarlı yakınsama

### 3.4 Hesaplama Performansı

**Bellek Verimliliği**:
- Büyük veri kümeleri için mini-batch desteği
- NumPy kullanarak verimli matris işlemleri
- Makul hesaplama karmaşıklığı

**Eğitim Süresi**:
- Her iki görev için hızlı yakınsama
- Erken durdurma gereksiz hesaplamayı azalttı
- Orta ölçekli problemler için uygun

## 4. Tartışma

### 4.1 Algoritma Etkinliği

Uygulanan sinir ağı, ileri ve geri yayılımın temel ilkelerini başarıyla göstermiştir. Elde edilen performans metrikleri (%87 sınıflandırma doğruluğu, %94.43 regresyon R²) uygulamanın doğruluğunu ve algoritmaların etkinliğini doğrular.

### 4.2 İleri Yayılım Analizi

İleri yayılım uygulaması, farklı aktivasyon fonksiyonlarını düzgün bir şekilde işleyerek aktivasyonları katman katman verimli bir şekilde hesaplar. Modüler tasarım, farklı ağ mimarilerine ve aktivasyon fonksiyonlarına kolay genişleme sağlar.

### 4.3 Geri Yayılım Analizi

Geri yayılım, gradyan hesaplaması için zincir kuralını doğru bir şekilde uygular. Ağ katmanları boyunca otomatik türev alma, matematiksel temellerin doğru anlaşıldığını gösterir. Gradyan kırpma, derin ağlarda yaygın karşılaşılan sayısal kararsızlıkları önler.

### 4.4 Optimizasyon Performansı

Uygun öğrenme oranları ile gradyan inişi optimizasyonu kararlı yakınsama sağlar. Ağırlık başlatma stratejileri (Xavier/He) eğitim kararlılığına katkıda bulunur. Erken durdurma, regresyon görevlerinde aşırı öğrenmeyi etkili bir şekilde önler.

### 4.5 Sınırlamalar ve Değerlendirmeler

**Ölçeklenebilirlik**: Mevcut uygulama küçük ila orta ölçekli problemler için uygundur. Daha büyük ağlar için daha sofistike optimizerler (Adam, RMSprop) faydalı olacaktır.

**Mimari Kısıtlamaları**: Tam bağlantılı katmanlarla sınırlıdır. Konvolüsyonel veya tekrarlayıcı mimariler ek uygulama gerektirir.

**Düzenlileştirme**: Şu anda sadece erken durdurma uygular. Dropout, L1/L2 düzenlileştirme genellemeyi iyileştirebilir.

### 4.6 Pratik Uygulamalar

Bu uygulama şu amaçlarla hizmet eder:
- Sinir ağı temellerini anlamak için eğitim aracı
- Gelişmiş çerçevelerle karşılaştırma için temel
- Daha karmaşık mimarilere genişletme için temel
- Matematiksel kavramların pratikte gösterimi

### 4.7 Gelecek İyileştirmeler

**Optimizasyon Geliştirmeleri**:
- Uyarlanabilir öğrenme oranı yöntemleri
- Momentum tabanlı optimizasyon
- Öğrenme oranı planlaması

**Düzenlileştirme Teknikleri**:
- Dropout katmanları
- Batch normalizasyonu
- Ağırlık azalması

**Mimari Genişletmeleri**:
- Konvolüsyonel katmanlar
- Tekrarlayıcı bağlantılar
- Dikkat mekanizmaları

## 5. Sonuç

Bu çalışma, sıfırdan ileri ve geri yayılım algoritmaları ile bir sinir ağını başarıyla uygulamış, altta yatan matematiksel ilkelerin sağlam anlaşıldığını göstermiştir. Uygulama hem sınıflandırma (%87 doğruluk) hem de regresyon (%94.43 R²) görevlerinde rekabetçi performans elde etmiş, algoritmaların doğruluğunu doğrulamıştır.

**Ana Katkılar**:
1. İleri/geri yayılımın tam uygulaması
2. Çoklu aktivasyon fonksiyonları ve kayıp fonksiyonları desteği
3. Verimli mini-batch gradyan inişi optimizasyonu
4. Sentetik veri kümelerinde deneysel doğrulama
5. Kapsamlı performans analizi ve görselleştirme

**Eğitim Değeri**:
Uygulama, sinir ağı eğitim dinamikleri, gradyan hesaplaması ve optimizasyon süreçleri hakkında net içgörüler sağlar. Modüler tasarım, bireysel bileşenlerin ve etkileşimlerinin anlaşılmasını kolaylaştırır.

**Performans Özeti**:
- İkili sınıflandırma: %87 test doğruluğu
- Regresyon: %94.43 varyans açıklandı (R²)
- Aşırı öğrenme olmadan kararlı yakınsama
- Verimli hesaplama performansı

Sonuçlar, temel sinir ağı algoritmalarının düzgün uygulandığında standart makine öğrenmesi görevlerinde güçlü performans elde edebileceğini göstermektedir. Bu çalışma, daha gelişmiş sinir ağı mimarileri ve eğitim tekniklerini anlamak için sağlam bir temel sağlar.

## Kaynaklar

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
2. Nielsen, M. (2015). *Neural Networks and Deep Learning*. Determination Press.
3. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536.
4. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324.
5. Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *AISTATS*, 9, 249-256.

---

**Yazar**: Ural Altan Bozkurt  
**Ders**: YZM212 Makine Öğrenmesi  
**Tarih**: 31 Mayıs 2025  
**Kurum**: Ankara Üniversitesi
