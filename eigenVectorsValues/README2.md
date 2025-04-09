# NumPy `linalg.eig` Fonksiyonu ile Özdeğer ve Özvektör Hesaplama

NumPy’nın `linalg.eig` fonksiyonu, karesel (square) bir matrisin özdeğerlerini ve ona karşılık gelen sağ özvektörlerini hesaplamak için kullanılır. Aşağıda, fonksiyonun dokümantasyonuna ve kaynak kodlarına dayanarak hesaplamaların nasıl gerçekleştiği açıklanmaktadır.

---

## 1. Dokümantasyon İncelemesi

- **Tanım ve Kullanım:**
  - Fonksiyon çağrısı: `w, v = np.linalg.eig(a)`
  - Parametre:
    - `a`: Karesel (square) matris. Girdi matrisinin türü ve boyutuna göre, hesaplama gerçek veya karmaşık sayılar üzerinde yapılır.
  - Dönüş Değerleri:
    - `w`: Matrisin özdeğerlerini içeren 1-boyutlu dizi.
    - `v`: Her özdeğere karşılık gelen sağ özvektörleri, sütun vektörler olarak içeren matris. Yani, her `v[:, i]` vektörü, `a @ v[:, i] == w[i] * v[:, i]` eşitliğini sağlar.

- **Belirtilen Özellikler:**
  - Fonksiyon, genellikle karmaşık (complex) hesaplamaları da destekleyecek şekilde tasarlanmıştır.
  - Belirli LAPACK (Linear Algebra PACKage) rutinleri kullanılarak, yüksek performanslı ve kararlı nümerik hesaplamalar gerçekleştirilir.
  - Özellikle, dokümantasyonda fonksiyonun kare matrisler için geçerli olduğuna dikkat çekilir; çünkü yalnızca kare matrislerde tanımlı olan özdeğer ayrışımı mümkündür.

Detaylı dokümantasyon bilgisine [buradan](https://numpy.org/doc/2.1/reference/generated/numpy.linalg.eig.html) erişilebilir. :contentReference[oaicite:0]{index=0}

---

## 2. Kaynak Kod İncelemesi

- **NumPy Linalg Modül Yapısı:**
  - NumPy’nın lineer cebir fonksiyonları, `numpy/linalg` dizini altında yer alır. `eig` fonksiyonu da bu dizin altında, Python arayüzlerinin ve düşük seviyeli LAPACK çağrılarının (örneğin `geev` gibi) bulunduğu modüller üzerinden implemente edilmiştir.
  - Kaynak kodlar, LAPACK kütüphanesindeki optimize edilmiş algoritmalara dayanmaktadır. Bu sayede, QR algoritması gibi nümerik yöntemlerle özdeğer ve özvektör hesaplamaları gerçekleştirilir.

- **Fonksiyonun İşleyişi:**
  - `np.linalg.eig` çağrısı, temel olarak ilgili LAPACK rutinini (`*_geev`) sarmalar.
  - Girdi matrisinin veri tipi ve boyutuna bağlı olarak, uygun hesaplama (gerçek veya kompleks) gerçekleştirilir.
  - Kaynak kod, girdi doğrulaması, bellek tahsisi ve sonrasında LAPACK çağrısının yapılması adımlarını barındırır.
  - İşlem tamamlandıktan sonra, LAPACK çıktısı, Python’da kullanılabilir NumPy dizilerine dönüştürülür. Bu dönüşüm sırasında hem özdeğerler hem de her özdeğere karşılık gelen özvektörler elde edilir.
  
Kaynak kodlarını detaylı incelemek için [NumPy Github Linalg dizini](https://github.com/numpy/numpy/tree/main/numpy/linalg) üzerinden erişim sağlanabilir.

---

## 3. Hesaplama Sürecinin Teknik Detayları

1. **Girdi Doğrulaması:**
   - Fonksiyon, verilen matrisin kare olup olmadığını kontrol eder. Kare olmayan matrisler için hesaplama yapılamaz ve hata mesajı döner.
   - Matrisin veri tipine bağlı olarak (float, double, complex) gerekli dönüşümler uygulanır.

2. **LAPACK Çağrısı:**
   - Uygun LAPACK fonksiyonu (`geev` veya ilgili varyantı) seçilerek matrisin özdeğer ayrışımı hesaplanır.
   - LAPACK, QR algoritması gibi nümerik yöntemler kullanarak, matrisin özdeğerlerini ve sağ özvektörlerini güvenilir bir şekilde hesaplar.
   - Bu algoritmalar, genellikle iteratif süreçler içerir ve konverjans (yakınsama) kontrolü yaparlar.

3. **Çıktıların Dönüştürülmesi:**
   - LAPACK tarafından hesaplanan sonuçlar, NumPy dizilerine dönüştürülerek kullanıcıya sunulur.
   - `w` dizisi, özdeğerleri; `v` ise özvektörleri içerir. Her özvektör, sırasıyla matrisin sütunları olarak temsil edilir.
   - Kullanıcı, bu diziler üzerinden ileri seviye hesaplamalar veya analizler yapabilir.

---

## 4. Özet

- NumPy `linalg.eig` fonksiyonu, kare bir matrisin özdeğerleri ve özvektörlerini hesaplamak için geliştirilmiştir.
- Fonksiyon, girdi doğrulaması, uygun LAPACK çağrısı ve sonuçların Python ortamına uygun şekilde dönüştürülmesi adımlarından oluşmaktadır.
- Dokümantasyon, fonksiyonun kullanım şeklini ve dönüş değerlerini açıkça belirtirken, kaynak kodlar ise LAPACK gibi optimize kütüphanelerin sağladığı nümerik stabiliteye dayanmaktadır.
- Bu yapı sayesinde, hem gerçek hem de kompleks matrisler üzerinde güvenilir ve yüksek performanslı hesaplamalar yapılabilmektedir.

---

## Kaynaklar

- [NumPy `linalg.eig` Dokümantasyonu](https://numpy.org/doc/2.1/reference/generated/numpy.linalg.eig.html)
- [NumPy Linalg Kaynak Kodları (GitHub)](https://github.com/numpy/numpy/tree/main/numpy/linalg)
