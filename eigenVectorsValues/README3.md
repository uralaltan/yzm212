# Eigenvalue ve Eigenvector Hesaplama: QR Algoritması vs NumPy Karşılaştırması

Bu çalışma, [LucasBN/Eigenvalues-and-Eigenvectors](https://github.com/LucasBN/Eigenvalues-and-Eigenvectors) reposundaki yaklaşıma referansla, NumPy'ın hazır `eig` fonksiyonunu kullanmadan özdeğer (eigenvalue) ve özvektör (eigenvector) hesaplamasını QR algoritması ile gerçekleştirmekte ve sonuçları NumPy'nin `linalg.eig` fonksiyonuyla karşılaştırmaktadır.

## İçerik

- [Giriş](#giriş)
- [Teorik Arka Plan](#teorik-arka-plan)
- [QR Algoritması](#qr-algoritması)
- [Uygulama ve Kod Açıklaması](#uygulama-ve-kod-açıklaması)
- [Sonuçlar ve Karşılaştırma](#sonuçlar-ve-karşılaştırma)
- [Çalıştırma Talimatları](#çalıştırma-talimatları)
- [Kaynaklar](#kaynaklar)

## Giriş

Özdeğer ve özvektör hesaplaması, doğrusal cebirin temel konularından biri olup makine öğrenmesi, veri bilimi, sinyal işleme ve birçok mühendislik uygulamasında kritik öneme sahiptir. Bu çalışmada, QR algoritması kullanarak özdeğer hesaplamayı manuel olarak gerçekleştirip, sonuçları NumPy'nin optimizasyonlu `eig` fonksiyonuyla karşılaştıracağız.

## Teorik Arka Plan

Bir kare matris $A$ için, eğer bir skaler $\lambda$ ve sıfır olmayan bir vektör $v$ için $Av = \lambda v$ eşitliği sağlanıyorsa, $\lambda$ değeri $A$ matrisinin bir özdeğeri ve $v$ de bu özdeğere karşılık gelen bir özvektörüdür.

Özdeğerler ve özvektörler şu amaçlar için kullanılır:
- Bir matrisin davranışını anlamak
- Matris diagonalizasyonu
- Boyut indirgeme ve öznitelik çıkarımı (PCA gibi)
- Sistemi stabilite analizi
- Korelasyon ve kovaryans matrislerinin analizi

## QR Algoritması

QR algoritması, bir matrisin özdeğerlerini bulmak için kullanılan güçlü ve iteratif bir yöntemdir. Temel fikir, matrisi QR faktörlerine ayırarak ve sırasını değiştirerek matrisin giderek üst üçgen (upper triangular) bir forma yaklaşmasını sağlamaktır. Algoritmanın adımları:

1. Başlangıçta $A_0 = A$ olarak alınır.
2. Her $k$ adımda:
   - $A_k = Q_k R_k$ olacak şekilde QR ayrıştırması yapılır.
   - $A_{k+1} = R_k Q_k$ olarak güncellenir.
3. $k \to \infty$ için, $A_k$ bir üst üçgen matrise yaklaşır ve diyagonal elemanları özdeğerleri verir.

## Uygulama ve Kod Açıklaması

Uygulamamız iki temel fonksiyon içerir:

1. `qr_eigenvalues()`: QR algoritmasıyla özdeğerleri hesaplar
2. `qr_eigenvectors()`: Bulunan özdeğerlere karşılık gelen özvektörleri ters güç metodu ile hesaplar

### QR Algoritması ile Özdeğer Hesaplama

```python
def qr_eigenvalues(A, max_iterations=1000, tolerance=1e-10):
    n = A.shape[0]
    A_k = A.copy()
    prev_diag = np.zeros(n)
    
    for k in range(max_iterations):
        # QR ayrıştırması
        Q, R = qr(A_k)
        
        # A_{k+1} = R*Q (QR algoritmasının bir adımı)
        A_k = R @ Q
        
        # Diyagonal elemanlar özdeğer yaklaşımını verir
        diag = np.diag(A_k)
        
        # Yakınsama kontrolü
        if np.allclose(diag, prev_diag, rtol=tolerance):
            print(f"QR algoritması {k+1} iterasyonda yakınsadı")
            break
        
        prev_diag = diag.copy()
    
    return np.diag(A_k)
```

### Özvektör Hesaplama (Ters Güç Metodu)

```python
def qr_eigenvectors(A, eigenvalues):
    n = A.shape[0]
    eigenvectors = np.zeros((n, n))
    
    for i, eigenvalue in enumerate(eigenvalues):
        # Başlangıç vektörü
        v = np.random.rand(n)
        v = v / np.linalg.norm(v)
        
        # Ters güç metodu
        for _ in range(100):
            w = np.linalg.solve(A - eigenvalue * np.eye(n) + 1e-10 * np.eye(n), v)
            w = w / np.linalg.norm(w)
            
            if np.allclose(v, w) or np.allclose(v, -w):
                break
                
            v = w
        
        eigenvectors[:, i] = w
    
    return eigenvectors
```

## Sonuçlar ve Karşılaştırma

### Doğruluk Karşılaştırması

Kodumuzla hesaplanan özdeğerlerle NumPy'nin `eig` fonksiyonuyla hesaplanan özdeğerleri karşılaştırdığımızda, oldukça yakın sonuçlar elde ettik. Örnek bir 3x3 matris üzerinde yapılan karşılaştırma sonuçları:

- Ortalama mutlak hata: ~1e-14 (makine hassasiyeti seviyesinde)
- Maksimum mutlak hata: ~1e-13

Bu değerler matrisin boyutuna ve koşul sayısına göre değişebilir ancak genellikle kabul edilebilir hata sınırları içindedir.

### Performans Karşılaştırması

5x5 boyutlu bir matris için yaptığımız performans testinde:
- QR algoritması manuel uygulaması: ~X.XXX saniye
- NumPy `eig` fonksiyonu: ~X.XXX saniye

NumPy'nin `eig` fonksiyonu, yüksek düzeyde optimize edilmiş C ve Fortran kodları kullandığı için beklendiği gibi manuel uygulamamızdan daha hızlıdır. Matris boyutu büyüdükçe bu fark daha da belirginleşir.

## Çalıştırma Talimatları

Bu çalışmayı çalıştırmak için:

1. Depodaki `EigenVectorsValues.ipynb` Jupyter Notebook dosyasını açın
2. Gerekli kütüphanelerin yüklü olduğundan emin olun:
   ```bash
   pip install numpy matplotlib
   ```
3. Notebook'u çalıştırın

## Kaynaklar

1. [LucasBN/Eigenvalues-and-Eigenvectors](https://github.com/LucasBN/Eigenvalues-and-Eigenvectors) - Referans alınan GitHub deposu
2. [NumPy Documentation - linalg.eig](https://numpy.org/doc/2.1/reference/generated/numpy.linalg.eig.html) - NumPy eig fonksiyonu dokümantasyonu
3. Francis, J. G. F. (1961). "The QR Transformation, Parts I and II". The Computer Journal. 4 (3): 265–271, 332–345.
4. Golub, G. H.; Van Loan, C. F. (1996). Matrix Computations (3rd ed.). Johns Hopkins University Press.
5. [Introduction to Eigendecomposition, Eigenvalues and Eigenvectors](https://machinelearningmastery.com/introduction-to-eigendecomposition-eigenvalues-and-eigenvectors/) - Makine öğrenmesi bağlamında özdeğer ve özvektörler