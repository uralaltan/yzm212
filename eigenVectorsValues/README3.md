# Eigenvalue Hesaplama: Manual Yöntem (QR Algoritması) & NumPy `linalg.eig` Karşılaştırması

Bu çalışma, [LucasBN/Eigenvalues-and-Eigenvectors](https://github.com/LucasBN/Eigenvalues-and-Eigenvectors) reposundaki yaklaşıma referansla, NumPy’nın hazır `eig` fonksiyonunu kullanmadan eigenvalue hesaplamasını QR algoritması ile uygulamaktadır. Ayrıca, aynı matriste NumPy’nin `np.linalg.eig` fonksiyonuyla elde edilen sonuçlar karşılaştırılarak değerlendirilmektedir.

---

## İçerik

- [Giriş](#giriş)
- [QR Algoritması ile Eigenvalue Hesaplama](#qr-algoritması-ile-eigenvalue-hesaplama)
- [NumPy `linalg.eig` Fonksiyonu ile Hesaplama](#numpy-linalg-eig-fonksiyonu-ile-hesaplama)
- [Sonuçların Karşılaştırılması](#sonuçların-karşılaştırılması)
- [Çalıştırma Talimatları](#çalıştırma-talimatları)
- [Kaynaklar](#kaynaklar)

---

## Giriş

Eigenvalue hesaplaması, kare matrislerdeki temel dinamikleri anlamak için önemlidir. Bu çalışma, önce QR algoritması temelinde yapılan iteratif bir yöntemle eigenvalue’leri elde eder, ardından NumPy’nın optimize edilmiş `np.linalg.eig` fonksiyonuyla hesaplanan sonuçlarla karşılaştırır.

---

## QR Algoritması ile Eigenvalue Hesaplama

QR algoritması, kare bir matrisi Q (orthogonal) ve R (upper-triangular) bileşenlerine ayırarak yinelemeli olarak şu şekilde kullanılır:
1. Başlangıçta \( A_0 = A \) alınır.
2. Her adımda \( A_k = Q_k R_k \) olacak şekilde QR ayrıştırması yapılır.
3. Sonrasında \( A_{k+1} = R_k Q_k \) olarak güncellenir.
4. Yeterli iterasyon sonrası \( A_k \) yaklaşık olarak üst üçgensel (upper-triangular) hale gelir; bu durumda, matristeki çapraz elemanlar eigenvalue değerlerine karşılık gelir.

Örnek olarak aşağıdaki fonksiyon QR algoritması ile eigenvalue hesaplamasını gerçekleştirmektedir:

```python
import numpy as np

def qr_algorithm_eigenvalues(A, iterations=100):
    """
    Basit QR algoritması kullanılarak kare A matrisinin eigenvalue'lerini iteratif olarak hesaplar.
    Varsayım: A, iyi şartlanmış bir kare matrisdir.
    """
    Ak = np.copy(A)
    for _ in range(iterations):
        Q, R = np.linalg.qr(Ak)
        Ak = R @ Q
    # Ak artık yaklaşık bir üst üçgensel matris; diyagonal elemanlar eigenvalue yaklaşık değerleridir.
    return np.diag(Ak)
