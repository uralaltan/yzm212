import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import qr, eig


# Manuel QR algoritması için fonksiyon (LucasBN/Eigenvalues-and-Eigenvectors'dan uyarlanmıştır)
def qr_eigenvalues(A, max_iterations=1000, tolerance=1e-10):
    """
    QR algoritması kullanarak bir matrisin özdeğerlerini hesaplar

    Parametreler:
    A: Özdeğerleri hesaplanacak kare matris
    max_iterations: Maksimum iterasyon sayısı
    tolerance: Yakınsama kriteri için tolerans değeri

    Dönüş:
    eigenvalues: Hesaplanan özdeğerler
    """
    n = A.shape[0]
    A_k = A.copy()

    # Yakınsama kontrolü için değişken
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
            print(f"QR algoritması {k + 1} iterasyonda yakınsadı")
            break

        prev_diag = diag.copy()

    if k == max_iterations - 1:
        print(f"QR algoritması maksimum {max_iterations} iterasyonda yakınsamadı")

    # Sonuç - diyagonal elemanlar özdeğerlerdir
    return np.diag(A_k)


def qr_eigenvectors(A, eigenvalues):
    """
    Özdeğerlere karşılık gelen özvektörleri ters güç metodu ile hesaplar

    Parametreler:
    A: Özdeğerleri hesaplanan kare matris
    eigenvalues: Hesaplanan özdeğerler

    Dönüş:
    eigenvectors: Her sütunu bir özvektör olan matris
    """
    n = A.shape[0]
    eigenvectors = np.zeros((n, n))

    for i, eigenvalue in enumerate(eigenvalues):
        # Başlangıç vektörü
        v = np.random.rand(n)
        v = v / np.linalg.norm(v)

        # Ters güç metodu (shift ve invert)
        for _ in range(100):
            # (A - lambda*I)^-1 * v
            w = np.linalg.solve(A - eigenvalue * np.eye(n) + 1e-10 * np.eye(n), v)
            w = w / np.linalg.norm(w)

            # Yakınsama kontrolü
            if np.allclose(v, w) or np.allclose(v, -w):
                break

            v = w

        eigenvectors[:, i] = w

    return eigenvectors


# Test için 3x3 matris oluşturalım
np.random.seed(42)  # Tekrarlanabilirlik için
A = np.random.rand(3, 3)
print("Oluşturulan matris A:")
print(A)

# Simetrik matris yapalım (daha kolay yakınsama için)
A_sym = A @ A.T
print("\nSimetrik matris A_sym = A @ A.T:")
print(A_sym)

# 1. Manuel QR Algoritması ile özdeğerler
manual_eigenvalues = qr_eigenvalues(A_sym)
print("\nManuel QR algoritmasıyla hesaplanan özdeğerler:")
print(manual_eigenvalues)

# Manuel özvektörler hesaplama
manual_eigenvectors = qr_eigenvectors(A_sym, manual_eigenvalues)
print("\nManuel hesaplanan özvektörler (her sütun bir özvektör):")
print(manual_eigenvectors)

# 2. NumPy eig ile özdeğerler ve özvektörler
np_eigenvalues, np_eigenvectors = eig(A_sym)
print("\nNumPy eig fonksiyonuyla hesaplanan özdeğerler:")
print(np_eigenvalues)
print("\nNumPy eig fonksiyonuyla hesaplanan özvektörler (her sütun bir özvektör):")
print(np_eigenvectors)

# Karşılaştırma
print("\n--- Karşılaştırma ---")
print("Özdeğerlerin mutlak farkı:")
# Özdeğerleri büyükten küçüğe sıralayıp karşılaştıralım
sorted_manual = np.sort(manual_eigenvalues)[::-1]
sorted_numpy = np.sort(np_eigenvalues.real)[::-1]  # Kompleks kısımları varsa yok sayıyoruz
diff = np.abs(sorted_manual - sorted_numpy)
print(diff)
print(f"Ortalama mutlak hata: {np.mean(diff)}")
print(f"Maksimum mutlak hata: {np.max(diff)}")

# Grafik görselleştirme
plt.figure(figsize=(10, 6))
bar_width = 0.35
x = np.arange(len(sorted_manual))

plt.bar(x - bar_width / 2, sorted_manual, bar_width, label='Manuel QR')
plt.bar(x + bar_width / 2, sorted_numpy, bar_width, label='NumPy eig')

plt.xlabel('Özdeğer İndeksi')
plt.ylabel('Özdeğer Değeri')
plt.title('QR Algoritması vs NumPy eig Karşılaştırması')
plt.xticks(x)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Özvektörlerin doğruluğunu kontrol edelim
print("\nÖzvektör doğrulama (Av - λv ≈ 0 olmalı):")
for i in range(len(manual_eigenvalues)):
    v = manual_eigenvectors[:, i]
    lambda_i = manual_eigenvalues[i]
    residual = np.linalg.norm(A_sym @ v - lambda_i * v)
    print(f"Manuel özvektör {i + 1} için artık normu: {residual}")

for i in range(len(np_eigenvalues)):
    v = np_eigenvectors[:, i]
    lambda_i = np_eigenvalues[i]
    residual = np.linalg.norm(A_sym @ v - lambda_i * v)
    print(f"NumPy özvektör {i + 1} için artık normu: {residual}")

# Daha yüksek boyutlu bir test
print("\n\n--- 5x5 Matris Testi ---")
n = 5
B = np.random.rand(n, n)
B_sym = B @ B.T  # Simetrik matris

# Hesaplama süresini karşılaştıralım
import time

start_time = time.time()
manual_eigenvalues_B = qr_eigenvalues(B_sym)
manual_time = time.time() - start_time
print(f"Manuel QR hesaplama süresi: {manual_time:.6f} saniye")

start_time = time.time()
numpy_eigenvalues_B, _ = eig(B_sym)
numpy_time = time.time() - start_time
print(f"NumPy eig hesaplama süresi: {numpy_time:.6f} saniye")
print(f"Hız farkı: NumPy, manuel hesaplamadan {manual_time / numpy_time:.2f} kat daha hızlı")
