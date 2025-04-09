import numpy as np

def qr_algorithm_eigenvalues(A, iterations=1000):
    """
    Basit QR algoritması kullanılarak kare A matrisinin eigenvalue'lerini iteratif olarak hesaplar.

    Parametreler:
        A : (n, n) boyutunda numpy.ndarray
            Eigenvalue'leri hesaplanacak kare matris.
        iterations : int, isteğe bağlı (varsayılan=1000)
            QR algoritmasının çalıştırılacağı iterasyon sayısı.

    Returns:
        numpy.ndarray: Yaklaşık olarak hesaplanan eigenvalue'lerin 1-boyutlu dizisi,
                       A matrisinin yaklaşık üst üçgensel hale gelmiş halinin diyagonal elemanları.
    """
    Ak = np.copy(A)
    for _ in range(iterations):
        Q, R = np.linalg.qr(Ak)
        Ak = R @ Q
    return np.diag(Ak)


def numpy_eigenvalues(A):
    """
    NumPy'nın np.linalg.eig fonksiyonunu kullanarak A matrisinin eigenvalue ve
    sağ eigenvektörlerini hesaplar.

    Parametreler:
        A : (n, n) boyutunda numpy.ndarray
            Eigenvalue ve eigenvektör hesaplaması yapılacak kare matris.

    Returns:
        tuple: (values, vectors)
            - values: eigenvalue'leri içeren 1-boyutlu dizi.
            - vectors: her eigenvalue'ye karşılık gelen sağ eigenvektörleri, sütun vektörler olarak.
    """
    values, vectors = np.linalg.eig(A)
    return values, vectors


def main():
    # Örnek Matris: Simetrik bir matris, hesaplama için iyi bir örnektir.
    A = np.array([[2, 1, 0],
                  [1, 2, 1],
                  [0, 1, 2]], dtype=float)

    # Manuel hesaplama (QR algoritması)
    manual_eigs = qr_algorithm_eigenvalues(A, iterations=1000)

    # NumPy'nın hazır fonksiyonuyla hesaplama
    numpy_eigs, numpy_eigvecs = numpy_eigenvalues(A)

    # Sonuçların karşılaştırılması
    print("Orijinal Matris A:")
    print(A)
    print("\nQR Algoritması ile Manuel Hesaplanan Eigenvalue'ler:")
    print(manual_eigs)
    print("\nnp.linalg.eig Fonksiyonu ile Hesaplanan Eigenvalue'ler:")
    print(numpy_eigs)
    print("\nnp.linalg.eig Fonksiyonu ile Hesaplanan Eigenvektörler (Sütun Vektörleri):")
    print(numpy_eigvecs)


if __name__ == '__main__':
    main()
