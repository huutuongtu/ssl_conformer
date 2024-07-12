# xác suất không có 2 cặp nào trùng nhau
def calculate_prob(n, c, b):
    res = 1
    for i in range(b):
        res = res*(c-i)/(n-i)*n/c
    return res

n_samples = 35000
n_classes = 146
for i in range(6):
    print(calculate_prob(n_samples, n_classes, 2**(i+4)))