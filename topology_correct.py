from PIL import Image
import numpy as np
import homographic_correct

predsetP1 = np.array([[[86, 36], [220, 30], [211, 158], [70, 168]],
                      [[725, 47], [846, 58], [870, 187], [746, 172]],
                      [[21, 723], [171, 709], [172, 860], [23, 874]]])

predsetP2 = np.array([[[68, 93], [116, 106], [114, 158], [68, 144]],
                      [[370, 135], [421, 132], [415, 180], [364, 184]],
                      [[68, 353], [111, 368], [110, 407], [68, 393]]])

def parallelogram_transform(v):
    v1 = np.array(v[0])
    v2 = np.array(v[1])
    v3 = np.array(v[2])
    v4 = np.array(v[3])
    
    A = np.array([v1, -v2, v3]).T
    
    try:
        lambd = np.append(np.linalg.inv(A) @ v4, [1])
    except:
        lambd = np.array([1, 1, 1, 1])
    
    ans = np.array([v1 * lambd[0], v2 * lambd[1], v3 * lambd[2], v4 * lambd[3]])
    
    S = np.linalg.norm(np.linalg.cross(ans[1] - ans[0], ans[3] - ans[0]))
    
    ans /= np.sqrt(S)
    return ans


def plane2vec(v):
    n = np.linalg.cross(v[1] - v[0], v[2] - v[0])
    n /= np.linalg.norm(n)
    
    return n * (n @ v[0])


def v2base(v, coords, nl = 7 * 10):
    i = (v[1] - v[0]) / nl
    j = (v[3] - v[0]) / nl
    O = v[0] - i * coords[0] - j * coords[1]
    return np.array([O, i, j])

def projectbase(base, coords, x):
    y = base[0] + base[1] * coords[0] + base[2] * coords[1]
    
    try:
        a = y[2] / (y[2] - x[2])
        return np.array([x[0] * a + y[0] * (1 - a),
                         x[1] * a + y[1] * (1 - a)])
    except:
        return np.array([0, 0])

def con9points(v):
    v = np.array(v)
    
    A = []
    for p in v:
        x, y, z = p
        A.append([x * x, y * y, z * z, x * y, y * z, x * z, x, y, z])
    
    A = np.array(A)
    
    b = np.array([1] * 9)
    
    try:
        return np.linalg.inv(A) @ b
    except:
        return np.array([0] * 9)
    

def print_as_con(v):
    l = ["xx", "yy", "zz", "xy", "yz", "xz", "x", "y", "z"]
    
    for i in range(len(v)):
        print(f"{v[i]:.15f}", l[i], "+", sep="", end="")
    print("0=1")


def pw(x, a):
    s = int(x >= 0)
    
    return s * abs(x) ** a

def flat(v):
    ans = np.zeros([4, 2])
    ans[1][0] = np.linalg.norm(v[3] - v[0])
    ans[2][0] = (v[3] - v[0]) @ (v[1] - v[0]) / np.linalg.norm(v[3] - v[0])
    ans[2][1] = np.linalg.norm(np.linalg.cross(v[3] - v[0], v[1] - v[0])) / np.linalg.norm(v[3] - v[0])
    ans[3] = ans[1] + ans[2]
    return ans

def evrFLAT(v):
    v = flat(v)
    return  np.exp(-np.linalg.norm(v[2] - np.array([0, v[1][0]])))

def evrP(v):
    v = np.array(v)
    P = np.linalg.norm(v[1] - v[0]) + np.linalg.norm(v[3] - v[0])
    P *= 2
    return 4 / P

def evrCOS(v):
        return 1 - abs((v[3] - v[0]) @ (v[1] - v[0]) / (np.linalg.norm(v[3] - v[0]) * np.linalg.norm(v[1] - v[0])))

def loss(v):
    return -np.log(evrFLAT(v))

def calc_loss(p, x):
    ans = 0
    for v in p:
        ans += loss(parallelogram_transform(v - x))
    
    return ans

def calc_lossMAX(p, x):
    return max(loss(parallelogram_transform(v - x)) for v in p)

def calc_grad(p, x, e):
    grad = np.array([0., 0., 0.])
    now = calc_lossMAX(p, x)
    
    for i in range(3):
        x[i] += e
        grad[i] = (calc_lossMAX(p, x) - now) / e
        x[i] -= e
    
    return grad

def fit(p, e = 0.001, it = 10000, alpha = 500):
    
    x = np.array([500., 500., 500.])
    
    for _ in range(it):
        grad = calc_grad(p, x, e) * alpha
        grad[2] = 0
        # for i in range(3):
        #     if grad[i] > 0:
        #         grad[i] = max(10., grad[i])
        #     else:
        #         grad[i] = min(-10., grad[i])
        
        x -= grad
    
    return x



def mark(p):
    image = Image.open("orig_bynarized1.png")
    pixels = image.load()
    
    for pp in p:
        for ppp in pp:
            pixels[*ppp] = (255, 0, 0)
    image.save("marked.png")

def print_as_points(a):
    ans = ""
    for p in a:
        ans += "("
        for x in p:
            ans += str(x) + ", "
        ans = ans[:-2] + ")\n"
    print(ans)

def scan(p):
    ans = []
    for x in range(-20, 20, 1):
        for y in range(-20, 20, 1):
            for z in range(-20, 20, 1):
                v = np.array([float(x), float(y), float(z)])
                if calc_lossMAX(p, v) < 0.1:
                    ans.append(v)
    print_as_points(ans)


def correct(orig):
    p = predsetP1
    sz = 41
    p = np.pad(p, pad_width=((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0)
    
    
    current = np.array([(0, 0), (1, 0), (1, 1), (0, 1)])
    current *= 70
    # current += [20, 20]
    
    expected1 = p[0]
    
    x1 = fit([expected1])
    base1 = v2base(parallelogram_transform(expected1 - x1) * 70 + x1, current[0], 70)
    
    # print_as_points([x1])
    # print_as_points([base1[0], base1[1] + base1[0], base1[2] + base1[0]])
    # print_as_points(expected1)
    # print_as_points(parallelogram_transform(expected1 - x1) * 70 + x1)
    
    current += [(sz - 7) * 10, 0]
    expected2 = p[1]
    
    x2 = fit([expected2])
    base2 = v2base(parallelogram_transform(expected2 - x2) * 70 + x2, current[0], 70)
    
    current += [-(sz - 7) * 10, (sz - 7) * 10]
    expected3 = p[2]
    
    x3 = fit([expected3])
    base3 = v2base(parallelogram_transform(expected3 - x3) * 70 + x3, current[0], 70)
    
    orig = orig.convert("RGBA")
    image = Image.new("RGBA", (500, 500))
    orig_pixels = orig.load()
    pixels = image.load()

    weight, height = image.size

    for x in range(weight):
        for y in range(height):
            
            # pixels[x, y] = col((x, y), (0, 0), (1023, 0), (0, 1023))
            # continue
            
            w1, w2, w3 = homographic_correct.baric((0, 0), (sz * 10, 0), (0, sz * 10), (x, y))
            # w1, w2, w3 = holographic_correct.baric((70, 70), (340, 70), (70, 340), (x, y))
            # if x < 70:
            #     w1, w2, w3 = holographic_correct.baric((70, 70), (340, 70), (70, 340), (70, y))
            # elif y < 70:
            #     w1, w2, w3 = holographic_correct.baric((70, 70), (340, 70), (70, 340), (x, 70))
            # a = 1
            # pw(w1, a)
            # pw(w2, a)
            # pw(w3, a)
            
            X = (x1 * w1 + x2 * w2 + x3 * w3) / (w1 + w2 + w3)
            base = (base1 * w1 + base2 * w2 + base3 * w3) / (w1 + w2 + w3)
            
            X = x1
            base = base1
            
            try:
                pixels[x, y] = orig_pixels[*projectbase(base, (x, y), X)]
                # pixels[x, y] = orig_pixels[*F((x, y), H3)]
            except:
                pixels[x, y] = (0, 0, 0, 0)
    
    return image


if __name__ == "__main__":
    # print(holographic_correct.baric((0, 0), (410, 0), (0, 410), (410, 410)))
    # quit()
    
    orig = Image.open("orig_bynarized.png")
    # orig = Image.open("grid.png")
    orig = correct(orig)
    orig.save("corrected.png")

if __name__ == "__main__" and False:
    p = np.array([[[1, 1, 0], [-1, 1, 0], [-2, -2, 0], [1, -1, 0]]])
    
    
    
    c = []
    
    for i in range(3):
        
        
        
        # p = np.array([[[68, 93], [116, 106], [114, 158], [68, 144]],
        #             [[370, 135], [421, 132], [415, 180], [364, 184]],
        #             [[68, 353], [111, 368], [110, 407], [68, 393]]])
        
        
        
        # p = np.array([[[0, 0], [4, 0], [6, 4], [3, 3]]])
        
        
        p = np.pad(p, pad_width=((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0)
        # print(p)
        p = [p[i]]
        
        
        x = fit(p)
        prl = parallelogram_transform(p[0] - x) * 300 + x
        print_as_points(p[0])
        print_as_points([x])
        print_as_points(prl)
        c.append(prl[0])
        c.append(prl[1])
        c.append(prl[3])
    
    
    print_as_con(con9points(c))