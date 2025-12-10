from math import *

x = [[86, 36],
    [220, 30],
    [211, 158],
    [70, 168],

    [725, 47],
    [846, 58],
    [870, 187],
    [746, 172],

    [748, 689],
    [812, 693],
    [809, 757],
    [745, 753],

    [21, 723],
    [171, 709],
    [172, 860],
    [23, 874]]

y = [[0, 0],
    [7, 0],
    [7, 7],
    [0, 7],

    [34, 0],
    [41, 0],
    [41, 7],
    [34, 7],

    [33, 33],
    [36, 33],
    [36, 36],
    [33, 36],

    [0, 34],
    [7, 34],
    [7, 41],
    [0, 41]]

args = [13.335642393550838, 30.16082490821568, 19.972791460214353, 0.001727594256190984, -1.33439516491101e-06, 56.68297671406648, 30.161018023036913]
args = [0] * 10
[55.483785754144456, -20.009076795309504, 17.93650633966609, 0.0025375892196731024, -4.773814811828802e-07, 54.410241389593736, 20.88259417582198, -0.0005530520748906926, 7.01811022405301e-08, 37.31837672621671]

def predict(x0, y0, args):
    r = (x0 - args[0]) ** 2 + (y0 - args[1]) ** 2
    ans = [x0 * (args[2] + args[3] * r + args[4] * r * r) + args[5],
            y0 * (args[6] + args[7] * r + args[8] * r * r) + args[9]]
    # ans = [ans[0] * cos(args[7]) - ans[1] * sin(args[7]), ans[1] * cos(args[7]) - ans[0] * sin(args[7])]
    return ans

def loss(args):
    ls = 0
    for i in range(len(y)):
        pred = predict(y[i][0], y[i][1], args)
        ls += (pred[0] - x[i][0]) ** 2 + (pred[1] - x[i][1]) ** 2
    return ls


def train():
    for _ in range(1000):
        
        print(_)
        for i in range(len(args)):
            
            l = -1000000000
            r = 1000000000
            while (r - l > 10**-10):
                m1 = l * 2 / 3 + r / 3
                m2 = l / 3 + r * 2 / 3
                
                args[i] = m1
                ls1 = loss(args)
                
                args[i] = m2
                ls2 = loss(args)
                
                # print(m1, ls1, m2, ls2)
                
                if ls1 > ls2:
                    if l == m1: break
                    l = m1
                elif ls1 < ls2:
                    if r == m2: break
                    r = m2
                else:
                    if l == m1: break
                    if r == m2: break
                    l = m1
                    r = m2
                    
    print(args)
    print(loss(args))
    
print(predict(0, 0, args))
train()
print(predict(0, 0, args))
