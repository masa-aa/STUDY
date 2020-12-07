# 清水寺 緯度: 34.994856 経度: 135.785046 -> {"清水寺":(34.994856, 135.785046)}
pos = {}
for i in range(25):
    s = input().split()
    pos[s[0]] = (float(s[2]), float(s[4]))

print(pos)
