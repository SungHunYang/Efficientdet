new_con = {1: 19760, 2: 11064, 3: 15753, 4: 12400, 5: 11113, 6: 1465}
new_fe = {1: 8450, 2: 6359, 3: 5253, 4: 3874, 5: 3084, 6: 932}
new_des = {1: 10546, 2: 7296, 3: 8669, 4: 7031, 5: 7468, 6: 1284}


total_t = 0
total_v = 0

train = {}
val = {}
rate = 0.92

cnt = 1
for i in new_con.values():
    train[cnt] = round(i*rate)
    val[cnt] = i - train[cnt]
    cnt+=1

print(f"Conc_Train : {train}")
print(f"Conc_Val : {val}")
print(f"Conc_Total : {sum(train.values())} / {sum(val.values())}")
print()

total_t += sum(train.values())
total_v += sum(val.values())

train = {}
val = {}

cnt = 1
for i in new_des.values():
    train[cnt] = round(i * rate)
    val[cnt] = i - train[cnt]
    cnt += 1

print(f"Des_Train : {train}")
print(f"Des_Val : {val}")
print(f"Des_Total : {sum(train.values())} / {sum(val.values())}")
print()
total_t += sum(train.values())
total_v += sum(val.values())


train = {}
val = {}

cnt = 1
for i in new_fe.values():
    train[cnt] = round(i * rate)
    val[cnt] = i - train[cnt]
    cnt += 1

print(f"Fe_Train : {train}")
print(f"Fe_Val : {val}")
print(f"Fe_Total : {sum(train.values())} / {sum(val.values())}")
print()
total_t += sum(train.values())
total_v += sum(val.values())

print(f"Train : Val = {total_t} : {total_v}")