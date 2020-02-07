horizontal_list = [f"{200+i*10}," for i in range(1,32)]
vertical_list = ["30," for i in range(31)]
pulse_direction_list = ["90," for i in range(31)]
with open("list.txt", mode='w') as f:
    f.writelines(horizontal_list)
    f.write("\n")
    f.writelines(vertical_list)
    f.write("\n")
    f.writelines(pulse_direction_list)
