import os
import re
#创建包含原子序号和元素符号对应的字典
atomicnum2elem = {1 : 'H',
                    2 : 'He',
                    3 : 'Li',
                    4 : 'Be',
                    5 : 'B',
                    6 : 'C',
                    7 : 'N',
                    8 : 'O',
                    9 : 'F',
                    10 : 'Ne',
                    11 : 'Na',
                    12 : 'Mg',
                    13 : 'Al',
                    14 : 'Si',
                    15 : 'P',
                    16 : 'S',
                    17 : 'Cl',
                    18 : 'Ar',
                    19 : 'K',
                    20 : 'Ca',
                    21 : 'Sc',
                    22 : 'Ti',
                    23 : 'V',
                    24 : 'Cr',
                    25 : 'Mn',
                    26 : 'Fe',
                    27 : 'Co',
                    28 : 'Ni',
                    29 : 'Cu',
                    30 : 'Zn',
                    31 : 'Ga',
                    32 : 'Ge',
                    33 : 'As',
                    34 : 'Se',
                    35 : 'Br',
                    36 : 'Kr',
                    37 : 'Rb',
                    38 : 'Sr',
                    39 : 'Y',
                    40 : 'Zr',
                    41 : 'Nb',
                    42 : 'Mo',
                    43 : 'Tc',
                    44 : 'Ru',
                    45 : 'Rh',
                    46 : 'Pd',
                    47 : 'Ag',
                    48 : 'Cd',
                    49 : 'In',
                    50 : 'Sn',
                    51 : 'Sb',
                    52 : 'Te',
                    53 : 'I',
                    54 : 'Xe',
                    55 : 'Cs',
                    56 : 'Ba',
                    57 : 'La',
                    58 : 'Ce',
                    59 : 'Pr',
                    60 : 'Nd',
                    61 : 'Pm',
                    62 : 'Sm',
                    63 : 'Eu',
                    64 : 'Gd',
                    65 : 'Tb',
                    66 : 'Dy',
                    67 : 'Ho',
                    68 : 'Er',
                    69 : 'Tm',
                    70 : 'Yb',
                    71 : 'Lu',
                    72 : 'Hf',
                    73 : 'Ta',
                    74 : 'W',
                    75 : 'Re',
                    76 : 'Os',
                    77 : 'Ir',
                    78 : 'Pt',
                    79 : 'Au',
                    80 : 'Hg',
                    81 : 'Tl',
                    82 : 'Pb',
                    83 : 'Bi',
                    84 : 'Po',
                    85 : 'At',
                    86 : 'Rn',
                    87 : 'Fr',
                    88 : 'Ra',
                    89 : 'Ac',
                    90 : 'Th',
                    91 : 'Pa',
                    92 : 'U',
                    93 : 'Np',
                    94 : 'Pu',
                    95 : 'Am',
                    96 : 'Cm',
                    97 : 'Bk',
                    98 : 'Cf',
                    99 : 'Es',
                    100 : 'Fm',
                    101 : 'Md',
                    102 : 'No',
                    103 : 'Lr',
                    104 : 'Rf',
                    105 : 'Db',
                    106 : 'Sg',
                    107 : 'Bh',
                    108 : 'Hs',
                    109 : 'Mt',
                    110 : 'Ds',
                    111 : 'Rg',
                    112 : 'Uub',}

interval = 5 #"standard orientation"和坐标行之间的行差
with open("coordinates-only.xyz", 'w', encoding = 'utf-8') as coordinates:
    for file in os.listdir():
        #只读取log文件
        if file[-3:] != 'log':
            continue
        with open(file, 'r') as f:
            numOfAtom = 0
            method = '' #计算方法
            scf_energy = 0
            coord_ls = []
            line_ls = f.readlines()

            # 遍历文件找到原子个数
            for line in line_ls:         
                if "NAtoms" in line:
                    numOfAtom = eval(line.split()[1])
                    break
                    
            #遍历文件找到优化过程中最后的坐标和最后的scf和方法
            for i, line in enumerate(line_ls):
                if "Standard orientation" in line:
                    start = i + interval
                    end = i + interval + numOfAtom
                    coord_ls = line_ls[start : end]

                if "SCF Done" in line:
                    scf_energy = eval(line.split()[4])
                    method = line.split()[2][2:-1]
            
        #将coord_list里的每行str转换为包含元素和坐标的列表
        for i, item in enumerate(coord_ls):
            item = item.split()
            item = [eval(value) for value in item]
            coord_ls[i] = [item[1], item[3], item[4], item[5]]
       
        #找到file_name
        # ts.log            --->    ts
        # ts-scf.log        --->    ts
        # ts-scf-scf.log    --->    ts
        file_name = re.findall('(.+?)(?:-scf)*.log', file)[0]


        coordinates.write("{}\n".format(file_name))
        
        for item in coord_ls:
            coordinates.write("%2s%15.6f%15.6f%15.6f\n"%(atomicnum2elem[item[0]], item[1], item[2], item[3]))
            
        coordinates.write('\n')

