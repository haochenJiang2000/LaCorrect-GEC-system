def split_data(data):
    split_data = []
    for seq in data:
        # 统一符号格式，去除特殊符号
        seq = seq.replace("?", "？")
        seq = seq.replace("!", "！")
        seq = seq.replace(",", "，")
        seq = seq.replace(";", "；")
        seq = seq.replace("\"", "”")

        # 去除句中空格
        seq = seq.replace(" ", "")
        seq = seq.replace("　", "")
        seq = seq.replace("­", "")
        seq = seq.strip()

        # 分句
        seq_list1 = []
        seq_list2 = []
        temp = seq.split('。')
        for i, l in enumerate(temp[:-1]):
            temp[i] = l + '。'
        seq_list1 += temp[:-1]
        if len(temp[-1]) > 5:
            seq_list1.append(temp[-1])

        for seq in seq_list1:
            temp = seq.split('！')
            for i, l in enumerate(temp[:-1]):
                temp[i] = l + '！'
            seq_list2 += temp[:-1]
            if len(temp[-1]) > 5:
                seq_list2.append(temp[-1])

        seq_list1 = []
        for seq in seq_list2:
            temp = seq.split('？')
            for i, l in enumerate(temp[:-1]):
                temp[i] = l + '？'
            seq_list1 += temp[:-1]
            if len(temp[-1]) > 5:
                seq_list1.append(temp[-1])

        seq_list2 = []
        for seq in seq_list1:
            temp = seq.split('；')
            for i, l in enumerate(temp[:-1]):
                temp[i] = l + '；'
            seq_list2 += temp[:-1]
            if len(temp[-1]) > 5:
                seq_list2.append(temp[-1])
        split_data = seq_list2[:]
    return split_data
