import os

def file_remove_same(input_file, output_file,time=0):
    """
        针对小文件去重
    :param input_file: 输入文件
    :param out_file: 去重后出文件
    :return:
    """
    def takeSecond(elem):
        if len(elem.split(','))!=11 and len(elem.split(','))!=12:
            return 0
        return float(elem.split(',')[0])+float(elem.split(',')[1])*10000
    
    with open(input_file, 'r', encoding='utf8') as f, open(output_file, 'w', encoding='utf8') as ff:
        data = [item.strip().strip(',') for item in f.readlines()]  # 针对最后一行没有换行符，与其他它行重复的情况
        new_data = list(set(data))
        new_data.sort(key = takeSecond)
        # print(len(new_data[0].split(',')))
        i,j=1,1
        while i < len(new_data):
            # print(i,len(new_data))
            if new_data[j].split(',')[0]==new_data[j-1].split(',')[0] and new_data[j].split(',')[1]==new_data[j-1].split(',')[1] and j!=0:
                del new_data[j-1]
            else:
                j+=1
            i+=1
        
        write = [item for item in new_data if item and (len(item.split(','))>=11) ]
        info = []
        for w in write:
            i = w.split(',')
            x = i[2:-2:2]
            y = i[3:-1:2]
            s = str(int(i[0])+time)+','+i[1]+','
            c=0
            for j in range(4):
                c+=float(x[j])+float(y[j])
                s += str(max(-640, min(1280+640, float(x[j])))) + ','
                s += str(max(-360, min(720+360, float(y[j])))) + ','
            if c==0:
                continue
            s += i[10]+'\n'
            info.append(s)
        ff.writelines(info)  # 针对去除文件中有多行空行的情况
        
def clean_result(root,sr,modes,time):
    for test_mode in test_modes:
        for mode in modes:
            r = os.path.join(root,mode)
            print(r)
            if not os.path.isdir(r):
                continue
            # sr = os.path.join(sv_root,'clean')
            srm = os.path.join(sr,mode)


            if not os.path.isdir(sr):
                os.makedirs(sr)
            if not os.path.isdir(srm):
                os.makedirs(srm)

            for file in os.listdir(r):
                if file[0] == '.':
                    continue
                print(file)
                file_remove_same(os.path.join(r,file),os.path.join(srm,file),time)

if __name__ == '__main__':
    
    root = '/path/to/res'
    modes = ['test','val']
    for method in ['our']:
        
        sr = os.path.join(root+'clean',method,'insresults')
        if not os.path.isdir(sr):
            os.makedirs(sr)
        clean_result(os.path.join(root,method,'insresults'),sr,modes)
