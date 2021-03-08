#by marvin 2021.03.05
import numpy as np
import soundfile as sf
import librosa
import random
import os

samplerate=8000

#获取目录下wav文件
def getFiles(path, suffix):
    return [os.path.join(root, file) for root, dirs, files in os.walk(path) for file in files if file.endswith(suffix)]

#按信噪比混合两个语音
def  mix_wave_snr(wavef1 , wavef2, snr, method="cut"):
    a, a_sr = librosa.load( wavef1, sr=samplerate)
    b, b_sr = librosa.load( wavef2, sr=samplerate)   
    L1 = a.shape[0]
    L2 = b.shape[0]
    if method == "cut":         #截除尾部
        min_length=min(L1, L2)
        a=a[:min_length]
        b=b[:min_length]
    elif method  == "append":     #补0，或重复 
        max_length=max(L1, L2)
        if L1 < L2 :
            a=np.append(a, np.zeros(max_length-L1))
            #L1=np.append(a, a[:max_length-L1])
        if L1 > L2 :
            b=np.insert(b, 0, np.zeros(max_length-L2)) #在开头补0
            #L2=np.append(b, b[:max_length-L2])
        
    sum_s = np.sum(a ** 2)
    sum_n = np.sum(b ** 2)
    #重新计算b的插入权重
    x = np.sqrt(sum_s/(sum_n * pow(10, snr/10)))
    b_new=x*b
    target = a + b_new 
    return  a, b_new, target

#循环一个目录，随即抽两个数据，随机选一个信噪比
def mix_one_dir(inDir, outDir):
    list_snr=np.arange(11)   # db
    merge_method="append"
    list_files =getFiles(inDir ,  '.wav')
    random.shuffle(list_files)

    for i in range(0, len(list_files)-1,2):
        snr=random.choice(list_snr)
        a,b,mix=mix_wave_snr(list_files[i], list_files[i+1], snr, merge_method)
        name1=list_files[i].split('/')[-2]
        name2=list_files[i+1].split('/')[-2]
        print(name1, name2, snr)
        if  not os.path.exists(outDir + '/s1/' ):
            os.makedirs(outDir + '/s1/')
            os.makedirs(outDir + '/s2/')
            os.makedirs(outDir + '/mix/')
        sf.write(outDir + '/s1/' + name1+ '.wav',  a, samplerate)
        sf.write(outDir + '/s2/' + name2 + '.wav', b, samplerate)
        sf.write(outDir + '/mix/'   + name1 + '_' +name2 +"_"+ str(snr) + 'db.wav', mix, samplerate)


#main
if  __name__ == '__main__':
    rootIndir='/data/wangmanhong/TasNet/data/jdd'
    rootOutdir='/data/wangmanhong/TasNet/data/jdd_mix2'
    for mydir in ["dev", "test", "train"] :
        print(mydir)
        srcdir=rootIndir + "/"+ mydir
        destdir=rootOutdir+ "/" + mydir
        mix_one_dir(srcdir,destdir)
    print("done")





