import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


def LT_tests(df):
    N=4
    results = []
    #test 15
    #hm0
    for i in range(len(df)):
        wvhgt_mean = df.hm0.mean()
        wvhgt_sd = df.hm0.std()
        t15=0
        t16=0
        t19=0
        t20=0
        if df.hm0[i]<(wvhgt_mean-(N*wvhgt_sd)) or df.hm0[i]>(wvhgt_mean+(N*wvhgt_sd)):
            j=4
            t15=1
	else:
            j=1
    #test 19
    #hm0,tm02,mdir
    minwh = 0 
        maxwh = 30 #look for climatological ranges - east/west??
        minwp = 0
        maxwp = 18 #This is a guess, not sure what the largest wave period should be
        minwd = 0.0
        maxwd = 360
        wvpd_max = max(df.tm02)
        wvpd_min = min(df.tm02)
        wvdir_max = max(df.mdir)
        wvdir_min = min(df.mdir)
        if df.hm0[i] < minwh or df.hm0[i] >= maxwh:
            j = 4
            t19=1
        elif wvpd_min <minwp or wvpd_max > maxwp or wvdir_min < minwd or maxwd >360:
            j = 3
            t19 = 1   
	#test 20
  #spike test
  #hm0
        delta = 3
        if i==len(df)-1:
            break
        #diff = abs(df.hm0[i]-df.hm0[i+1])
        if abs(df.hm0[i]-df.hm0[i+1])>=delta:
            j=4
            t20=1
  #test 16
  #flatline test
  #hm0
        epsilon = 0.001
        if i == len(df)-5:
            break
        
        diff_1 = abs(df.hm0[i]-df.hm0[i+1])
        diff_2 = abs(df.hm0[i+1]-df.hm0[i+2])
        diff_3 = abs(df.hm0[i+2]-df.hm0[i+3])
        diff_4 = abs(df.hm0[i+3]-df.hm0[i+4])
        diff_5 = abs(df.hm0[i+4]-df.hm0[i+5])
        if diff_1<epsilon and diff_2<epsilon and diff_3<epsilon and diff_4<epsilon and diff_5<epsilon and df.hm0[i]!=0:
            j=4
            t16=1
        results.append([df.time[i],df.hm0[i],df.tm02[i],df.mdir[i],j,t15,t16,t19,t20])
        results_df = pd.DataFrame(results, columns=["time", "hm0","tm02","mdir","flag","test_15","test_16","test_19","test_20"])
        results_df.to_csv('QC.csv', index=False)
    print(results_df)
