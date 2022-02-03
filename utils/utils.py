import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import collections

from sklearn.preprocessing import StandardScaler

def get_file_structure():
    path_dataset = './dataset'
    path_data2013 = os.path.join(path_dataset, '2013data')
    
    data2013 = sorted(os.listdir(path_data2013))
    for folder in data2013:
        if not folder.startswith('KELS'):
            data2013.remove(folder)    

    kels = []
    for idx, folder in enumerate(data2013):
        csv_files = sorted(os.listdir(os.path.join(path_data2013, data2013[idx])))
        kels.append(csv_files)
    
        # # extract file names
        # file_names = []
        # for file_ in csv_files:
        #     file_names.append(file_.split(' ')[-1].split('_')[0])
        
    # kels: [[L2Y1S, ...]]
    return kels

def get_csvs():
    path_dataset = './dataset'
    path_data2013 = os.path.join(path_dataset, '2013data')
    data2013 = sorted(os.listdir(path_data2013))
    for folder in data2013:
        if not folder.startswith('KELS'):
            data2013.remove(folder)  

    kels = get_file_structure()
    csv_files = []

    for kels_idx, folder in enumerate(kels):
        csv_year = []
        
        for idx, survey in enumerate(folder):
            if survey.endswith('.csv'):
                csv_year.append(pd.read_csv(os.path.join(path_data2013, data2013[kels_idx], survey), low_memory=False, thousands=','))
            # exec("%s = pd.read_csv(os.path.join(path_data2013, data2013[kels_idx], survey), low_memory=False, thousands=',')" % (kels[kels_idx][idx]))
            # exec("csv_year.append(%s)" % (kels[kels_idx][idx]))
        
        csv_files.append(csv_year)

    return csv_files

def get_savs():
    path_dataset = './dataset'
    path_preprocessed = './preprocessed/raw'
    kels2013 = sorted(os.listdir(path_dataset))
    replaces = [["남학생", 1], ["여학생",2], ["특별시",1], ["대도시", 2], ["중소도시", 3], ["읍면지역",4], ["일반", 1], ["소규모", 2], ["국공립", 1], ["사립", 2], ["미참여", 0], ["참여", 1], ["기초미달", 1], ["기초", 2], ["보통", 3], ["우수", 4], ["ⓞ받은 적이 없다", 0], ["ⓞ 받은 적이 없다", 0], ["① 전혀 그렇지 않다", 1],["① 전혀 하지 않는다", 1], ["① 전혀 도움이 되지 않는다", 1], ["① 전혀 만족하지 않는다", 1], ["① 전혀 없다", 1], ["② 그렇지 않다", 2], ["② 만족하지 않는다", 2], ["② 별로 하지 않는다", 2], ["②", 2], ["② 거의 없다", 2], ["② 도움이 되지 않는 편이다", 2], ["② 만족하지 않는다", 2], ["③ 보통이다", 3], ["③", 3], ["③ 가끔 있다", 3], ["④ 그렇다", 4], ["④", 4], ["④ 자주 있다", 4], ["④ 도움이 되는 편이다", 4], ["④ 만족한다", 4], ["④ 가끔 하는 편이다", 4], ["⑤ 매우 그렇다", 5], ["⑤ 자주 하는 편이다", 5], ["⑤ 매우 많다", 5], ["⑤ 매우 도움이 된다", 5],  ["⑤ 매우 만족한다", 5],  ["전혀 그렇지 않다", 1], ["그렇지 않다", 2], ["보통이다", 3], ["그렇다", 4], ["매우 그렇다", 5], ["20% 이하", 1], ["21~40%", 2], ["21-40%", 2], ["41~60%", 3], ["41-60%", 3], ["61~80%", 4], ["61-80%", 4], ["81% 이상", 5], ["10분 이하", 1], ["11~20분", 2], ["11-20분", 2], ["21~30분", 3], ["21-30분", 3], ["31분 이상", 4], ["31-40분", 4], ["31~40분", 4], ["31분~40분", 4], ["41분 이상", 5], ["있다", 1], ["없다", 2], ["해당", 1], ["비해당", 2], ["전혀 없음", 0], ["전혀없음", 0], ["1시간", 1], ["2시간", 2], ["3시간", 3], ["4시간", 4], ["5시간", 5], ["6시간", 6], ["6시간 이상", 6], ["7시간", 7], ["8시간", 8], ["9시간 이상", 9], ["하지않음", 1], ["1시간 미만", 2], ["1시간 이상 2시간 미만", 3], ["1시간 이상-2시간 미만", 3], ["2시간 이상 3시간 미만", 4], ["2시간 이상-3시간 미만", 4], ["3시간 이상", 5], ["하지 않음", 1], ["전혀 하지 않는다", 1], ["별로 하지 않는다", 2], ["보통이다", 3], ["가끔 하는 편이다", 4], ["자주 하는 편이다", 5], ["전혀 없다", 1], ["거의 없다" , 2], ["가끔 있다", 3], ["자주 있다", 4], ["매우 많다", 5], ["예", 1], ["아니오", 2], ["받은 적이 없다", 0], ["전혀 도움이 되지 않는다", 1], ["도움이 되지 않는 편이다", 2], ["보통이다", 3], ["도움이 되는 편이다", 4], ["매우 도움이 된다", 5], ["전혀 만족하지 않는다", 1], ["만족하지 않는다" ,2], ["보통이다", 3], ["만족한다", 4], ["매우 만족한다", 5], ["찬성", 1], ["반대", 2], ["없음", 0], ["있음", 1], ["읽지 않음", 0], ["1권", 1], ["2권", 2], ["3권", 3], ["4권", 4], ["5권", 5], ["6권", 6], ["7권", 7], ["8권", 8], ["9권", 9], ["10권 이상", 10], ["전혀 하지 않는다", 0], ["연1회", 1], ["학기에 1-2회", 2], ["6개월에 1-2회", 2], ["3개월에 1-2회", 3], ["분기에 1-2회", 3], ["월 1-2회", 4], ["주 1-2회", 5], ["선행학습 안했음", 0], ["중학교 3학년", 1], ["고등학교 1학년", 2], ["고등학교 2학년", 3], ["고등학교 3학년", 4], ["전혀 안함", 0], ["30분 미만", 1], ["30분-1시간 미만", 2], ["1-2시간 미만", 3] ,["2-3시간 미만", 4], ["3-5시간 미만", 5], ["5-8시간 미만", 6], ["8시간 이상", 7], ["없음", 0], ["있음", 1], ["전혀 하지 않음", 0], ["받은 적이 없다", 0], ["(2개 수준)심화", 1], ["(2개 수준)보통(기본)", 2], ["(3개 수준)심화", 3], ["(3개 수준)보통(기본)", 4], ["(3개 수준)기초", 5], ["연 1회", 1], ["전혀 하지 않는다", 0], ["6개월에 1~2회", 2], ["3개월에 1~2회",3], ["월 1~2회", 4], ["주 1~2회", 5], ["전혀 하지 않음", 0], ["1시간 미만", 1], ["1시간 이상~2시간 미만", 2], ["2시간 이상~3시간 미만",3], ["3시간 이상", 4], ["30분 미만", 1], ["30분~1시간 미만", 2], ["1~2시간 미만", 3], ["2~3시간 미만", 4], ["3시간 이상", 5], ]
    idx = 0

    for folder in kels2013:
        if folder.startswith('KELS2013'):
            path_folder = os.path.join(path_dataset, folder)
            kels2013_data = sorted(os.listdir(path_folder))

            for sav_file in kels2013_data:
                if (sav_file.endswith('.sav') or sav_file.endswith(".SAV")) and sav_file.startswith('1'):
                    idx += 1
                    df_sav = pd.read_spss(os.path.join(path_folder, sav_file))
                    for replace in replaces:
                        df_sav = df_sav.replace(replace[0], replace[1])
                        df_sav.to_csv(path_preprocessed+'/L2Y'+str(idx)+'S_raw.csv')

def get_sav_label6():
    file_ = './dataset/KELS2013_6차/4. L2Y6A_학생평가.sav'
    df_sav = pd.read_spss(file_)
    df_sav = df_sav.set_index(['L2SID'])
    
    replaces = [['① 1등급', 1], ['② 2등급', 2], ['③ 3등급', 3], ['④ 4등급', 4], ['⑤ 5등급', 5], ['⑥ 6등급', 6], ['⑦ 7등급', 7], ['⑧ 8등급', 8], ['⑨ 9등급', 9], ['응시함', 1], ['응시안함', 2], ['성적 확인불가', 3]]
    columns_del = ['L2GENDER', 'L2Y6_SCHID', 'L2Y6_REG', 'L2Y6_SCHTYPE1', 'L2Y6_SCHTYPE2', 'L2Y6A'] 
    df_sav = df_sav.drop(columns_del, axis=1)
    for replace in replaces:
        df_sav = df_sav.replace(replace[0], replace[1])

    df_sav = df_sav.astype(float)
    df_label = pd.DataFrame(index=df_sav.index)
    df_label['L2Y6_K_CS'] = df_sav[['L2Y6A01_1_1', 'L2Y6A01_2_1', 'L2Y6A01_3_1']].mean(axis=1).round(0)
    df_label['L2Y6_M_CS'] = df_sav[['L2Y6A01_1_2', 'L2Y6A01_2_2', 'L2Y6A01_3_2']].mean(axis=1).round(0)
    df_label['L2Y6_E_CS'] = df_sav[['L2Y6A01_1_3', 'L2Y6A01_2_3', 'L2Y6A01_3_3']].mean(axis=1).round(0)
    
    df_drop = df_label.dropna()
    df_drop.to_csv('./preprocessed/preprocessed/L2Y6S_label.csv')





def plot_2d(df_input, df_label, label="L2Y1_E_CS"):
    df = pd.merge(df_input, df_label[label], left_index=True, right_index=True, how='left')
    df = df.rename(columns={0:'col1', 1:'col2'})
    grouped = df.groupby(label)

    plt.rcParams['figure.figsize'] = [10, 8]
    fig, ax = plt.subplots()

    for name, group in grouped:
        ax.plot(group.col1, group.col2, marker='o', linestyle='', label=name, alpha=.3)

    ax.legend(fontsize=12, loc='upper left') # legend position

    plt.title('2d Scatter', fontsize=20)
    plt.show()

def count_arr(np_arr):
    label = sorted(set(np_arr))
    count = []
    for l in label:
        count.append(collections.Counter(np_arr)[l])
    return count

def preprocessing_stu(dataframe, year):
    
    columns = list(dataframe.columns)
    columns_del = []

    columns_target_y1 = ['L2Y1_K_SCORE', 'L2Y1_E_SCORE', 'L2Y1_M_SCORE', 'L2Y1_K_CS', 'L2Y1_E_CS', 'L2Y1_M_CS', 'L2Y1_K_THETA', 'L2Y1_E_THETA','L2Y1_M_THETA']
    columns_basicinfo_y1 = ['L2GENDER', 'L2Y1_SCHID', 'L2Y1_REG', 'L2Y1_SCHSIZE', 'L2Y1_SCHTYPE', 'L2Y1S']
    columns_basicinfo_y3 = ['L2GENDER', 'L2Y1_SCHID', 'L2Y1_REG', 'L2Y1_SCHTYPE', 'L2Y1_SCHTYPE2', 'L2Y1S']
    columns_basicinfo_y6 = ['L2GENDER', 'L2Y1_SCHID', 'L2Y1_REG', 'L2Y1_SCHTYPE1', 'L2Y1_SCHTYPE2', 'L2Y1S']
    columns_target, columns_basicinfo = [], []

    for col in columns_target_y1:
        columns_target.append(col.replace('L2Y1','L2Y'+str(year)))
    
    if year==1 or year==2:
        for col in columns_basicinfo_y1:
            columns_basicinfo.append(col.replace('L2Y1','L2Y'+str(year)))
    elif year == 6:
        for col in columns_basicinfo_y6:
            columns_basicinfo.append(col.replace('L2Y1','L2Y'+str(year)))
    else:
        for col in columns_basicinfo_y3:
            columns_basicinfo.append(col.replace('L2Y1','L2Y'+str(year)))    

    ## abnormal columns 
    # year1
    if year == 1:            
        columns_0to9 =['L2Y1S28', 'L2Y1S2901', 'L2Y1S2902', 'L2Y1S31']
        columns_0or1 = ['L2Y1S22', 'L2Y1S23', 'L2Y1S24', 'L2Y1S25']
        columns_1to4 = ['L2Y1S1001', 'L2Y1S1002', 'L2Y1S1003']
        columns_categorical = ['L2Y1S35', 'L2Y1S34', 'L2Y1S33']
        # lots of missing value in question about addicting on phone
        columns_phone = ['L2Y1S3001', 'L2Y1S3002', 'L2Y1S3003', 'L2Y1S3004', 'L2Y1S3005']
        
        for col in columns:
            if col.endswith('O'):
                columns_del.append(col)
        if col.startswith('L2Y1S24'):
            columns_del.append(col)

        columns_del.extend(columns_basicinfo)
        columns_del.extend(columns_categorical)
        columns_del.extend(columns_0or1)
        columns_del.extend(columns_phone)

    # year2
    elif year ==2:
        columns_0or1 = ['L2Y2S24_1_M~']
        columns_1or2 = ['L2Y2S40', 'L2Y2S22', 'L2Y2S23', 'L2Y2S24']
        columns_1to4 = ['L2Y2S10~', 'L2Y2LT~']
        columns_0to9 = ['L2Y2S29', 'L2Y2S31']
        columns_0to6 = ['L2Y2S28']
        columns_categorical = ['L2Y2S33', 'L2Y2S34', 'L2Y2S35', 'L2Y2S36', 'L2Y2S41']

        for col in columns:
            if col.endswith('O'):
                columns_del.append(col)

            # columns 0or1
            if col.startswith('L2Y2S24'):
                columns_del.append(col)

        columns_del.extend(columns_basicinfo)
        columns_del.extend(columns_categorical)
        columns_del.extend(columns_1or2)

    elif year==3:
        columns_1or2 = ['L2Y3S20', 'L2Y3S27', 'L2Y3S44']
        columns_0or1 = ['L2Y3S26~', 'L2Y3S28~']
        columns_0to5 = ['L2Y3S2301', 'L2Y3S2302', 'L2Y3S31~']
        columns_0to10 = ['L2Y3S29']
        columns_0to6 = ['L2Y3S32']
        columns_0to9 = ['L2Y3S33']
        columns_0to4 = ['L2Y3S34~']
        columns_0to7 = ['L2Y3S35~']
        columns_categorical = ['L2Y3S37','L2Y3S39~', 'L2Y3S40', 'L2Y3S38']

        for col in columns:
            if col.endswith('O'):
                columns_del.append(col)

            # columns 0or1
            if col.startswith('L2Y3S26') or col.startswith('L2Y3S28'):
                columns_del.append(col)

            # columns_categorical
            if col.startswith('L2Y3S38') or col.startswith('L2Y3S39') or col=='L2Y3S37' or col=='L2Y3S40':
                columns_del.append(col)

        columns_del.extend(columns_basicinfo)
        columns_del.extend(columns_1or2)
        columns_del.extend(['L2Y3S45'])

    elif year==4:
        columns_0or1 = ['L2Y4S26~','L2Y4S39~','L2Y4S40~']
        columns_1or2 = ['L2Y4S21','L2Y4S25~','L2Y4S27','L2Y4S46']
        columns_0to5 = ['L2Y4S23~', 'L2Y4S24~', 'L2Y4S31~']
        columns_0to10 = ['L2Y4S29']
        columns_0to4 = ['L2Y4S32~','L2Y4S35~']
        columns_0to6 = ['L2Y4S33']
        columns_0to9 = ['L2Y4S34']
        columns_0to7 = ['L2Y4S36~']
        columns_categorical = ['L2Y4S38', 'L2Y4S41']

        for col in columns:
            if col.endswith('O'):
                columns_del.append(col)

            # columns 0or1
            if col.startswith('L2Y4S26') or col.startswith('L2Y4S39') or col.startswith('L2Y4S40'):
                columns_del.append(col)

            # columns 1or2
            if col.startswith('L2Y4S25') or col.startswith('L2Y4S46') or col=='L2Y4S21' or col=='L2Y4S27' or col=='L2Y4S47':
                columns_del.append(col)

        columns_del.extend(columns_basicinfo)
        columns_del.extend(columns_categorical)
        # columns_del.extend(['L2Y4S47'])

    elif year==5:
        columns_1or2 = ['L2Y5S20','L2Y5S27','L2Y5S47~']
        columns_0or1 = ['L2Y5S26~']
        columns_0to5 = ['L2Y5S22~','L2Y5S23~', 'L2Y5S25~','L2Y5S31~','L2Y5S36~']
        columns_0to10 = ['L2Y5S29']
        columns_0to3 = ['L2Y5S32~']
        columns_0to6 = ['L2Y5S33']
        columns_0to9 = ['L2Y5S34']
        columns_0to4 = ['L2Y5S35~']
        columns_1to4 = ['L2Y5S41~']
        columns_categorical = ['L2Y5S38','L2Y5S39', 'L2Y5S42']

        for col in columns:
            if col.endswith('O'):
                columns_del.append(col)

            # columns 0or1
            if col.startswith('L2Y5S26'):
                columns_del.append(col)

            # columns 1or2
            if col.startswith('L2Y5S47') or col=='L2Y5S20' or col=='L2Y5S27':
                columns_del.append(col)

            # almost of student missed
            if col.startswith('L2Y5S25'):
                columns_del.append(col)

            if col.startswith('L2Y5S41'):
                columns_del.append(col)

        columns_del.extend(columns_basicinfo)
        columns_del.extend(columns_categorical)

    elif year==6:
        columns_1or2 = ['L2Y6S20', 'L2Y6S26']
        columns_0or1 = ['L2Y6S24~']
        columns_0to5 = ['L2Y6S23~', 'L2Y6S30~', 'L2Y6S35~', 'L2Y6S37~']
        columns_1to10 = ['L2Y6S28']
        columns_0to2 = ['L2Y6S31~']
        columns_0to6 = ['L2Y6S32']
        columns_0to9 = ['L2Y6S33']
        columns_0to4 = ['L2Y6S34~']
        columns_categorical = ['L2Y6S38', 'L2Y6S39~', 'L2Y6S40', 'L2Y6S41', 'L2Y6S42', 'L2Y6S43', 'L2Y6S48~']

        for col in columns:
            if col.endswith('O'):
                columns_del.append(col)

            # columns 0or1
            if col.startswith('L2Y6S24'):
                columns_del.append(col)

            # columns categorical
            if col.startswith('L2Y6S39') or col.startswith('L2Y6S48') or col=='L2Y6S38' or col=='L2Y6S40' or col=='L2Y6S42' or col=='L2Y6S43':
                columns_del.append(col)

            if col.startswith('L2Y6S38') or col.startswith('L2Y6S41') or col=='L2Y6S37':
                columns_del.append(col)

        columns_del.extend(columns_basicinfo)
        columns_del.extend(columns_1or2)


    # delete unnecessary columns
    dataframe['L2SID'] = dataframe['L2SID'].apply(np.int64)
    df_drop = dataframe.set_index(['L2SID'])
    df_drop = df_drop.drop(columns_del, axis=1)
    
    # delete null value
    df_drop = df_drop[df_drop != "#NULL!"]
    # L2SID float to int
    df_drop = df_drop.astype(float)
    # negative to zero
    df_drop[df_drop < 0] = 0
    

    # move value(has another sclae) to fit average
    # columns_drop = df_drop.columns
    # for col in columns_drop:
    #     # columns_1to4
    #     if col.startswith('L2Y2S10') or col.startswith('L2Y2LT'):
    #         df_drop[col] += 0


    ## merge 
    # bulid itemized columns
    columns_merge = set()
    
    if year == 2:
        for col in list(df_drop.columns):
            if (col not in columns_target) and not col.startswith("L2Y2LT"):
                columns_merge.add(col[:7])
            elif col.startswith("L2Y2LT"):
                columns_merge.add(col[:8])
    else:
        for col in list(df_drop.columns):
            if col not in columns_target:
                columns_merge.add(col[:7])
    
    columns_merge = list(columns_merge)

    columns_merge_dict = {key:0 for key in columns_merge}
    df_merge = pd.DataFrame(index=df_drop.index)

    # add itemized columns to merge
    for col in columns_merge_dict:
        df_merge[col] = 0

    # merge itemized columns
    for col in list(df_drop.columns):
        for col_itemized in columns_merge_dict:
            if col.startswith(col_itemized):
                columns_merge_dict[col_itemized] += 1
                df_merge[col_itemized] += df_drop[col]

    for col in columns_merge_dict:
        df_merge[col] /= columns_merge_dict[col]
    

    # drop nan 
    rows_with_nan = [index for index, row in df_merge.iterrows() if row.isnull().any()]
    df_merge = df_merge.drop(rows_with_nan)
    
    # split dataframe to target or not
    if year != 6:
        df_target = pd.DataFrame(df_drop[columns_target], index=df_merge.index)
    else:
        df_target = pd.DataFrame([])

    return df_merge, df_target

def preprocessing_label6():
    pass

def preprocessing():
    path_raw = './preprocessed/raw'
    path_preprocessed = './preprocessed/preprocessed'
    file_names = sorted(os.listdir(path_raw))

    for file_name in file_names:
        df_raw = pd.read_csv(os.path.join(path_raw, file_name))
        year = int(file_name[-10])
        df_input, df_label = preprocessing_stu(df_raw, year)
        input_title = 'L2Y'+str(year)+'S_input.csv'
        label_title = 'L2Y'+str(year)+'S_label.csv'

        df_input.to_csv(os.path.join(path_preprocessed, input_title))
        df_label.to_csv(os.path.join(path_preprocessed, label_title))

def merge_df(df_arr):
    df_num = len(df_arr)