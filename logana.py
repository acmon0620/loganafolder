#v040 データアップのサイドバー化
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from scipy.stats import skew, kurtosis, anderson, shapiro, boxcox
import func

footer_text = "Ver.0.40"

st.markdown(
    f"""
    <style>
    .footer {{
        position: fixed;
        left: 92%;
        bottom: 0%;
        width: 8%;
        background-color: #f5f5f5;
        color: #000;
        text-align: center;
        padding: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    f"""
    <div class="footer">
        {footer_text}
    </div>
    """,
    unsafe_allow_html=True
)

st.header("SDD LOG解析", divider='red')

# streamlitの表示用リストとタブ
kishu = ['小型BAS','大型BAS','全回転BAS','BUF','DA',7300]
jikan = ['1us', '1ms', '333ns']
options = ['糸切れ','目飛び','締り1','締り2','締り3']
tab1, tab2, tab3, tab4 = st.tabs(["張力解析", "ワーク間解析", "データ整理", "3Dグラフ"])

with st.sidebar:
    st.write("### ▼データアップロード")
    upload_files = st.file_uploader("データをアップロードしてください", type=['bin', 'sddb'], accept_multiple_files=True)
    uploaded_files = sorted(upload_files, key=lambda file: file.name)   # file名で昇順に並び替え
    file_names = [file.name for file in uploaded_files]
    if uploaded_files:
        file_name = st.selectbox('解析するファイルを選択してください', file_names)
        file_no = file_names.index(file_name)

    st.divider()
    st.write("### ▼設定")
    type = st.selectbox('機種設定', kishu)
    count_unit = st.selectbox('1カウントの時間', jikan)
    st.write("3G : 1us")
    st.write("2G : 333ns")

if count_unit == '1us':
    clk = 1000
elif count_unit == '1ms':
    clk = 1000000
else:
    clk = 333

if type == 'BUF':
    stand = 35  # 基準角度
    ten_st = 29 # 天秤開始角度
    ten_ed = 57 # 天秤終了角度
    km_st = 220 # 釜開始角度
    km_ed = 248 # 釜終了角度
    d_st = 137  # 出会い開始角度
    d_ed = 151  # 出会い終了角度
    dd = 2      # 区間2倍
elif type == 'DA':
    stand = 253
    ten_st = 29
    ten_ed = 57
    km_st = 220
    km_ed = 248
    d_st = 131
    d_ed = 145
    dd = 2
elif type == '小型BAS':
    stand = 42
    ten_st = 32
    ten_ed = 60
    km_st = 220
    km_ed = 248
    d_st = 128
    d_ed = 156
    dd = 1
elif type == '大型BAS':
    stand = 42
    ten_st = 32
    ten_ed = 60
    km_st = 220
    km_ed = 248
    d_st = 128
    d_ed = 156
    dd = 1
elif type == '全回転BAS':
    stand = 42
    ten_st = 22
    ten_ed = 50
    km_st = 220
    km_ed = 248
    d_st = 128
    d_ed = 156
    dd = 1
else:
    stand = 17
    ten_st = 36
    ten_ed = 64
    km_st = 220
    km_ed = 248
    d_st = 128
    d_ed = 156
    dd = 1

if uploaded_files:
    # 配列生成
    n_arrays = []   # 1ワーク分の3次元配列
    for uploaded_file in uploaded_files:
        n_array = func.make_work(uploaded_file, stand)
        n_arrays.append(n_array)
    w_array = np.stack(n_arrays, axis=0)    # 複数ワーク分の4次元配列（同じ針数のワークしか受け付けない）

    # 横軸
    angle = np.arange(0, 360, 1.40625)  # 角度に変換
    x = np.array(range(len(n_array))) # 針数
    xa = np.array(range(len(n_array)*256)) # 全データ
    xw = np.array(range(len(w_array)))  # ワーク数

# メインプログラム
if __name__ == "__main__":
    with tab1:
        if uploaded_files:
            st.write("### ▼張力解析")
            with st.expander("### 区間設定", expanded=False):
                st.write("解析する張力総和区間を設定してください")
                col1, col2 = st.columns(2)
                with col1:
                    stt = st.number_input('開始位相', value=ten_st)
                    st.markdown(f"<span style='font-size: 12px;'>換算</span> {stt * 1.40625}°", unsafe_allow_html=True)  #カウント数を角度に換算
                with col2:
                    end = st.number_input('終了位相', value=ten_ed)
                    st.markdown(f"<span style='font-size: 12px;'>換算</span> {end * 1.40625}°", unsafe_allow_html=True)
                st.write("出会い張力総和区間を設定してください")
                col3, col4 = st.columns(2)
                with col3:
                    dstt = st.number_input('出会い開始位相', value=d_st)
                    st.markdown(f"<span style='font-size: 12px;'>換算</span> {dstt * 1.40625}°", unsafe_allow_html=True)
                with col4:
                    dend = st.number_input('出会い終了位相', value=d_ed)
                    st.markdown(f"<span style='font-size: 12px;'>換算</span> {dend * 1.40625}°", unsafe_allow_html=True)

            with st.expander("### 不良フラグ設定", expanded=False):
                option = st.selectbox('確認する不良1を選択してください', options)
                option2 = st.selectbox('確認する不良2を選択してください', options, index=1) # 初期値を目飛びに設定

            with st.expander("### 速度フラグ設定", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    flagH = st.number_input('加速判定速度', value=100)
                with col2:
                    flagL = st.number_input('減速判定速度', value=100)

            # 総和張力
            sowa = np.arange(len(n_array))  # 1針毎の総和値を入れる配列を作成
            for i in range(len(n_array)):  # 指定区間の総和値をsowa配列に保存する
                sowa[i] = np.sum(w_array[file_no,i,stt:end,6]) - np.sum(w_array[file_no,i,dstt:dend,6])*dd  # 出会い区間128:156

            # 各不良のデータ列
            errorlist = {
                '糸切れ' : 3,
                '目飛び' : 4,
                '締り1' : 5,
                '締り2' : 7,
                '締り3' : 8
            }

            # 不良フラグ
            flag = np.arange(len(n_array))  # 1針毎の総和値を入れる配列を作成
            flag2 = np.arange(len(n_array))
            for i in range(len(n_array)):  # 指定区間の総和値をflag配列に保存する
                flag[i] = np.sum(w_array[file_no,i,:,errorlist[option]])
                flag2[i] = np.sum(w_array[file_no,i,:,errorlist[option2]])

            # 速度フラグ
            speed = np.arange(len(n_array))
            speed[0] = 0  # 0rpm
            speed_dim = np.arange(len(n_array))
            speed_dim[0] = 0
            speed_flag = np.arange(len(n_array))
            for i in range(1,len(n_array)):  # 指定区間の総和値をspeed配列に保存する
                speed[i] = 60000000000 // (clk * (w_array[file_no,i,127,11] - w_array[file_no,i-1,127,11]))  # 10:フリーランカウンタ
                speed_dim[i] = speed[i] - speed[i-1]
                if speed_dim[i] > flagH :
                    speed_flag[i] = 1
                elif speed_dim[i] < 0 and abs(speed_dim[i]) > flagL :
                    speed_flag[i] = -1
                else :
                    speed_flag[i] = 0

            # 全張力横並び表示
            flat = w_array[file_no,:,:,6].flatten()

            # 針数毎の張力を重ねて表示
            data1 = []
            for i in range(len(n_array)):
                random_color = tuple(np.random.choice(range(256), size=3)) # RGBの範囲内でランダムな色を生成
                trace = go.Scatter(
                    x = angle,
                    y = w_array[file_no,i,:,6],
                    mode = 'lines',
                    name = f"{i+1}針",
                    line = dict(color = 'rgb'+str(random_color), width = 2) # 'rgb()'形式の文字列に変換
                )
                data1.append(trace)

            # データの作成
            data2 = go.Scatter(x=xa, y=flat, line=dict(color="mediumblue"), name='Array #{}'.format(i)) # 張力横並び波形
            data3 = go.Scatter(x=x+1, y=sowa, mode='lines+markers', line=dict(color="#757575"), name='総和値'.format(i))    # 総和張力
            data4 = go.Scatter(x=x+1, y=flag, mode='lines+markers', line=dict(color="#FF5722"), name=f'{option}フラグ'.format(i), yaxis='y2')   # 不良1フラグ
            data5 = go.Scatter(x=x+1, y=flag2, mode='lines+markers', line=dict(color="#AA00CC"), name=f'{option2}フラグ'.format(i), yaxis='y2') # 不良2フラグ
            data6 = go.Scatter(x=x+1, y=speed, mode='lines+markers', line=dict(color="#6495ED"), name='縫製速度'.format(i), yaxis='y1') # 縫製速度
            data7 = go.Scatter(x=x+1, y=speed_flag, mode='lines+markers', line=dict(color="#AAA5D1"), name='加減速フラグ'.format(i), yaxis='y2')    # 速度フラグ

            st.write("表示したい波形を選択してください")
            Sowa = st.checkbox('総和値', key='-Sowa-')
            Flag = st.checkbox('不良フラグ', key='-Flag-')
            Speed = st.checkbox('速度フラグ', key='-Speed-')

            # レイアウトの設定
            layout = go.Layout(
                title='Analys',
                xaxis=dict(title='Nd Count'),
                yaxis=dict(title='Tens, speed, flags'),
                showlegend=True
            )
            layout1 = go.Layout(
                title = 'Tension per stitch',
                xaxis = dict(title = 'Angle'),
                yaxis = dict(title = 'Tension'),
                showlegend = True
            )
            layout2 = go.Layout(
                title='Tension per Work',
                xaxis=dict(
                    title='Count',
                    tickvals=list(range(0, len(flat), 256)),  # 横軸の目盛りの位置
                    ticktext=[i+1 for i in range(0,len(flat) // 256)],  # 横軸の目盛りのラベル
                    rangeslider=dict(visible=True)
                ),
                yaxis=dict(title='Tension')
            )

            # data_gに選択された波形データを格納
            data_g = []
            if Sowa == True:
                data_g.extend([data3])
            if Flag == True:
                data_g.extend([data4])
                data_g.extend([data5])
            if Speed == True:
                data_g.extend([data6])
                data_g.extend([data7])

            # フィギュアの作成
            fig3 = go.Figure(data=data_g, layout=layout)
            fig3.update_layout(yaxis1=dict(side='left'), yaxis2=dict(side='right', overlaying = 'y'))
            # フィギュアの表示
            st.plotly_chart(fig3, use_container_width=True)
            # フィギュアの作成
            fig1 = go.Figure(data=data1, layout=layout1)
            # フィギュアの表示
            st.plotly_chart(fig1, use_container_width=True)
            # フィギュアの作成
            fig2 = go.Figure(data=data2, layout=layout2)
            # フィギュアの表示
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.write('### データをアップロードしてください')

    with tab2:
        if uploaded_files:
            st.write("### ▼ワーク間解析")
            with st.expander("### 区間設定", expanded=False):
                st.write("解析する張力総和区間を設定してください")
                col1, col2 = st.columns(2)
                with col1:
                    w_stt = st.number_input('開始位相.', value=ten_st)
                    st.markdown(f"<span style='font-size: 12px;'>換算</span> {w_stt * 1.40625}°", unsafe_allow_html=True)  #カウント数を角度に換算
                with col2:
                    w_end = st.number_input('終了位相.', value=ten_ed)
                    st.markdown(f"<span style='font-size: 12px;'>換算</span> {w_end * 1.40625}°", unsafe_allow_html=True)
                st.write("出会い張力総和区間を設定してください")
                col3, col4 = st.columns(2)
                with col3:
                    w_dstt = st.number_input('出会い開始位相.', value=d_st)
                    st.markdown(f"<span style='font-size: 12px;'>換算</span> {w_dstt * 1.40625}°", unsafe_allow_html=True)
                with col4:
                    w_dend = st.number_input('出会い終了位相.', value=d_ed)
                    st.markdown(f"<span style='font-size: 12px;'>換算</span> {w_dend * 1.40625}°", unsafe_allow_html=True)

            w_num = st.number_input(f'合算時ワーク数を設定してください。アップロード中の総ワーク数：{len(w_array)}', min_value=1, step=1, value=50)
            input_nd = st.number_input(f'何針目を解析しますか？  最大針数：{len(n_array)}針', min_value=1, max_value=len(n_array) , step=1)
            nd = input_nd - 1
            ndsowa = np.arange(len(w_array))  # 各ワークのnd針目の総和値を入れる配列を作成
            for n in range(len(w_array)):  # 指定区間の総和値をsowa配列に保存する
                ndsowa[n] = np.sum(w_array[n,nd,w_stt:w_end,6]) - np.sum(w_array[n,nd,w_dstt:w_dend,6])*dd  # 出会い区間128:156
            # データの作成
            data8 = go.Scatter(x=xw+1, y=ndsowa, mode='lines+markers', line=dict(color="#77AF9C"),  name='総和値'.format(i))
            # レイアウトの作成
            layout8 = go.Layout(
                title='総和張力',
                xaxis=dict(title='Work Count'),
                yaxis=dict(title='Section Tension'),
                showlegend=True
            )

            # ヒストグラムを作成
            hist = px.histogram(ndsowa, nbins=15, title=f'{input_nd}針目のHistogram', labels={'value': 'tension', 'count': 'Frequency'}, color_discrete_sequence=["#77AF9C"])
            # 歪度の計算
            ske = skew(ndsowa)
            # 尖度の計算
            kurt = kurtosis(ndsowa)
            # Shapiro-Wilk検定
            shap = shapiro(ndsowa)
            # Anderson–Darling検定
            ander = anderson(ndsowa)

            # ワーク毎の*針目のグラフを重ねて表示
            # データの作成
            data14 = []
            for i in range(len(w_array)):
                random_color = tuple(np.random.choice(range(256), size=3)) # RGBの範囲内でランダムな色を生成
                trace = go.Scatter(
                    x = angle,
                    y = w_array[i,nd,:,6],
                    mode = 'lines',
                    name = f"{i+1}ワーク目",
                    line = dict(color = 'rgb'+str(random_color), width = 2) # 'rgb()'形式の文字列に変換
                )
                data14.append(trace)
            # レイアウトの作成
            layout14 = go.Layout(
                title = f'各ワークの{input_nd}針目の張力波形',
                xaxis = dict(title = 'Angle'),
                yaxis = dict(title = 'Tension'),
                showlegend = True
            )

            # 標準偏差の移動平均
            std_devs1 = []
            for i in range(len(w_array)-w_num):
                std_dev = np.std(ndsowa[i:i+w_num])  # 標準偏差を計算します
                std_devs1.append(std_dev)  # 結果をリストに追加します
            # データの作成
            data9 = go.Scatter(x=xw+1, y=std_devs1, mode='lines+markers', line=dict(color="#77AF9C"),  name='STD1'.format(i))
            # レイアウトの作成
            layout9 = go.Layout(
                title=f'{w_num}ワーク毎の移動STD',
                xaxis=dict(title='work'),
                yaxis=dict(title='std'),
                showlegend=True
            )

            # 10iワーク毎の標準偏差
            std_devs2 = []
            for i in range(int(len(w_array)/10)):
                std_dev2 = np.std(ndsowa[0:(i+1)*10])  # 標準偏差を計算します
                std_devs2.append(std_dev2)  # 結果をリストに追加します
            # データの作成
            data10 = go.Scatter(x=xw+1, y=std_devs2, mode='lines+markers', line=dict(color="#77AF9C"),  name='STD2'.format(i))
            # レイアウトの作成
            layout10 = go.Layout(
                title='10ワークずつ増やしたSTD',
                xaxis=dict(title='1/10 work'),
                yaxis=dict(title='std'),
                showlegend=True
            )

            # 標準偏差の合算
            std_devs3 = []
            for i in range(1,len(w_array)):
                if i % w_num == 0:
                    std_dev3 = np.std(ndsowa[i-w_num:i-1])  # 標準偏差を計算します
                    #st.write(std_dev3)
                    if std_devs3:
                        last_std_dev3 = std_devs3[-1]
                        std_devs3.append((last_std_dev3 + std_dev3)*0.5)
                    else:
                        std_devs3.append(std_dev3)  # 最初の場合はそのまま追加

            # データの作成
            data11 = go.Scatter(x=xw+1, y=std_devs3, mode='lines+markers', line=dict(color="#77AF9C"),  name='STD3'.format(i))
            # レイアウトの作成
            layout11 = go.Layout(
                title='合算STD',
                xaxis=dict(title='work'),
                yaxis=dict(title='std'),
                showlegend=True
            )

            # フィギュアの作成
            fig8 = go.Figure(data=data8, layout=layout8)
            # フィギュアの表示
            st.plotly_chart(fig8, use_container_width=True)
            # フィギュアの作成
            fig14 = go.Figure(data=data14, layout=layout14)
            # フィギュアの表示
            st.plotly_chart(fig14, use_container_width=True)
            # フィギュアの表示
            st.plotly_chart(hist, use_container_width=True)
            st.write(f'歪度：{ske}（正 : 左寄り、負 : 右寄り、0に近いほど正規性あり）')
            st.write(f'尖度：{kurt}（正 : 尖っている、負 : 尖ってない、0に近いほど正規性あり）')
            st.write(f'Shapiro-Wilk検定 p値 {shap.pvalue}')
            st.write(f'Anderson–Darling検定 statistic = {ander.statistic}')

            with st.expander("標準偏差のグラフ", expanded=False):
                # フィギュアの作成
                fig9 = go.Figure(data=data9, layout=layout9)
                # フィギュアの表示
                st.plotly_chart(fig9, use_container_width=True)
                # フィギュアの作成
                fig10 = go.Figure(data=data10, layout=layout10)
                # フィギュアの表示
                st.plotly_chart(fig10, use_container_width=True)
                # フィギュアの作成
                fig11 = go.Figure(data=data11, layout=layout11)
                # フィギュアの表示
                st.plotly_chart(fig11, use_container_width=True)

        else:
            st.write('### データをアップロードしてください')

    with tab3:
        if uploaded_files:
            st.write("### ▼データ整理")
            with st.expander("### 区間設定", expanded=False):
                st.write("解析する張力総和区間を設定してください")
                col1, col2 = st.columns(2)
                with col1:
                    w_stt = st.number_input('区間1-開始位相', value=ten_st)
                    st.markdown(f"<span style='font-size: 12px;'>換算</span> {w_stt * 1.40625}°", unsafe_allow_html=True)  #カウント数を角度に換算
                with col2:
                    w_end = st.number_input('区間1-終了位相', value=ten_ed)
                    st.markdown(f"<span style='font-size: 12px;'>換算</span> {w_end * 1.40625}°", unsafe_allow_html=True)
                st.write("出会い張力総和区間を設定してください")
                col3, col4 = st.columns(2)
                with col3:
                    w_dstt = st.number_input('出会い開始位相,', value=d_st)
                    st.markdown(f"<span style='font-size: 12px;'>換算</span> {w_dstt * 1.40625}°", unsafe_allow_html=True)
                with col4:
                    w_dend = st.number_input('出会い終了位相,', value=d_ed)
                    st.markdown(f"<span style='font-size: 12px;'>換算</span> {w_dend * 1.40625}°", unsafe_allow_html=True)

            for i in range(len(n_array)):
                x[i] = i+1

            # 総和張力算出
            nwsowa = np.zeros((len(w_array), len(n_array))) # ワークx針数の配列を作成
            nwpeak = np.zeros((len(w_array), len(n_array))) # peak値
            nwphase = np.zeros((len(w_array), len(n_array))) # peak位相
            for w in range(len(w_array)):
                for nd in range(len(n_array)):  # 指定区間の総和値をsowa配列に保存する
                    nwsowa[w, nd] = np.sum(w_array[w,nd,w_stt:w_end,6]) - np.sum(w_array[w,nd,w_dstt:w_dend,6])*dd  # 出会い区間128:156
                    nwpeak[w, nd] = np.amax(w_array[w,nd,w_stt:w_end,6])
                    nwphase[w, nd] = np.argmax(w_array[w,nd,w_stt:w_end,6]) * 1.40625 + w_stt * 1.40625

            std = np.zeros(len(n_array))        # 標準偏差算出
            ave = np.zeros(len(n_array))        # 平均張力算出
            kurt_box = np.zeros(len(n_array))   # 尖度の計算
            skew_box = np.zeros(len(n_array))   # 歪度の算出
            shap_box = np.zeros(len(n_array))   # Shapiro-Wilk検定
            ander_box = np.zeros(len(n_array))  # Anderson–Darling検定
            # boxcox変換
            nw_boxcox = np.zeros((len(n_array), len(w_array)))
            shap_boxcox = np.zeros(len(n_array))
            ander_boxcox = np.zeros(len(n_array))
            for i in range(len(n_array)):
                std[i] = np.std(nwsowa[:,i])
                ave[i] = np.mean(nwsowa[:,i])
                kurt_box[i] = kurtosis(nwsowa[:,i])
                skew_box[i] = skew(nwsowa[:,i])
                shap_box[i] = shapiro(nwsowa[:,i]).pvalue
                ander_box[i] = anderson(nwsowa[:,i]).statistic
                transformed_data, _ = boxcox(nwsowa[:, i])
                nw_boxcox[i, :] = transformed_data
                shap_boxcox[i] = shapiro(nw_boxcox[i,:]).pvalue
                ander_boxcox[i] = anderson(nw_boxcox[i,:]).statistic

            # データ一覧化
            joho = []
            joho_peak = []
            joho_phase = []
            for i in range(len(w_array)):
                row = []
                row_peak = []
                row_phase = []
                for j in range(len(n_array)):
                    row.append(nwsowa[i,j])
                    row_peak.append(nwpeak[i,j])
                    row_phase.append(nwphase[i,j])
                joho.append(row)
                joho_peak.append(row_peak)
                joho_phase.append(row_phase)
            joho_peak.insert(0,x)
            joho_phase.insert(0,x)
            joho.insert(0,x)
            joho.insert(1,ave)
            joho.insert(2,std)
            joho.insert(3,skew_box)
            joho.insert(4,kurt_box)
            joho.insert(5,shap_box)
            joho.insert(6,shap_boxcox)
            joho.insert(7,ander_box)
            joho.insert(8,ander_boxcox)
            df = pd.DataFrame(joho)
            df_peak = pd.DataFrame(joho_peak)
            df_phase = pd.DataFrame(joho_phase)
            new_index = ['針数', '平均張力', '標準偏差','歪度', '尖度', 'shapiro', 'boxcox_shap', 'anderson', 'boxcox_ander'] + [str(i) + 'ワーク目' for i in range(1, len(df)+1)]
            new_index_p = ['針数'] + [str(i) + 'ワーク目' for i in range(1, len(df)+1)]
            df.index = new_index[:len(df.index)]  # 新しいインデックスを設定する
            df_peak.index = new_index_p[:len(df_peak.index)]
            df_phase.index = new_index_p[:len(df_phase.index)]

            k = st.number_input('閾値係数k（青：閾値より小さい　赤：閾値より大きい）', value=3.0)
            upper_threshold = ave + (std * k)
            lower_threshold = ave - (std * k)
            styled_df = df.style.apply(lambda x: [func.color_by_threshold(val, upper_threshold[x.name], lower_threshold[x.name]) for val in x])
            # データ一覧
            with st.expander("### データ一覧", expanded=False):
                st.write(styled_df)
            # ピーク張力と位相
            with st.expander("### ピーク張力と位相", expanded=False):
                st.write(df_peak)
                st.write(df_phase)

            # csv出力
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="CSVファイルをダウンロード",
                data=csv_data,
                file_name='data.csv',
                mime='text/csv'
            )
            csv_data_peak = df_peak.to_csv(index=False)
            st.download_button(
                label="peak張力のCSVファイルをダウンロード",
                data=csv_data_peak,
                file_name='data_peak.csv',
                mime='text/csv'
            )
            csv_data_phase = df_phase.to_csv(index=False)
            st.download_button(
                label="peak位相のCSVファイルをダウンロード",
                data=csv_data_phase,
                file_name='data_phase.csv',
                mime='text/csv'
            )

        else:
            st.write('### データをアップロードしてください')

    with tab4:
        if uploaded_files:
            st.write("### ▼3Dグラフ")
            with st.expander("### 区間設定", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    s_stt = st.number_input('.開始位相', value=ten_st)
                    st.markdown(f"<span style='font-size: 12px;'>換算</span> {s_stt * 1.40625}°", unsafe_allow_html=True)  #カウント数を角度に換算
                with col2:
                    s_end = st.number_input('.終了位相', value=ten_ed)
                    st.markdown(f"<span style='font-size: 12px;'>換算</span> {s_end * 1.40625}°", unsafe_allow_html=True)
                st.write("出会い張力総和区間を設定してください")
                col3, col4 = st.columns(2)
                with col3:
                    s_dstt = st.number_input('.出会い開始位相', value=d_st)
                    st.markdown(f"<span style='font-size: 12px;'>換算</span> {s_dstt * 1.40625}°", unsafe_allow_html=True)
                with col4:
                    s_dend = st.number_input('.出会い終了位相', value=d_ed)
                    st.markdown(f"<span style='font-size: 12px;'>換算</span> {s_dend * 1.40625}°", unsafe_allow_html=True)
            with st.expander("### 表示設定", expanded=False):
                st.write(f"表示するワーク数を設定してください。現在アップロード数：{len(w_array)}ワーク")
                col5, col6 = st.columns(2)
                with col5:
                    n_s = st.number_input('開始ワーク数', value=1, min_value=1, max_value=len(w_array) , step=1)
                with col6:
                    n_e = st.number_input('終了ワーク数', value=len(w_array), min_value=1, max_value=len(w_array) , step=1)
                st.write(f"表示する針数を設定してください。最大針数：{len(n_array)}針")
                col7, col8 = st.columns(2)
                with col7:
                    nd_s = st.number_input('開始針数', value=1, min_value=1, max_value=len(n_array) , step=1)
                with col8:
                    nd_e = st.number_input('終了針数', value=len(n_array), min_value=1, max_value=len(n_array) , step=1)
                wn = st.number_input('ここで指定したワーク毎に標準偏差を計算する', value=1)

            xsowa = []
            ysowa = []
            zsowa = []
            xstd = []
            ystd = []
            zstd = []

            wsowa = np.zeros((len(n_array),len(w_array)))
            for n in range(n_s-1,n_e):
                for nd in range(nd_s-1,nd_e):
                    xsowa.append(n)
                    ysowa.append(nd)
                    wsowa[nd][n] = np.sum(w_array[n,nd,s_stt:s_end,6]) - np.sum(w_array[n,nd,s_dstt:s_dend,6])*dd
                    zsowa.append(wsowa[nd][n])

            wstd = np.arange(len(n_array))
            std_10 = np.zeros((len(n_array),len(w_array)//10))
            for p in range(n_s-1,n_e,wn):
                for q in range(nd_s-1,nd_e):
                    xstd.append(p)
                    ystd.append(q)
                    zstd.append(np.std(wsowa[q][0:p]))

            with st.expander("### 閾値設定", expanded=False):
                # 任意の閾値
                col1, col2 = st.columns(2)
                with col1:
                    th_sowa_u = st.number_input('総和張力の上限閾値を設定', value=3000)
                    th_sigma_u = st.number_input('総和張力の上限閾値を設定', value=100)
                with col2:
                    th_sowa_t = st.number_input('総和張力の加減閾値を設定', value=6000)
                    th_sigma_t = st.number_input('総和張力の加減閾値を設定', value=1000)

            # 色を変更する条件を指定
            color_sowa = ["#77AF9C" if th_sowa_u <= z_val <= th_sowa_t else 'red' for z_val in zsowa]
            color_sigma = ['#87ceeb' if th_sigma_u <= z_val < th_sigma_t else 'red' for z_val in zstd]

            # データの作成
            data12 = go.Scatter3d(
                x=xsowa,
                y=ysowa,
                z=zsowa,
                mode='markers',
                marker=dict(color=color_sowa,size=2),
                showlegend=True)
            # レイアウトの作成
            layout12 = go.Layout(
                title='総和張力',
                scene = dict(
                    xaxis_title='work',
                    yaxis_title='needle',
                    zaxis_title='Tension'
                    ))
            # フィギュアの作成
            fig12 = go.Figure(data=data12, layout=layout12)
            # フィギュアの表示
            st.plotly_chart(fig12, use_container_width=True)

            # データの作成
            data13 = go.Scatter3d(
                x=xstd,
                y=ystd,
                z=zstd,
                mode='markers',
                marker=dict(color=color_sigma,size=2),
                showlegend=True)
            # レイアウトの作成
            layout13 = go.Layout(
                title=f'{wn}ワーク毎の標準偏差',
                scene = dict(
                    xaxis_title='(n)work',
                    yaxis_title='needle',
                    zaxis_title='STD'
                    ))
            # フィギュアの作成
            fig13 = go.Figure(data=data13, layout=layout13)
            # フィギュアの表示
            st.plotly_chart(fig13, use_container_width=True)
