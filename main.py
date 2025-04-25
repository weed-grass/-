import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import openai

st.set_page_config(page_title="안테나 QAQC 대시보드", layout="wide")



class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_dim=31, hidden_dim=64, latent_dim=32, output_dim=4):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.latent = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, seq=1, feature]
        _, (hidden, _) = self.encoder(x)
        z = self.latent(hidden[-1])
        return self.decoder(z)

if "openai_api_key" not in st.session_state:
    key = st.text_input(
        "OpenAI API Key를 입력하세요",
        type="password",
        help="*입력하신 키는 세션에만 저장되며, 코드나 로그에 남지 않습니다.*"
    )
    if key:
        st.session_state.openai_api_key = key

# 2) 입력된 키를 openai.api_key 에 설정
if "openai_api_key" in st.session_state:
    openai.api_key = st.session_state.openai_api_key
else:
    st.warning("OpenAI API Key를 입력하시면 자동 인사이트 기능을 사용할 수 있습니다.")
@st.cache_data(show_spinner=False)
def generate_insight(variable_name: str,
                     normal_stats: dict,
                     defect_stats: dict) -> str:
    prompt = (
        f"‘{variable_name}’ 변수에 대해 아래 두 가지를 **번호와 소제목**으로 구분하여, "
        f"각 항목마다 **반드시 줄바꿈**해서 작성해 주세요.\n\n"
        f"1) 정상 vs 불량 분포 차이가 특히 **어떤 값 영역**에서 두드러지는지\n"
        f"2) 이 차이를 줄이려면 **공정의 어느 부분**을 점검하거나 조정해야 할지\n\n"
        f"정상 – 평균 {normal_stats['mean']:.2f}, 표준편차 {normal_stats['std']:.2f}\n"
        f"불량 – 평균 {defect_stats['mean']:.2f}, 표준편차 {defect_stats['std']:.2f}"
    )

    try:
        resp = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system",
                 "content": "당신은 제조 품질 데이터를 분석하여 간결하고 명확한 인사이트를 제공하는 한국어 전문가입니다."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=500,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"인사이트 생성 중 오류가 발생했습니다: {e}"


# ▶ train.csv 업로드 받기
uploaded_train = st.file_uploader(
    "▶ train.csv 파일을 업로드하세요",
    type="csv",
    help="로컬에 있는 train.csv를 선택하면 자동으로 읽어들입니다."
)
if uploaded_train is None:
    st.warning("분석을 위해 반드시 train.csv를 업로드해야 합니다.")
    st.stop()
else:
    train = pd.read_csv(uploaded_train)

# ▶ train_eda.csv 업로드 받기 (모델 예측 페이지나 EDA 페이지에서 쓰신다면)
uploaded_eda = st.file_uploader(
    "▶ train_eda.csv 파일을 업로드하세요",
    type="csv",
    help="모델 예측이나 EDA용으로 사용됩니다."
)
if uploaded_eda is None:
    st.warning("train_eda.csv를 업로드해 주세요.")
    st.stop()
else:
    train_eda = pd.read_csv(uploaded_eda)

# 컬럼 한글명 기준 스펙 딕셔너리
spec_dict = {
    '안테나 Gain 평균(각도1)': (0.2, 2),
    '안테나 1 Gain 편차': (0.2, 2.1),
    '안테나 2 Gain 편차': (0.2, 2.1),
    '평균 신호대 잡음비 (SNR)': (7, 19),
    '안테나 Gain 평균(각도2)': (22, 36.5),
    '신호대 잡음비 (각도1)': (-19.2, 19),
    '안테나 Gain 평균(각도3)': (2.4, 4),
    '신호대 잡음비 (각도2)': (-29.2, -24),
    '신호대 잡음비 (각도3)': (-29.2, -24),
    '신호대 잡음비 (각도4)': (-30.6, -20),
    '안테나 Gain 평균(각도4)': (19.6, 26.6),
    '신호대 잡음비 (각도5)': (-29.2, -24),
    '신호대 잡음비 (각도6)': (-29.2, -24),
    '신호대 잡음비 (각도7)': (-29.2, -24)
}

# 불량 판정
defect_mask = pd.DataFrame(index=train.index)
for col, (min_val, max_val) in spec_dict.items():
    if col in train.columns:
        defect_mask[col] = (train[col] < min_val) | (train[col] > max_val)

# 불량 여부 컬럼 추가 (불량 여부는 하나라도 True인 경우)
train["불량여부"] = defect_mask.any(axis=1)

# 사이드바에서 페이지 선택
page = st.sidebar.selectbox("페이지 선택", ["홈", "불량 개수 확인","이상 탐지", "X 변수 분포 분석", "모델 예측 페이지"])


if page == "홈":
    st.image("썸네일.png", use_container_width=True)
    # 2) 이미지 태그 CSS를 덮어쓰기
    st.markdown(
        """
        <style>
        /* <img> 태그가 컨테이너 폭 100%를 사용하도록 강제 */
        .stImage > img {
            width: 100% !important;
            height: auto !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

elif page == "이상 탐지":
    st.title("이상치 탐지")

    # 1) 몇 개의 행을 볼지 선택
    sample_size = st.slider("몇 개의 행을 샘플링할까요?", min_value=1, max_value=10, value=3)

    # 2) 전체 이상치 마스크
    anomaly_mask = pd.DataFrame(False, index=train.index, columns=spec_dict.keys())
    for col, (mn, mx) in spec_dict.items():
        if col in train.columns:
            anomaly_mask[col] = (train[col] < mn) | (train[col] > mx)

    # 3) 이상치가 한 행이라도 있는 인덱스만 추출
    anomaly_indices = train.index[anomaly_mask.any(axis=1)].tolist()
    if not anomaly_indices:
        st.success("이상치가 없습니다!")
    else:
        # 4) 랜덤 샘플링
        sampled = np.random.choice(anomaly_indices,
                                   size=min(len(anomaly_indices), sample_size),
                                   replace=False)

        # 5) 샘플 행 각각에서 벗어난 컬럼만 기록
        records = []
        for idx in sampled:
            for col, (mn, mx) in spec_dict.items():
                if col in train.columns:
                    val = train.at[idx, col]
                    if val < mn or val > mx:
                        records.append({
                            "row_index": idx,
                            "column": col,
                            "value": val,
                            "spec_min": mn,
                            "spec_max": mx
                        })

        # 6) 결과 보여주기
        st.subheader(f"샘플 {len(sampled)}개 행의 이상치")
        st.table(pd.DataFrame(records))

elif page == "불량 개수 확인":
    # 불량 개수
    defect_counts = defect_mask.sum().sort_values(ascending=False)

    # Streamlit UI
    st.title("안테나 성능 불량 판정 대시보드")

    st.subheader("항목별 불량 개수")
    st.bar_chart(defect_counts)

    # 선택 항목
    selected_col = st.selectbox("상세 분석 항목을 선택하세요:", list(spec_dict.keys()))
    min_val, max_val = spec_dict[selected_col]

    # 히스토그램 시각화
    fig, ax = plt.subplots()
    ax.hist(train[selected_col], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(min_val, color='red', linestyle='--', label='Spec Min')
    ax.axvline(max_val, color='red', linestyle='--', label='Spec Max')
    ax.set_title(f"{selected_col} 값 분포")
    ax.legend()
    st.pyplot(fig)

    # 불량 수 출력
    st.markdown(f"**불량 샘플 수: {defect_mask[selected_col].sum()} 건**")



elif page == "X 변수 분포 분석":
    # X 변수 분포 분석 페이지
    st.write("X 변수 분포 분석 페이지")

    # X 변수 그룹화
    x_groups = {
        "PCB 체결 관련 변수": [
            'PCB 체결 시 누름량 (Step 1)', 'PCB 체결 시 누름량 (Step 2)', 'PCB 체결 시 누름량 (Step 3)', 'PCB 체결 시 누름량 (Step 4)',
        ],
        "방열 재료 관련 변수": [
            '방열 재료 1 무게', '방열 재료 1 면적', '방열 재료 2 면적', '방열 재료 3 면적',
        ],
        "안테나 관련 변수": [
            '안테나 패드 높이 차이', '1번 안테나 패드 위치', '2번 안테나 패드 위치', '3번 안테나 패드 위치', '4번 안테나 패드 위치', '5번 안테나 패드 위치',
        ],
        "커넥터 관련 변수": [
            '커넥터 1번 핀 치수', '커넥터 2번 핀 치수', '커넥터 3번 핀 치수', '커넥터 4번 핀 치수', '커넥터 5번 핀 치수', '커넥터 6번 핀 치수',
        ],
        "스크류 관련 변수": [
            '스크류 체결 RPM 1', '스크류 체결 RPM 2', '스크류 체결 RPM 3', '스크류 체결 RPM 4',
        ],
        "하우징 관련 변수": [
            '하우징 PCB 안착부 1 치수', '하우징 PCB 안착부 2 치수', '하우징 PCB 안착부 3 치수',
        ],
        "레이돔 관련 변수": [
            '안테나 부분 레이돔 기울기','레이돔 치수 (안테나1)', '레이돔 치수 (안테나2)', '레이돔 치수 (안테나3)', '레이돔 치수 (안테나4)',
        ],
        "기타 변수": [
            '실란트 본드 소요량', 'Cal 투입 전 대기 시간', 'RF1 SMT 납량', 'RF2 SMT 납량',
            'RF3 SMT 납량', 'RF4 SMT 납량', 'RF5 SMT 납량', 'RF6 SMT 납량', 'RF7 SMT 납량',
        ]
    }

    # 그룹 선택
    selected_group = st.selectbox("그룹을 선택하세요", list(x_groups.keys()))

    # 선택된 그룹에서 변수 선택
    selected_x = st.selectbox("분포를 보고 싶은 X 변수를 선택하세요", x_groups[selected_group])

    # 한 번에 여러 그래프를 나란히 표시하기 위한 subplot 설정
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # 2개의 서브플롯을 1행에 나란히 배치

    # 정상 데이터 분포
    ax[0].hist(train.loc[~train["불량여부"], selected_x], bins=40, color='skyblue', alpha=0.7, label='정상')
    ax[0].set_title(f"정상: {selected_x}")
    ax[0].set_xlabel(selected_x)
    ax[0].set_ylabel("샘플 수")

    # 불량 데이터 분포
    ax[1].hist(train.loc[train["불량여부"], selected_x], bins=40, color='salmon', alpha=0.7, label='불량')
    ax[1].set_title(f"불량: {selected_x}")
    ax[1].set_xlabel(selected_x)
    ax[1].set_ylabel("샘플 수")

    # 플롯 설정
    plt.tight_layout()  # 서브플롯 간의 간격 자동 조정

    st.pyplot(fig)  # Streamlit에서 그래프 출력
    normal = train.loc[~train["불량여부"], selected_x]
    defect = train.loc[ train["불량여부"], selected_x]
    stats_normal = {"mean": normal.mean(), "std": normal.std()}
    stats_defect = {"mean": defect.mean(), "std": defect.std()}
    if st.button("분석 시작"):
        st.markdown("----")
        st.subheader("분포도 분석")
        with st.spinner("GPT에게 분석 중…"):
            insight = generate_insight(selected_x, stats_normal, stats_defect)

            
        st.text_area("", insight, height=200, disabled=True)    


elif page == "모델 예측 페이지":
    st.title("모델 예측 페이지")

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"Using device: {device}")

    # 모델 정의 & 로드 (학습 때 쓰던 파라미터와 완전 일치!)
    class LSTMAutoEncoder(nn.Module):
        def __init__(self, input_dim, hidden_dim=64, latent_dim=32, output_dim=4):
            super().__init__()
            self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.latent  = nn.Linear(hidden_dim, latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        def forward(self, x):
            x = x.unsqueeze(1)
            _, (h, _) = self.encoder(x)
            z = self.latent(h[-1])
            return self.decoder(z)

    model = LSTMAutoEncoder(input_dim=31, hidden_dim=64, latent_dim=32, output_dim=4).to(device)
    model.load_state_dict(torch.load('lstm_autoencoder.pth', map_location=device))
    model.eval()


    # 텐서로 변환 & 예측
    X_tensor = torch.tensor(X_df.values, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy()

    # 실제 Y값 불러오기
    y_cols = ['Gain_CV','SNR_CV','Gain_Spread','SNR_Spread']
    true_df = pd.read_csv("train_eda.csv")[y_cols]

    # 결과 DataFrame
    result_df = pd.DataFrame(y_pred, columns=y_cols)
    compare_df = pd.concat([true_df.reset_index(drop=True),
                            result_df.reset_index(drop=True)],
                           axis=1,
                           keys=["Actual", "Predicted"])
    st.subheader("실제값 vs 예측값 (첫 20개 샘플)")
    st.dataframe(compare_df.head(20))

    # 시각화: 각 Y별 실제와 예측을 겹쳐서 그리기 (첫 100개 샘플)
    st.subheader("실제 vs 예측 라인 차트 (첫 100개)")
    fig, ax = plt.subplots(figsize=(10,5))
    for col in y_cols:
        ax.plot(true_df[col].iloc[:100].values, label=f"{col} 실제")
        ax.plot(result_df[col].iloc[:100].values, '--', label=f"{col} 예측")
    ax.set_xlabel("샘플 인덱스")
    ax.set_ylabel("값")
    ax.legend()
    st.pyplot(fig)

elif page == "모델 예측 페이지":
    st.title("모델 예측 페이지")

    # 1) 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"Using device: {device}")

    # 2) LSTM 모델 정의 (학습 때 구조와 완전히 동일하게)
    class LSTMAutoEncoder(nn.Module):
        def __init__(self, input_dim, hidden_dim=64, latent_dim=32, output_dim=4):
            super().__init__()
            self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.latent  = nn.Linear(hidden_dim, latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        def forward(self, x):
            x = x.unsqueeze(1)
            _, (h, _) = self.encoder(x)
            z = self.latent(h[-1])
            return self.decoder(z)

    model = LSTMAutoEncoder(input_dim=31, hidden_dim=64, latent_dim=32, output_dim=4).to(device)
    model.load_state_dict(torch.load('lstm_autoencoder.pth', map_location=device))
    model.eval()

    # 3) 입력 데이터 로드
    X_df = pd.read_csv("X_train.csv").drop(columns=["Unnamed: 0"])

    # 4) 예측
    X_tensor = torch.tensor(X_df.values, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy()

    # 5) DataFrame 화 & 슬라이싱
    full_y_cols = ['Gain_CV','SNR_CV','Gain_Spread','SNR_Spread']
    pred_full = pd.DataFrame(y_pred, columns=full_y_cols)

    # **오직 SNR만 선택**
    y_cols = ['SNR_CV','SNR_Spread']
    pred = pred_full[y_cols]

    # **디버그: pred.columns 가 제대로 나오는지 확인**
    st.write("예측 DataFrame 컬럼:", pred.columns.tolist())  
    st.write(pred.head())

    # 6) 실제값 불러오기 & 동일 슬라이싱
    true = pd.read_csv("train_eda.csv")[y_cols]
    st.write("실제 DataFrame 컬럼:", true.columns.tolist())
    st.write(true.head())

    # 7) 비교 및 시각화
    st.subheader("실제 vs 예측 차트 (첫 100개 샘플)")
    fig, ax = plt.subplots(figsize=(8,4))
    for col in y_cols:
        ax.plot(true[col].values[:100],    label=f"{col} 실제")
        ax.plot(pred[col].values[:100], '--', label=f"{col} 예측")
    ax.legend()
    st.pyplot(fig)

