```
federated-regressive-learning/
├── README.md
├── LICENSE
├── CITATION.cff
├── .gitignore
├── pyproject.toml              # 혹은 setup.cfg / requirements.txt (env/에 둬도 OK)
├── .github/
│   └── workflows/
│       └── ci.yml              # Lint + unit-test + smoke-run
├── env/
│   ├── requirements.txt
│   └── environment.yml
├── docker/
│   └── Dockerfile
├── frl/                        # 라이브러리(패키지) 본체
│   ├── __init__.py
│   ├── algo_frl.py             # FRL 핵심 알고리즘(가중 계산·집계)
│   ├── baselines.py            # FedAvg/FedProx 등 비교기법 래퍼
│   ├── aggregation.py          # 서버 집계 로직(β 정규화 등)
│   ├── metrics.py              # Acc/Prec/Rec/F1, ECE, 거리통계 등
│   ├── utils.py                # 시드 고정, 로깅, 공통 유틸
│   ├── data/
│   │   ├── loaders.py          # MNIST/CIFAR-10 등 공개 데이터 로더
│   │   ├── ugei_placeholder.py # UGEI 인터페이스 스텁(절대 데이터 포함 금지)
│   │   └── transforms.py
│   └── scenarios/
│       ├── scenario_gen.py     # S1/S2/S3 생성기(계층샘플링, 클래스 결손 등)
│       ├── s1_equal_dist_diff_size.yaml
│       ├── s2_hetero_dist_diff_size.yaml
│       └── s3_class_missing.yaml
├── configs/                    # 실험 설정 묶음
│   ├── train_default.yaml      # 공통 학습 하이퍼파라미터
│   ├── optimizers.yaml         # Adam/SGDM 등 설정
│   ├── datasets.yaml           # 공개 데이터셋 경로·다운로드 옵션
│   └── scenarios.yaml          # 기본 시나리오 선택·파라미터
├── scripts/                    # 실행/재현 스크립트
│   ├── run_federated.py        # 통합 러너(CLI 인자로 시나리오/모델 선택)
│   ├── make_fig_tables.py      # 결과 요약표·플롯 자동 생성
│   └── reproduce_all.sh        # 대표 재현 파이프라인
├── examples/
│   ├── mnist_s1/               # 튜토리얼 노트북·명령 예시
│   └── cifar10_s2/
├── docs/
│   ├── REPRODUCIBILITY.md      # 환경 고정, 시드, 하드웨어, 버전 고지
│   ├── SCENARIOS.md            # S1/S2/S3 정의·생성 규칙·분포 벡터 표
│   ├── PRIVACY.md              # UGEI 비공개 원칙·법적 주의 문구
│   └── CHANGELOG.md
├── tests/
│   ├── test_scenario_gen.py    # 시나리오 생성 단위 테스트
│   ├── test_algo_frl.py        # FRL 가중/집계 단위 테스트
│   └── test_integration.py     # 소규모 통합 스모크 테스트
├── results/                    # 자동 생성(로그·표·그림)
│   ├── logs/.gitkeep
│   └── figures/.gitkeep
└── data/                       # 로컬 전용(절대 커밋 금지)
    └── .gitkeep
```
