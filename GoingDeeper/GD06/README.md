# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 염철헌
- 리뷰어 : 오창원


# PRT(Peer Review Template)
- [ ]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
        - 원하는 최종결과로 데이터 전처리부터 모델, 평가까지 결과물들이 정리되어 있습니다.
        - ![image](https://github.com/user-attachments/assets/0dc27c38-3066-4807-8122-1f0e9ac54cce)

    
- [ ]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭을 왜 핵심적이라고 생각하는지 확인
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드의 기능, 존재 이유, 작동 원리 등을 기술했는지 확인
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 데이터 증강과 모델 학습하는 부분이 가장 중요했다고 생각합니다.
        - 데이터 증강으로 데이터를 늘리는 것과 train_step에서 토크나이저가 바뀌면서 수정할 부분이 가장 많은 시간이 들었습니다.
        - ![image](https://github.com/user-attachments/assets/ea8b6799-461c-4cd0-84b2-49a848412b4f)
        - ![image](https://github.com/user-attachments/assets/9d0df76a-76c3-463c-9879-1923b2af35bb)


        
- [ ]  **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나
새로운 시도 또는 추가 실험을 수행해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 프로젝트 평가 기준에 더해 추가적으로 수행한 나만의 시도, 
    실험이 기록되어 있는지 확인
        - 유의어 대체, 임의의 단어 삭제, 임의의 단어 추가 증강기법으로 추가 사용하셨습니다.
        - ![image](https://github.com/user-attachments/assets/82b965f6-07fc-4213-ac2f-a9d516da022d)
        - 또 시드를 다르게 할 때 학습 분포가 어떻게 바뀌는지도 실험하셨어요
        - ![image](https://github.com/user-attachments/assets/d4c17bec-b4d9-4054-b838-91aacaea5751)

        
- [ ]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 회고 깔끔하게 잘 정리해주셨습니다.
        - ![image](https://github.com/user-attachments/assets/b6d54f88-07e6-4554-841c-e79e517b1b63)

        
- [ ]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화/모듈화했는지 확인
        - 코드가 간결하고 모듈화 되어 있습니다.
        - ![image](https://github.com/user-attachments/assets/5e8717e2-61fe-4e95-8bbb-cc6b6ded5e34)



# 회고(참고 링크 및 코드 개선)
```
철헌님과 이야기 나누면 늘 뭔가를 배우게 됩니다 ㅎㅎ 앞으로도 많이 알려주십셔
코드 만드느라 고생하셨습니다.
```
