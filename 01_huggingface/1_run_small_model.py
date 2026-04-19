# 파이프라인은 뭔가? 모델을 쉽게 사용할 수 있는 도구이다.
# 그 모델이란 토큰화, 모델 호출, 출력 후 토큰화 하는 것을 포함한 모든 과정을 말한다.
# 더 자세히 말해보자면, 모델 호출 전에 토큰화를 하고, 모델 호출 후에 출력을 토큰화 하는 것을 말한다.
# 토큰화는 뭐냐면 문자열을 모델에 넣을 수 있는 형태로 변환하는 것을 말한다.
# 예를 들어서 "Hello, world!" 라는 문자열을 모델에 넣을 수 있는 형태로 변환하는 것을 토큰화라고 한다.
# 출력 후 토큰화는 모델의 출력을 문자열로 변환하는 것을 말한다.
from transformers import pipeline

# 아래의 코드에 대한 설명은 다음과 같다.
# classifier = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
# 이 코드는 감성 분석 모델을 사용하여 문자열의 감성을 분석하는 파이프라인을 생성한다.
# "sentiment-analysis"는 감성 분석 모델의 이름이다.

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
)

texts = [
    "This accelerator is amazingly efficient.",
    "The latency is terrible and the software stack is messy."
]

outputs = classifier(texts)
print(outputs)