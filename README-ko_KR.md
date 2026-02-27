<div align="center">
<!--<h1 style="display: flex; align-items: center; gap: 10px;">
  <img src="https://github.com/MrOdd-Use/Auto_Research_Engine/assets/13554167/a45bac7c-092c-42e5-8eb6-69acbf20dde5" alt="Logo" width="25">
  Auto_Research_Engine
</h1>-->
<img src="https://github.com/MrOdd-Use/Auto_Research_Engine/assets/13554167/20af8286-b386-44a5-9a83-3be1365139c3" alt="Logo" width="80">


####

[![Website](https://img.shields.io/badge/Official%20Website-gptr.dev-teal?style=for-the-badge&logo=world&logoColor=white&color=0891b2)](https://gptr.dev)
[![Documentation](https://img.shields.io/badge/Documentation-DOCS-f472b6?logo=googledocs&logoColor=white&style=for-the-badge)](https://docs.gptr.dev)
[![Discord Follow](https://img.shields.io/discord/1127851779011391548?style=for-the-badge&logo=discord&label=Chat%20on%20Discord)](https://discord.gg/QgZXvJAccX)

[![PyPI version](https://img.shields.io/pypi/v/Auto_Research_Engine?logo=pypi&logoColor=white&style=flat)](https://badge.fury.io/py/Auto_Research_Engine)
![GitHub Release](https://img.shields.io/github/v/release/MrOdd-Use/Auto_Research_Engine?style=flat&logo=github)
[![Open In Colab](https://img.shields.io/static/v1?message=Open%20in%20Colab&logo=googlecolab&labelColor=grey&color=yellow&label=%20&style=flat&logoSize=40)](https://colab.research.google.com/github/MrOdd-Use/Auto_Research_Engine/blob/master/docs/docs/examples/pip-run.ipynb)
[![Docker Image Version](https://img.shields.io/docker/v/elestio/Auto_Research_Engine/latest?arch=amd64&style=flat&logo=docker&logoColor=white&color=1D63ED)](https://hub.docker.com/r/gptresearcher/Auto_Research_Engine)
[![Twitter Follow](https://img.shields.io/twitter/follow/assaf_elovic?style=social)](https://twitter.com/assaf_elovic)

[English](README.md) |
[中文](README-zh_CN.md) |
[日本語](README-ja_JP.md) |
[한국어](README-ko_KR.md)
</div>

# 🔎 Auto_Research_Engine

**Auto_Research_Engine�?다양�?작업�?대�?포괄적인 온라�?연구�?수행하도�?설계�?자율 에이전트입니�?**

�?에이전트�?세부적이�?사실�?기반하며 편견 없는 연구 보고서를 생성�?�?있으�? 관�?리소스와 개요�?초점�?맞춘 맞춤�?옵션�?제공합니�?  최근 발표�?[Plan-and-Solve](https://arxiv.org/abs/2305.04091) �?[RAG](https://arxiv.org/abs/2005.11401) 논문에서 영감�?받아 Auto_Research_Engine�?잘못�?정보, 속도, 결정론적 접근 방식, 신뢰�?문제�?해결하고, 동기�?작업�?아닌 병렬 에이전트 작업�?통해 �?안정적이�?빠른 성능�?제공합니�?

**우리�?목표�?AI�?힘을 활용하여 개인�?조직에게 정확하고 편향 없는 사실�?기반�?정보�?제공하는 것입니다.**

## �?Auto_Research_Engine인가?

- 직접 수행하는 연구 과정은 객관적인 결론�?도출하는 �?시간�?오래 걸리�? 적절�?리소스와 정보�?찾는 �?�?주가 걸릴 �?있습니다.
- 현재�?대규모 언어 모델(LLM)은 과거 정보�?기반�?훈련되었으며, 환각 현상�?발생�?위험�?높아 연구 작업에는 적합하지 않습니다.
- 현재 LLM은 짧은 토큰 출력으로 제한되며, 2,000단어 이상�?길고 자세�?연구 보고서를 작성하는 데는 충분하지 않습니다.
- �?검색을 지원하�?서비�?�? ChatGPT 또는 Perplexity)�?제한�?리소스와 콘텐츠만�?고려하여 경우�?따라 피상적이�?편향�?답변�?제공합니�?
- �?소스만을 사용하면 연구 작업에서 올바�?결론�?도출�?�?편향�?발생�?�?있습니다.

## 데모
https://github.com/user-attachments/assets/092e9e71-7e27-475d-8c4f-9dddd28934a3

## 아키텍처
주요 아이디어�?"플래�?와 "실행" 에이전트�?실행하는 것으�? 플래너는 연구�?질문�?생성하고, 실행 에이전트�?생성�?�?연구 질문�?따라 가�?관련성 높은 정보�?찾습니다. 마지막으�?플래너는 모든 관�?정보�?필터링하�?집계하여 연구 보고서를 작성합니�?
<br /> <br /> 
에이전트�?`gpt-4o-mini`와 `gpt-4o`(128K 컨텍스트)�?활용하여 연구 작업�?완료합니�? 필요�?따라 각각�?사용하여 비용�?최적화합니다. **평균 연구 작업은 �?2분이 소요되며, 비용은 �?$0.005입니�?**.

<div align="center">
<img align="center" height="600" src="https://github.com/MrOdd-Use/Auto_Research_Engine/assets/13554167/4ac896fd-63ab-4b77-9688-ff62aafcc527">
</div>

구체적으�?
* 연구 쿼리 또는 작업�?기반으로 도메인별 에이전트�?생성합니�?
* 주어�?작업�?대�?객관적인 의견�?형성�?�?있는 일련�?연구 질문�?생성합니�?
* �?연구 질문�?대�?크롤�?에이전트�?실행하여 작업�?관련된 정보�?온라�?리소스에�?수집합니�?
* 수집�?�?리소스에�?관�?정보�?요약하고 출처�?기록합니�?
* 마지막으�? 요약�?모든 정보�?필터링하�?집계하여 최종 연구 보고서를 생성합니�?

## 튜토리얼
 - [동작원리](https://docs.gptr.dev/blog/building-Auto_Research_Engine)
 - [설치방법](https://www.loom.com/share/04ebffb6ed2a4520a27c3e3addcdde20?sid=da1848e8-b1f1-42d1-93c3-5b0b9c3b24ea)
 - [라이�?데모](https://www.loom.com/share/6a3385db4e8747a1913dd85a7834846f?sid=a740fd5b-2aa3-457e-8fb7-86976f59f9b8)


## 기능
- 📝 로컬 문서 �?�?소스�?사용하여 연구, 개요, 리소�?�?학습 보고�?생성
- 📜 2,000단어 이상�?길고 상세�?연구 보고�?생성 가�?- 🌐 연구�?20�?이상�?�?소스�?집계하여 객관적이�?사실�?기반�?결론 도출
- 🖥�?경량 HTML/CSS/JS와 프로덕션�?(NextJS + Tailwind) UX/UI 포함
- 🔍 자바스크립트 지�?�?소스 스크래핑 기능
- 📂 연구 과정에서 맥락�?메모�?추적 �?유지
- 📄 연구 보고서를 PDF, Word 등으�?내보내기 지�?
## 📖 문서

전체 문서(설치, 환경 설정, 간단�?예시)�?보려�?[여기](https://docs.gptr.dev/docs/Auto_Research_Engine/getting-started)�?참조하세�?

- 시작하기 (설치, 환경 설정, 간단�?예시)
- 맞춤 설정 �?구성
- 사용 방법 예시 (데모, 통합, 도커 지�?
- 참고자료 (전체 API 문서)

## ⚙️ 시작하기
### 설치
> **1단계** - Python 3.11 또는 �?이상�?버전�?설치하세�? [여기](https://www.tutorialsteacher.com/python/install-python)�?참조하여 단계�?가이드�?확인하세�?

> **2단계** - 프로젝트�?다운로드하고 해당 디렉토리�?이동하세�?

```bash
git clone https://github.com/MrOdd-Use/Auto_Research_Engine.git
cd Auto_Research_Engine
```

> **3단계** - �?가지 방법으로 API 키를 설정하세�? 직접 export하거�?`.env` 파일�?저장하세요.

Linux/Windows에서 임시 설정�?하려�?export 방법�?사용하세�?

```bash
export OPENAI_API_KEY={OpenAI API �?입력}
export TAVILY_API_KEY={Tavily API �?입력}
```

(선택 사항) 향상�?트레이싱 �?관�?가능성�?위해 다음�?설정�?수도 있습니다:

```bash
# export LANGCHAIN_TRACING_V2=true
# export LANGCHAIN_API_KEY={LangChain API �?입력}
```

�?영구적인 설정�?원한다면, 현재�?`Auto_Research_Engine` 디렉토리�?`.env` 파일�?생성하고 환경 변수를 입력하세�?(export 없이).

- 기본 LLM은 [GPT](https://platform.openai.com/docs/guides/gpt)이지�? `claude`, `ollama3`, `gemini`, `mistral` �?다른 LLM�?사용�?�?있습니다. LLM 제공자를 변경하�?방법은 [LLMs 문서](https://docs.gptr.dev/docs/Auto_Research_Engine/llms)�?참조하세�? �?프로젝트�?OpenAI GPT 모델�?최적화되�?있습니다.
- 기본 검색기�?[Tavily](https://app.tavily.com)이지�? `duckduckgo`, `google`, `bing`, `searchapi`, `serper`, `searx`, `arxiv`, `exa` 등의 검색기�?사용�?�?있습니다. 검�?제공자를 변경하�?방법은 [검색기 문서](https://docs.gptr.dev/docs/Auto_Research_Engine/retrievers)�?참조하세�?

### 빠른 시작

> **1단계** - 필요�?종속�?설치

```bash
pip install -r requirements.txt
```

> **2단계** - FastAPI�?에이전트 실행

```bash
python -m uvicorn main:app --reload
```

> **3단계** - 브라우저에서 http://localhost:8000 으로 이동하여 연구�?시작하세�?

<br />

**[Poetry](https://docs.gptr.dev/docs/Auto_Research_Engine/getting-started#poetry) 또는 [가�?환경](https://docs.gptr.dev/docs/Auto_Research_Engine/getting-started/getting-started#virtual-environment)�?대�?배우�?싶다�? [문서](https://docs.gptr.dev/docs/Auto_Research_Engine/getting-started/getting-started)�?참조하세�?**

### PIP 패키지�?실행하기
```bash
pip install Auto_Research_Engine
```

```python
...
from gpt_researcher import GPTResearcher

query = "�?Nvidia 주식�?오르�?있나�?"
researcher = GPTResearcher(query=query)
# 주어�?질문�?대�?연구 수행
research_result = await researcher.conduct_research()
# 보고�?작성
report = await researcher.write_report()
...
```

**�?많은 예제와 구성 옵션은 [PIP 문서](https://docs.gptr.dev/docs/Auto_Research_Engine/gptr/pip-package)�?참조하세�?**

## Docker�?실행

> **1단계** - [Docker 설치](https://docs.gptr.dev/docs/Auto_Research_Engine/getting-started/getting-started-with-docker)

> **2단계** - `.env.example` 파일�?복사하고 API 키를 추가�?�? 파일�?`.env`�?저장하세요.

> **3단계** - docker-compose 파일에서 실행하고 싶지 않은 서비스를 주석 처리하세�?

```bash
$ docker-compose up --build
```

> **4단계** - docker-compose 파일에서 아무 것도 주석 처리하지 않았다면, 기본적으�?�?가지 프로세스가 시작됩니�?
 - localhost:8000에서 실행 중인 Python 서버<br>
 - localhost:3000에서 실행 중인 React �?br>

브라우저에서 localhost:3000으로 이동하여 연구�?시작하세�?

## 🔍 관�?가능성 (Observability)

Auto_Research_Engine�?**LangSmith**�?지원하�?복잡�?다중 에이전트 워크플로우의 트레이싱�?관�?가능성�?향상시키�? 디버깅과 최적화를 용이하게 합니�?

트레이싱�?활성화하려면:
1. 다음 환경 변수를 설정하십시오:
   ```bash
   export LANGCHAIN_TRACING_V2=true
   export LANGCHAIN_API_KEY=당신의_API_�?   export LANGCHAIN_PROJECT="Auto_Research_Engine"
   ```
2. 평소와 같이 연구 작업�?실행하십시오. 모든 LangGraph 기반 에이전트 상호 작용은 자동으로 추적되며 LangSmith 대시보드에�?시각화됩니다.

## 📄 로컬 문서�?연구하기

Auto_Research_Engine�?사용하여 로컬 문서�?기반으로 연구 작업�?수행�?�?있습니다. 현재 지원되�?파일 형식은 PDF, 일반 텍스�? CSV, Excel, Markdown, PowerPoint, Word 문서입니�?

1단계: `DOC_PATH` 환경 변수를 설정하여 문서가 있는 폴더�?지정하세요.

```bash
export DOC_PATH="./my-docs"
```

2단계:
 - 프론트엔�?앱을 localhost:8000에서 실행 중이라면, "Report Source" 드롭다운 옵션에서 "My Documents"�?선택하세�?
 - Auto_Research_Engine�?[PIP 패키지](https://docs.tavily.com/guides/Auto_Research_Engine/Auto_Research_Engine#pip-package)�?실행 중이라면, `report_source` 인수�?"local"�?설정하여 `GPTResearcher` 클래스를 인스턴스화하세요. [코드 예제](https://docs.gptr.dev/docs/Auto_Research_Engine/context/tailored-research)�?참조하세�?

## 👪 다중 에이전트 어시스턴�?
AI가 프롬프트 엔지니어�?�?RAG에서 다중 에이전트 시스템으�?발전함에 따라, 우리�?[LangGraph](https://python.langchain.com/v0.1/docs/langgraph/)�?구축�?새로�?다중 에이전트 어시스턴트를 소개합니�?

LangGraph�?사용하면 여러 에이전트�?전문 기술�?활용하여 연구 과정�?깊이와 질을 크게 향상시킬 �?있습니다. 최근 [STORM](https://arxiv.org/abs/2402.14207) 논문에서 영감�?받아, �?프로젝트�?AI 에이전트 팀�?주제�?대�?연구�?계획에서 출판까지 함께 수행하는 방법�?보여줍니�?

평균 실행은 5-6 페이지 분량�?연구 보고서를 PDF, Docx, Markdown 형식으로 생성합니�?

[여기](https://github.com/MrOdd-Use/Auto_Research_Engine/tree/master/multi_agents)에서 확인하거�?[문서](https://docs.gptr.dev/docs/Auto_Research_Engine/multi_agents/langgraph)에서 자세�?내용�?참조하세�?

## 🖥�?프론트엔�?애플리케이션

Auto_Research_Engine�?사용�?경험�?개선하고 연구 프로세스�?간소화하�?위해 향상�?프론트엔드를 제공합니�? 프론트엔드는 다음�?같은 기능�?제공합니�?

- 연구 쿼리�?입력�?�?있는 직관적인 인터페이�?- 연구 작업�?실시�?진행 상황 추적
- 연구 결과�?대화형 디스플레�?- 맞춤�?연구 경험�?위한 설정 가�?
�?가지 배포 옵션�?있습니다:
1. FastAPI�?제공되는 경량 정적 프론트엔�?2. 고급 기능�?제공하는 NextJS 애플리케이션

프론트엔�?기능�?대�?자세�?설치 방법 �?정보�?원하시면 [문서 페이지](https://docs.gptr.dev/docs/Auto_Research_Engine/frontend/introduction)�?참조하세�?

## 🚀 기여하기
우리�?기여�?적극 환영합니�? 관심이 있다�?[기여 가이드](https://github.com/MrOdd-Use/Auto_Research_Engine/blob/master/CONTRIBUTING.md)�?확인�?주세�?

[로드맵](https://trello.com/b/3O7KBePw/Auto_Research_Engine-roadmap) 페이지�?확인하고, 우리 [Discord 커뮤니티](https://discord.gg/QgZXvJAccX)�?가입하�?우리�?목표�?함께 참여�?주세�?
<a href="https://github.com/MrOdd-Use/Auto_Research_Engine/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=MrOdd-Use/Auto_Research_Engine" />
</a>

## ✉️ 지�?/ 문의
- [커뮤니티 Discord](https://discord.gg/spBgZmm3Xe)
- 저�?이메�? assaf.elovic@gmail.com

## 🛡�?면책 조항

�?프로젝트�?Auto_Research_Engine�?실험적인 응용 프로그램이며, 명시적이거나 묵시적인 보증 없이 "있는 그대�? 제공됩니�? 우리�?�?코드�?학술�?목적으로 Apache 2 라이선스 하에 공유하고 있습니다. 여기�?있는 것은 학술�?조언�?아니�? 학술 또는 연구 논문�?사용하는 것을 권장하지 않습니다.

편향되지 않은 연구 주장�?대�?우리�?견해:
1. Auto_Research_Engine�?주요 목표�?잘못�?정보와 편향�?사실�?줄이�?것입니다. �?방법은 무엇일까�? 우리�?�?많은 사이트를 스크래핑할수�?잘못�?데이터의 가능성�?줄어든다�?가정합니다. 여러 사이트에�?정보�?스크래핑하고 가�?빈번�?정보�?선택하면, 모든 정보가 틀�?확률은 매우 낮습니다.
2. 우리�?편향�?완전�?제거하려�?하지�?않지�? 가능한 �?줄이�?것을 목표�?합니�? **우리�?인간�?LLM�?가�?효과적인 상호작용�?찾기 위한 커뮤니티입니�?**
3. 연구에서 사람들도 이미 자신�?연구하는 주제�?대�?의견�?가지�?있기 때문�?편향되는 경향�?있습니다. �?도구�?많은 의견�?스크래핑하며, 편향�?사람이라�?결코 읽지 않았�?다양�?견해�?고르�?설명합니�?

**GPT-4 모델�?사용�?경우, 토큰 사용�?때문�?비용�?많이 �?�?있습니다.** �?프로젝트�?사용하는 경우, 자신�?토큰 사용�?�?관�?비용�?모니터링하고 관리하�?것은 본인�?책임입니�? OpenAI API 사용량을 정기적으�?확인하고, 예상�?못한 비용�?방지하기 위해 필요�?한도�?설정하거�?알림�?설정하는 것이 좋습니다.


---

<p align="center">
<a href="https://star-history.com/#MrOdd-Use/Auto_Research_Engine">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=MrOdd-Use/Auto_Research_Engine&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=MrOdd-Use/Auto_Research_Engine&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=MrOdd-Use/Auto_Research_Engine&type=Date" />
  </picture>
</a>
</p>

