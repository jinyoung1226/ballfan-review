#!/usr/bin/env python
# coding: utf-8

# In[215]:


#pip install fastapi uvicorn chromadb sentence-transformers


# In[47]:


#pip install langchain langchain-openai openai


# In[2]:


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import chromadb
import os
import nest_asyncio
import uvicorn
from dotenv import load_dotenv
import requests


# In[14]:


# FastAPI 앱 인스턴스 생성
app = FastAPI()

# OpenAI API 키 설정
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key;  

# ChatGPT 모델 인스턴스 생성
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 장소 키워드 추출 프롬프트 템플릿 생성
place_prompt = PromptTemplate(
    input_variables=["review_text"], 
    template=(
        "다음 리뷰에서 장소, 가게 이름, 위치에 해당하는 이름만 추출해서 리스트 형태로 출력해줘.\n"
        "네이버 지도에 넣었을 때 장소가 표시 안 될 것 같으면 절대로 출력하지 마.\n"
        "출력은 반드시 Python 리스트 형식으로 해줘.\n\n"
        "리뷰: {review_text}\n"
        "출력:"
    )
)

# 리뷰 요약 프롬프트 템플릿 생성
summary_prompt = PromptTemplate(
    input_variables=["review_text"],
    template=(
        "다음 리뷰를 읽고, 그 리뷰를 요약하는 표현 2개만 갯수 지켜서 작성해줘.\n"
        "- 각 표현은 감정이나 인상을 담은 짧은 문장으로 만들어줘.\n"
        "- 형식은 반드시 '이모지 + 문장' 형태여야 해. 예: '🧼 청결해요!', '👀 시야가 좋아요!'\n"
        "- 문장은 '~해요!', '~좋아요!'처럼 자연스러운 말투로 끝나야 해.\n"
        "- 너무 길게 쓰지 말고, 한 문장 내외로 해줘.\n"
        "- 반드시 Python 리스트 형식으로 출력해. 예: ['🧼 청결해요!', '👀 시야가 좋아요!']\n"
        "- 적절한 표현이 없다면 빈 리스트 []로 반환해.\n\n"
        "리뷰: \"{review_text}\"\n"
        "출력:"
    )
)

# 모델, Chroma DB 세팅
model = SentenceTransformer("jhgan/ko-sbert-nli")
chroma_client = chromadb.PersistentClient(path="./chroma_DB")
seat_collection = chroma_client.get_or_create_collection(name="seat_vectors")
review_collection = chroma_client.get_or_create_collection(name="review_vectors")

# 리뷰 텍스트 요청 바디 스키마
class ReviewText(BaseModel):
    review_id: int
    review: str
    stadium: str
    
# 좌석 정보 요청 바디 스키마
class SeatReview(BaseModel):
    review_id: int
    seat: str
    stadium: str
    
# 네이버 API 키
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

# 리뷰 작성 시 좌석 정보 임베딩 후, 벡터 DB에 저장하는 api
@app.post("/review/save/seat")
def save_seat_review(data: SeatReview):
    # 좌석 정보 임베딩
    embedding = model.encode(data.seat)

    # chroma DB에 저장
    seat_collection.add(
        ids=[str(data.review_id)],
        documents=[data.seat],
        embeddings=[embedding.tolist()],
        metadatas=[{
            "review_id": data.review_id,
            "seat": data.seat,
            "stadium": data.stadium,
            "type": "seat"
        }]
    )

    return {"message": "좌석 정보 벡터 저장 완료"}



# 리뷰 작성 시 리뷰 텍스트 임베딩 후, 벡터 DB에 저장하는 api
@app.post("/review/save/text")
def save_review(data: ReviewText):
    # 리뷰 텍스트 임베딩
    embedding = model.encode(data.review)

    # chroma DB에 저장
    review_collection.add(
        ids=[str(data.review_id)],
        documents=[data.review],
        embeddings=[embedding.tolist()],
        metadatas=[{
            "review_id": data.review_id,
            "review": data.review,
            "stadium": data.stadium,
            "type": "review"
        }]
    )

    return {"message": "리뷰 텍스트 벡터 저장 완료"}


# 리뷰 삭제 시, 좌석과 텍스트 벡터 db에서 review_id 기반으로 삭제하는 메서드 
@app.delete("/review/delete/{review_id}")
def delete_review(review_id: int):
    try:
        # 좌석 벡터 삭제
        seat_collection.delete(ids=[str(review_id)])

        # 리뷰 텍스트 벡터 삭제
        review_collection.delete(ids=[str(review_id)])

        return {"message": f"review_id {review_id} 에 해당하는 벡터 삭제 완료"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"벡터 삭제 중 오류 발생: {str(e)}")



# 리뷰 조회 시, 해당 리뷰의 seat 텍스트를 임배딩 후, 벡터 DB에서 유사 좌석 검색하고 유사한 리뷰 응답하는 api
@app.post("/review/get/seat")
def get_seat_review(data: SeatReview):
    # 조회 리뷰 좌석을 임베딩
    query_vector = model.encode(data.seat)

    # Chroma DB에서 유사 좌석 5개 검색
    results = seat_collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=5,
        where={"$and": [{"stadium": data.stadium},{"type": "seat"}]},
        include=["metadatas"] # review_id는 metadata에 저장되어 있음
    )

    # 유사한 review_id 리스트 추출
    review_ids = [meta["review_id"] for meta in results["metadatas"][0]]

    # 자기 자신(review_id)를 제외
    filtered_ids = [rid for rid in review_ids if rid != data.review_id]

    return {"review_ids": filtered_ids}



# 리뷰 조회 시, 해당 리뷰의 텍스트를 임배딩 후, 벡터 DB에서 유사 리뷰 검색하고 유사한 리뷰 응답하는 api
@app.post("/review/get/text")
def get_review(data: ReviewText):
    # 조회 리뷰 텍스트를 임베딩
    query_vector = model.encode(data.review)

    # Chroma DB에서 유사 리뷰 5개 검색
    results = review_collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=5,
        where={"$and": [{"stadium": data.stadium},{"type": "review"}]},
        include=["metadatas"] # review_id는 metadata에 저장되어 있음
    )

    # 유사한 review_id 리스트 추출
    review_ids = [meta["review_id"] for meta in results["metadatas"][0]]

    # 자기 자신(review_id)를 제외
    filtered_ids = [rid for rid in review_ids if rid != data.review_id]

    return {"review_ids": filtered_ids}



# 리뷰 조회 시, 리뷰 요약 키워드 정보 응답하는 api
@app.post("/review/get/summary")
def get_summarize_review(data:ReviewText):
    # 프롬프트 채우고 GPT 호출
    filled_prompt = summary_prompt.format(review_text=data.review)
    response = llm.invoke(filled_prompt)
    extracted = response.content.strip()

    # 요약 결과가 비어있으면 빈 리스트 반환
    if extracted in ("[]", "", "None"):
        return {"summary": []}

    try:
        summary = [kw.strip().strip("'\"") for kw in extracted.strip("[]").split(",")]
        return {"summary": summary}
    except Exception:
        return {"summary": []}



# 리뷰 조회 시, 해당 리뷰의 텍스트 gpt에 넘겨 장소 키워드 추출 후, 네이버 검색 api를 통해 장소 정보 응답하는 api
@app.post("/review/get/place")
def get_place_map_information(data: ReviewText):
    review = data.review
    stadium = data.stadium.strip()

    # 경기장 지역명 추출  
    region = stadium.split()[0] if stadium else ""

    # 프롬프트 채우고 GPT 호출
    filled_prompt = place_prompt.format(review_text=review)
    response = llm.invoke(filled_prompt)
    extracted = response.content.strip()
    print(extracted)

    if extracted in ("[]", "", "None"):
        print("여기1")
        return {"places": []}  # 비어 있으면 빈 리스트 반환

    # 키워드 리스트로 파싱
    try:
        extracted_clean = extracted.replace("```python", "").replace("```", "").strip()
        keywords = [kw.strip().strip("'\"") for kw in extracted_clean.strip("[]").split(",") if kw.strip()]
        print(keywords)
    except Exception:
        print("여기2")
        return {"places": []}  # 파싱 실패해도 빈 리스트

    result_list = []
    for keyword in keywords:
        combined_keyword = f"{region} {keyword.strip()}"
        print(f"🔍 검색 키워드: {combined_keyword}")
        place = query_naver_local_api(combined_keyword)
        if place:
            result_list.append(place)

    return {"places": result_list}  # 결과가 없으면 빈 리스트 그대로 반환



# 네이버 검색 api 조회 함수
def query_naver_local_api(keyword: str):
    url = f"https://openapi.naver.com/v1/search/local.json"
    headers = {
        'X-Naver-Client-Id': CLIENT_ID,
        'X-Naver-Client-Secret': CLIENT_SECRET,
    }
    params = {
        'query': keyword,
        'display': 1
    }

    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    print(f"[{keyword}] 검색결과:", data) 
    if data["items"]:
        item = data["items"][0]
        return {
            "title": item["title"].replace("<b>", "").replace("</b>", ""),
            "roadAddress": item["roadAddress"],
            "longitude": item["mapx"],
            "latitude": item["mapy"],
            "map_url": f"https://map.naver.com/v5/search/{keyword}"
        }
    return None