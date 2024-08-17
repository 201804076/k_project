from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from amadeus import Client, ResponseError
import openai
import logging
import re

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 프론트엔드 도메인 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SQLite 데이터베이스 설정
DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 유저 모델 정의
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

# 데이터베이스 초기화
Base.metadata.create_all(bind=engine)

# 의존성: 데이터베이스 세션 생성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic 모델 정의
class UserCreate(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

# 비밀번호 해시를 위한 유틸리티 함수
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# Amadeus 및 OpenAI API 키 설정
amadeus = Client(
    client_id="fe3si2AT8QusGNTAxDgxA7WXerqJc5b5",
    client_secret="0RVDy7zAUaqrCl55"
)

openai.api_key = "sk-cAhJUzH84yjULvNihOe0meYXsRKzM08qhApWayPY24T3BlbkFJKI6DVVqvbtyVI59_Dwcdwr-K08WD8JgTYfWHeZRX0A"

# 항공편 검색 요청 형식을 정의하는 Pydantic 모델
class FlightSearchRequest(BaseModel):
    originLocationCode: str
    destinationLocationCode: str
    departureDate: str
    returnDate: str = None
    adults: int = 1

    @validator('departureDate')
    def validate_date(cls, v):
        if not re.match(r'\d{4}-\d{2}-\d{2}', v):
            raise ValueError('departureDate는 YYYY-MM-DD 형식이어야 합니다.')
        return v

class ChatRequest(BaseModel):
    message: str

messages = [
    {"role": "system", "content": "You are a helpful Travel Guide, and only use Korean."}
]

@app.post("/flights")
async def search_flights(request: FlightSearchRequest):
    try:
        logger.info(f"Received flight search request: {request.dict()}")

        search_params = {
            "originLocationCode": request.originLocationCode,
            "destinationLocationCode": request.destinationLocationCode,
            "departureDate": request.departureDate,
            "adults": request.adults,
        }

        if request.returnDate:
            search_params["returnDate"] = request.returnDate

        response = amadeus.shopping.flight_offers_search.get(**search_params)

        if not response.data:
            logger.info("No flights found for the given criteria.")
            return {"flight_data": [], "chatbot_message": "해당 조건에 맞는 항공편이 없습니다."}

        logger.info(f"Amadeus API response: {response.data}")
        return {"flight_data": response.data, "chatbot_message": f"{len(response.data)}개의 항공편을 찾았습니다."}

    except ResponseError as error:
        logger.error(f"Amadeus API Error: {error}\nError Message: {error.response.result}")
        raise HTTPException(status_code=500, detail="Amadeus API Error")
    except HTTPException as http_error:
        logger.error(f"Input validation error: {http_error.detail}")
        raise http_error
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Unexpected Server Error")


@app.get("/")
async def root():
    return {"message": "Hello, World!"}

@app.post("/signup")
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(user.password)
    new_user = User(username=user.username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"username": new_user.username}

@app.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    if not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    return {"message": "Login successful"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_message = request.message
    flight_data = request.flightData if "flightData" in request else None
    logger.info(f"User message: {user_message}")  # 추가된 로그
    logger.info(f"Flight data received: {flight_data}")  # 추가된 로그

    messages.append({"role": "user", "content": user_message})
    
    if flight_data:
        # flight_data를 활용해 추가 정보를 제공
        messages.append({"role": "system", "content": f"현재 항공편 정보: {len(flight_data)}개의 항공편이 있습니다."})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        assistant_message = response.choices[0].message["content"]
        messages.append({"role": "assistant", "content": assistant_message})
        return {"response": assistant_message}
    except Exception as e:
        logger.error(f"Error in chat_endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
