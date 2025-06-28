# backend/main.py  ───────────────────────────────────────────
import os, uuid, datetime as dt, hmac, hashlib, base64, asyncio, json
from sqlalchemy import inspect
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Depends, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import (
    create_engine, Column, String, DateTime, ForeignKey, select, delete
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
from dotenv import load_dotenv, find_dotenv
import nest_asyncio
import aiosqlite

# ── LangChain / OpenAI ─────────────────────────────────────
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.runnables import RunnableConfig
from graphparser.rag_tool import make_rag_tool
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_tavily import TavilySearch
from langgraph_supervisor import create_supervisor

# ── 기본 설정 ───────────────────────────────────────────────
load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "YOUR_KEY"
APP_SECRET     = os.getenv("APP_SECRET",  "change-me")

# Directory to store uploaded images
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── LLM 준비 ────────────────────────────────────────────────
client = OpenAI(api_key=OPENAI_API_KEY)
llm    = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

### ── Tool 설정 ──────────────────────────────────────── ###
# ── Python REPL 도구 ────────────────────────────────────────
python_repl_tool = PythonREPLTool()
# ── RAG 검색 도구 ────────────────────────────────────────
rag_search_tool = make_rag_tool("RS")
# ── 웹 검색 도구 ────────────────────────────────────────
tavily_tool=TavilySearch(max_results=5,topic="general",include_raw_content=True)

### ── Prompt 설정 ──────────────────────────────────────── ###
reliability_searcher_text = Path("./prompts/reliability_searcher.md").read_text()
reliability_searcher_prompt = ChatPromptTemplate.from_messages([
    ("system", reliability_searcher_text),  
    MessagesPlaceholder(variable_name="messages")
])
coder_text = Path("./prompts/coder.md").read_text()
coder_prompt = ChatPromptTemplate.from_messages([
    ("system", coder_text),  
    MessagesPlaceholder(variable_name="messages")
])
websearcher_text = Path("./prompts/websearcher.md").read_text()
websearcher_prompt = ChatPromptTemplate.from_messages([
    ("system", websearcher_text),  
    MessagesPlaceholder(variable_name="messages")
])

### ── Agent 설정 ──────────────────────────────────────── ###
reliability_searcher_agent = create_react_agent(llm, tools=[rag_search_tool],prompt=reliability_searcher_prompt, name='reliability_searcher')
coder_agent = create_react_agent(model=llm,tools=[python_repl_tool],prompt=coder_prompt, name="coder")
websearcher_agent = create_react_agent(model= llm, tools=[tavily_tool], prompt=websearcher_prompt, name='websearcher')

# ── Checkpointer 설정 ───────────────────────────────────────
async def _init_saver(db_path: str = "lg.sqlite") -> AsyncSqliteSaver:
    conn = await aiosqlite.connect(db_path)
    return AsyncSqliteSaver(conn)

try:
    # asyncio.run() 는 이미 루프가 돌아가는 환경에서는 오류를 던지므로 예외 처리
    checkpointer = asyncio.run(_init_saver())
except RuntimeError:
    nest_asyncio.apply()
    checkpointer = asyncio.get_event_loop().run_until_complete(_init_saver())

# System Agent (Head 설정) - FIXED PROMPT
app = create_supervisor(
    [reliability_searcher_agent, websearcher_agent, coder_agent],
    model=llm,
    prompt=(
        "You are the team supervisor. Your job is to orchestrate multiple specialist agents to achieve the user's goal.\n"
        "\n"
        "There are three subordinate agents you can delegate tasks to:\n"
        "• research_agent — use this when you need information in the reliability or quality‑engineering domain.\n"
        "• websearcher_agent — use this when you need up‑to‑date information such as breaking news or other recent web content.\n"
        "• coder_agent — use this when you need to write or execute code, do calculations, or create visualizations.\n"
        "\n"
        "Workflow rules:\n"
        "1. Decide which agent(s) to call and in what order.\n"
        "2. After an agent returns, carefully read its answer. **If the answer contains a citation or '📌 출처' block, you must preserve it verbatim in your final reply.**\n"
        "3. If an agent omits citations when they are required (e.g. research_agent answer without '📌 출처'), ask that agent once more for the sources, then include them.\n"
        "4. When you have enough information, compile a comprehensive final answer in Korean.\n"
        "5. **IMPORTANT**: Only output your final consolidated answer. Do not repeat or echo the individual agent responses.\n"
        "6. Keep any citation blocks exactly as given (do not paraphrase or remove them).\n"
        "\n"
        "Always follow these rules. Never invent citations. Provide only ONE final response, not multiple responses."
    ),add_handoff_back_messages=True
)
simple_agent = app.compile(checkpointer = checkpointer)

# ── DB (SQLite, SQLAlchemy ORM) ────────────────────────────
Base   = declarative_base()
engine = create_engine("sqlite:///chat.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Thread(Base):
    __tablename__ = "threads"
    id       = Column(String, primary_key=True)
    user_id  = Column(String, index=True)
    title    = Column(String, default="새 대화")
    created  = Column(DateTime, default=dt.datetime.utcnow)
    messages = relationship("Message", cascade="all,delete")

class Message(Base):
    __tablename__ = "messages"
    id        = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    thread_id = Column(String, ForeignKey("threads.id"))
    role      = Column(String)   # "user" | "assistant"
    content   = Column(String)
    ts        = Column(DateTime, default=dt.datetime.utcnow)
    steps     = Column(String)
    images    = relationship("MessageImage", cascade="all,delete", back_populates="message")

class MessageImage(Base):
    __tablename__ = "message_images"
    id         = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    message_id = Column(String, ForeignKey("messages.id"))
    url        = Column(String)  # image URL
    message    = relationship("Message", back_populates="images")

Base.metadata.create_all(engine)

def _ensure_schema():
    """Add missing columns to existing tables if the database is from an older version."""
    inspector = inspect(engine)
    if "message_images" in inspector.get_table_names():
        cols = [col["name"] for col in inspector.get_columns("message_images")]
        if "url" not in cols:
            with engine.begin() as conn:
                conn.exec_driver_sql("ALTER TABLE message_images ADD COLUMN url TEXT")
    if "messages" in inspector.get_table_names():
        cols = [col["name"] for col in inspector.get_columns("messages")]
        if "steps" not in cols:
            with engine.begin() as conn:
                conn.exec_driver_sql("ALTER TABLE messages ADD COLUMN steps TEXT")

_ensure_schema()

# ── DB 세션 의존성 ─────────────────────────────────────────

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ── 간단 서명 기반 Auth ────────────────────────────────────

def _sign(user: str) -> str:
    sig = hmac.new(APP_SECRET.encode(), user.encode(), hashlib.sha256).digest()
    return f"{user}.{base64.urlsafe_b64encode(sig).decode()}"

def _verify(tok: str) -> str:
    try:
        user, sig = tok.split(".")
        expect = base64.urlsafe_b64encode(
            hmac.new(APP_SECRET.encode(), user.encode(), hashlib.sha256).digest()
        ).decode()
        if hmac.compare_digest(sig, expect):
            return user
    except Exception:
        ...
    raise HTTPException(401, "Invalid token")

def current_user(auth: str = Header(..., alias="Authorization")):
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Bearer required")
    return _verify(auth.removeprefix("Bearer ").strip())

# ── FastAPI 앱 ──────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
app.mount("/images", StaticFiles(directory=UPLOAD_DIR), name="images")

# Convert local image URLs to data URIs for OpenAI access
def _prepare_image_for_openai(url: str) -> str:
    """Return a data URL if the image is local, otherwise return the URL."""
    try:
        parsed = urlparse(url)
        if not parsed.scheme or parsed.hostname in {"localhost", "127.0.0.1"}:
            fname = os.path.basename(parsed.path)
            path = os.path.join(UPLOAD_DIR, fname)
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            ext = os.path.splitext(fname)[1].lower().lstrip(".") or "png"
            if ext in {"jpg", "jpeg"}:
                mime = "jpeg"
            elif ext in {"png", "gif", "webp"}:
                mime = ext
            else:
                mime = "png"
            return f"data:image/{mime};base64,{b64}"
    except Exception as e:
        print(f"Image conversion error: {e}")
    return url

# ---------- 1) 로그인 --------------------------------------
class LoginReq(BaseModel):
    user: str
    key:  str

@app.post("/login")
def login(body: LoginReq):
    if body.key != "open-sesame" or len(body.user) < 3:
        raise HTTPException(401, "Bad credentials")
    return {"token": _sign(body.user)}

# ---------- 2) Thread CRUD ---------------------------------
@app.post("/threads")
def new_thread(db: Session = Depends(get_db), user=Depends(current_user)):
    tid = str(uuid.uuid4())
    db.add(Thread(id=tid, user_id=user))
    db.commit()
    return {"thread_id": tid, "title": "새 대화"}

@app.get("/threads")
def list_threads(db: Session = Depends(get_db), user=Depends(current_user)):
    rows = db.scalars(select(Thread).where(Thread.user_id == user)).all()
    return [{"id": t.id, "title": t.title} for t in rows]

@app.patch("/threads/{tid}")
def rename_thread(tid: str, body: dict, db: Session = Depends(get_db), user=Depends(current_user)):
    th = db.get(Thread, tid)
    if not th or th.user_id != user:
        raise HTTPException(404)
    th.title = (body.get("title") or "제목 없음")[:50]
    db.commit()
    return {"ok": True, "title": th.title}

@app.delete("/threads/{tid}")
def delete_thread(tid: str, db: Session = Depends(get_db), user=Depends(current_user)):
    rows = db.execute(delete(Thread).where(Thread.id == tid, Thread.user_id == user)).rowcount
    if rows:
        db.commit(); return {"ok": True}
    raise HTTPException(404)

# ---------- 이미지 업로드 ----------------------------------
@app.post("/upload")
async def upload_image(file: UploadFile = File(...), user=Depends(current_user)):
    ext = os.path.splitext(file.filename)[1] or ".png"
    fname = f"{uuid.uuid4()}{ext}"
    path  = os.path.join(UPLOAD_DIR, fname)
    with open(path, "wb") as f:
        f.write(await file.read())
    return {"url": f"/images/{fname}"}

# ---------- 3) 메시지 조회 ---------------------------------
@app.get("/messages/{tid}")
def get_messages(tid: str, db: Session = Depends(get_db), user=Depends(current_user)):
    db.scalar(select(Thread).where(Thread.id == tid, Thread.user_id == user)) \
        or (_ for _ in ()).throw(HTTPException(404))
    msgs = db.scalars(select(Message).where(Message.thread_id == tid).order_by(Message.ts)).all()

    def to_dict(m: Message):
        img = m.images[0].url if m.images else None
        return {
            "role": m.role,
            "content": m.content,
            "image": img,
            "timestamp": m.ts.isoformat(),
            "steps": json.loads(m.steps) if m.steps else [],
        }

    return [to_dict(m) for m in msgs]

# ---------- 4) 대화 (동기) ---------------------------------
class ChatReq(BaseModel):
    thread_id: str
    question:  str
    image: Optional[str] = None  # base64 혹은 URL

@app.post("/chat")
def chat(req: ChatReq, db: Session = Depends(get_db), user=Depends(current_user)):
    # 0) 권한 체크
    db.scalar(select(Thread).where(Thread.id == req.thread_id, Thread.user_id == user)) \
        or (_ for _ in ()).throw(HTTPException(404))

    # 1) LangGraph 에이전트 실행 ----------------------------
    img = _prepare_image_for_openai(req.image) if req.image else None
    human = HumanMessage(content=req.question if not img else [
        {"type": "text", "text": req.question},
        {"type": "image_url", "image_url": {"url": img, "detail": "auto"}}
    ])

    state = {"messages": [human]}
    cfg = RunnableConfig(configurable={"thread_id": req.thread_id}, callbacks=[])

    result = simple_agent.invoke(state, cfg)
    answer = result["messages"][-1].content

    # 2) 이미지 데이터 분리 -------------------------------
    image_url = None
    if "IMAGE_DATA:" in answer:
        parts = answer.split("IMAGE_DATA:")
        if len(parts) > 1:
            b64 = parts[1].strip()
            img_bytes = base64.b64decode(b64)
            fname = f"{uuid.uuid4()}.png"
            path = os.path.join(UPLOAD_DIR, fname)
            with open(path, "wb") as f:
                f.write(img_bytes)
            image_url = f"/images/{fname}"
        answer = parts[0].strip()

    # 3) DB 기록 -------------------------------------------
    user_msg = Message(thread_id=req.thread_id, role="user", content=req.question)
    if req.image:
        user_msg.images.append(MessageImage(url=req.image))

    assistant_msg = Message(thread_id=req.thread_id, role="assistant", content=answer)
    if image_url:
        assistant_msg.images.append(MessageImage(url=image_url))

    db.add_all([user_msg, assistant_msg])
    db.commit()

    return {"role": "assistant", "content": answer, "image": image_url}

@app.post("/chat/stream")
async def chat_stream(req: ChatReq, db: Session = Depends(get_db), user=Depends(current_user)):
    # 0) 권한 체크
    db.scalar(select(Thread).where(Thread.id == req.thread_id, Thread.user_id == user)) \
        or (_ for _ in ()).throw(HTTPException(404))

    # 1) async generator ----------------------------------
    async def gen():
        assistant_acc = ""
        image_url = None
        current_agent = None
        supervisor_streaming = False
        previous_agent = None
        steps_data = [{"type": "step", "content": "🤖 질문을 분석하고 있습니다..."}]

        def add_step(step_type: str, content: str):
            steps_data.append({"type": step_type, "content": content})
        
        try:
            img = _prepare_image_for_openai(req.image) if req.image else None
            human = HumanMessage(content=req.question if not img else [
                {"type": "text", "text": req.question},
                {"type": "image_url", "image_url": {"url": img, "detail": "auto"}}
            ])

            state = {"messages": [human]}
            cfg = RunnableConfig(configurable={"thread_id": req.thread_id}, callbacks=[])

            # 초기 supervisor 시작 메시지 (실시간 스트림만 전송)
            yield f"[STEP] 🤖 질문을 분석하고 있습니다...\n".encode()

            async for event in simple_agent.astream_events(state, cfg, version="v1"):
                event_name = event.get("name", "")
                event_type = event.get("event", "")
                
                # Track current agent and detect transfers
                if event_type == "on_chain_start":
                    previous_agent = current_agent
                    
                    if "reliability_searcher" in event_name:
                        current_agent = "reliability_searcher"
                        if previous_agent != current_agent:
                            add_step("step", "📋 신뢰성 전문가에게 전달합니다...")
                            yield f"[STEP] 📋 신뢰성 전문가에게 전달합니다...\n".encode()
                            add_step("step", "🔍 신뢰성 정보를 검색하고 있습니다...")
                            yield f"[STEP] 🔍 신뢰성 정보를 검색하고 있습니다...\n".encode()
                    elif "websearcher" in event_name:
                        current_agent = "websearcher"
                        if previous_agent != current_agent:
                            add_step("step", "🌐 웹 검색 전문가에게 전달합니다...")
                            yield f"[STEP] 🌐 웹 검색 전문가에게 전달합니다...\n".encode()
                            add_step("step", "🔍 웹에서 최신 정보를 검색하고 있습니다...")
                            yield f"[STEP] 🔍 웹에서 최신 정보를 검색하고 있습니다...\n".encode()
                    elif "coder" in event_name:
                        current_agent = "coder"
                        if previous_agent != current_agent:
                            add_step("step", "💻 코딩 전문가에게 전달합니다...")
                            yield f"[STEP] 💻 코딩 전문가에게 전달합니다...\n".encode()
                            add_step("step", "🐍 코드를 실행하고 있습니다...")
                            yield f"[STEP] 🐍 코드를 실행하고 있습니다...\n".encode()
                    elif "supervisor" in event_name.lower() or event_name == "LangGraph":
                        if current_agent and current_agent != "supervisor":
                            # 에이전트에서 supervisor로 돌아왔을 때
                            add_step("step", f"🤖 {get_agent_name(current_agent)} 작업이 완료되어 결과를 취합하고 있습니다...")
                            yield f"[STEP] 🤖 {get_agent_name(current_agent)} 작업이 완료되어 결과를 취합하고 있습니다...\n".encode()
                        current_agent = "supervisor"

                # Tool execution events with agent context
                elif event_type == "on_tool_start":
                    tool_name = event.get("name", "")
                    if tool_name == "RS":
                        add_step("step", "📚 신뢰성 데이터베이스 검색 중...")
                        yield f"[STEP] 📚 신뢰성 데이터베이스 검색 중...\n".encode()
                    elif tool_name == "TavilySearch":
                        add_step("step", "🔍 웹 검색 실행 중...")
                        yield f"[STEP] 🔍 웹 검색 실행 중...\n".encode()
                    elif tool_name == "PythonREPLTool":
                        add_step("step", "🐍 Python 코드 실행 중...")
                        yield f"[STEP] 🐍 Python 코드 실행 중...\n".encode()

                elif event_type == "on_tool_end":
                    tool_name = event.get("name", "")
                    output = event.get("data", {}).get("output", "")
                    if hasattr(output, "content"):
                        output = output.content
                    elif not isinstance(output, str):
                        output = str(output)

                    if "IMAGE_DATA:" in output:
                        parts = output.split("IMAGE_DATA:")
                        if len(parts) > 1:
                            b64 = parts[1].strip()
                            try:
                                img_bytes = base64.b64decode(b64)
                                fname = f"{uuid.uuid4()}.png"
                                path = os.path.join(UPLOAD_DIR, fname)
                                with open(path, "wb") as f:
                                    f.write(img_bytes)
                                image_url = f"/images/{fname}"
                                add_step("observation", "📊 그래프가 생성되었습니다.")
                                yield "[OBS] 📊 그래프가 생성되었습니다.\n".encode()
                            except Exception as e:
                                print(f"Image processing error: {e}")
                                add_step("observation", "⚠️ 이미지 처리 중 오류가 발생했습니다.")
                                yield "[OBS] ⚠️ 이미지 처리 중 오류가 발생했습니다.\n".encode()
                    else:
                        # Show tool completion with agent context
                        agent_name = get_agent_name(current_agent)
                        if tool_name == "RS":
                            add_step("observation", f"✅ {agent_name}: 신뢰성 데이터 검색 완료")
                            yield f"[OBS] ✅ {agent_name}: 신뢰성 데이터 검색 완료\n".encode()
                        elif tool_name == "TavilySearch":
                            add_step("observation", f"✅ {agent_name}: 웹 검색 완료")
                            yield f"[OBS] ✅ {agent_name}: 웹 검색 완료\n".encode()
                        elif tool_name == "PythonREPLTool":
                            add_step("observation", f"✅ {agent_name}: 코드 실행 완료")
                            yield f"[OBS] ✅ {agent_name}: 코드 실행 완료\n".encode()
                        else:
                            add_step("observation", f"✅ {agent_name}: 작업 완료")
                            yield f"[OBS] ✅ {agent_name}: 작업 완료\n".encode()

                # Chat model streaming - ONLY from supervisor
                elif event_type == "on_chat_model_stream":
                    # Only stream if this is the supervisor's response
                    if current_agent == "supervisor" or "supervisor" in event_name.lower():
                        chunk = event.get("data", {}).get("chunk", {})
                        if hasattr(chunk, "content") and chunk.content:
                            token = chunk.content
                            if "IMAGE_DATA:" not in token:
                                # 최종 응답 시작 시 메시지
                                if not supervisor_streaming:
                                    add_step("step", "🎯 최종 답변을 생성하고 있습니다...")
                                    yield f"[STEP] 🎯 최종 답변을 생성하고 있습니다...\n".encode()
                                supervisor_streaming = True
                                assistant_acc += token
                                yield token.encode()

            # If supervisor didn't stream (fallback), get final result
            if not supervisor_streaming:
                add_step("step", "🎯 최종 답변을 준비하고 있습니다...")
                yield f"[STEP] 🎯 최종 답변을 준비하고 있습니다...\n".encode()
                result = simple_agent.invoke(state, cfg)
                final_answer = result["messages"][-1].content
                
                if "IMAGE_DATA:" in final_answer:
                    parts = final_answer.split("IMAGE_DATA:")
                    final_answer = parts[0].strip()
                    if len(parts) > 1 and not image_url:
                        try:
                            b64 = parts[1].strip()
                            img_bytes = base64.b64decode(b64)
                            fname = f"{uuid.uuid4()}.png"
                            path = os.path.join(UPLOAD_DIR, fname)
                            with open(path, "wb") as f:
                                f.write(img_bytes)
                            image_url = f"/images/{fname}"
                        except Exception as e:
                            print(f"Image processing error: {e}")
                
                assistant_acc = final_answer
                yield final_answer.encode()

            # Signal completion
            yield "[DONE]\n".encode()

            # 2) DB 저장 ----------------------------------
            user_msg = Message(thread_id=req.thread_id, role="user", content=req.question)
            if req.image:
                user_msg.images.append(MessageImage(url=req.image))

            clean_content = assistant_acc.split("IMAGE_DATA:")[0].strip() if "IMAGE_DATA:" in assistant_acc else assistant_acc
            assistant_msg = Message(
                thread_id=req.thread_id,
                role="assistant",
                content=clean_content,
                steps=json.dumps(steps_data),
            )
            if image_url:
                assistant_msg.images.append(MessageImage(url=image_url))

            db.add_all([user_msg, assistant_msg])
            db.commit()

        except Exception as e:
            print(f"Stream error: {e}")
            yield f"Error: {str(e)}".encode()

    return StreamingResponse(
        gen(),
        media_type="text/plain; charset=utf-8",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
        }
    )

# Helper function to get agent display name
def get_agent_name(agent_type: str) -> str:
    agent_names = {
        "reliability_searcher": "신뢰성 전문가",
        "websearcher": "웹 검색 전문가", 
        "coder": "코딩 전문가",
        "supervisor": "관리자"
    }
    return agent_names.get(agent_type, agent_type)