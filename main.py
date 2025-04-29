import logging
import os
import json
import base64
from email.utils import parseaddr

import httplib2
import pandas as pd
import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from apscheduler.schedulers.background import BackgroundScheduler
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as AuthRequest
from googleapiclient.discovery import build
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn

# --- CONFIGURATION ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
# Paths for credentials/tokens
CREDENTIALS_FILE = r'C:\Users\Admin\Documents\Python\EmailAutoReply\credentials.json'
TOKEN_FILE = r'C:\Users\Admin\Documents\Python\EmailAutoReply\token.json'
RESPONSES_FILE = r"C:\Users\Admin\Documents\Python\EmailAutoReply\response.json"
PRICE_LIST_FILE = r'C:\Users\Admin\Documents\Python\EmailAutoReply\giaban.xlsx'
ALLOWED_DOMAIN = 'winmart.masangroup.com'
PENDING_DIR = r"C:\Users\Admin\Documents\Python\EmailAutoReply\pending_emails"

# --- WRITE CREDENTIAL FILES FROM ENV VARS ---
def _write_from_env(env_var: str, filename: str):
    b64 = os.getenv(env_var)
    if b64:
        try:
            data = base64.b64decode(b64.encode())
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as f:
                f.write(data)
            logging.info(f"Wrote {filename} from env var {env_var}")
        except Exception as e:
            logging.error(f"Failed to write {filename} from env var {env_var}: {e}")

# Create credentials and token files before usage
_write_from_env('GOOGLE_CRED_B64', CREDENTIALS_FILE)
_write_from_env('GOOGLE_TOKEN_B64', TOKEN_FILE)

# --- SETUP LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- LOAD PRICE LIST ---
price_df = pd.read_excel(PRICE_LIST_FILE, dtype={'MÃ Sản phẩm': str})
price_col = next((col for col in price_df.columns if 'Giá' in col), None)
price_df['unit_price'] = (
    price_df[price_col].astype(str)
        .str.replace(r"[^0-9]", "", regex=True)
        .astype(float)
)
PRICE_MAP = dict(zip(price_df['MÃ Sản phẩm'], price_df['unit_price']))

# --- LOAD RESPONSES & MODEL ---
with open(RESPONSES_FILE, 'r', encoding='utf-8') as f:
    RESPONSES = json.load(f)
TOKENIZER = AutoTokenizer.from_pretrained("vinai/phobert-base")
MODEL = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=4)
CATEGORIES = ["đổi trả", "hủy vật lý", "phản ánh sản phẩm", "khác"]

# --- GMAIL SERVICE INIT & REFRESH ---
def init_gmail_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    # Refresh if expired
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(AuthRequest())
            with open(TOKEN_FILE, 'w', encoding='utf-8') as token_f:
                token_f.write(creds.to_json())
            logger.info("Refreshed access token and updated token.json")
        except Exception as e:
            logger.error(f"Failed to refresh token: {e}", exc_info=True)

    http = httplib2.Http(timeout=60)
    return build('gmail', 'v1', credentials=creds, http=http)

# --- EMAIL PROCESSING HELPERS ---
def extract_table_from_html(html: str):
    try:
        return pd.read_html(html)[0]
    except Exception:
        return None


def calculate_total_price(table):
    if table is None:
        return 0
    df = table.copy()
    if 'MÃ Sản phẩm' in df.columns and 'Số lượng' in df.columns:
        df['quantity'] = pd.to_numeric(df['Số lượng'], errors='coerce').fillna(0).astype(int)
        df['unit_price'] = df['MÃ Sản phẩm'].astype(str).map(PRICE_MAP).fillna(0)
        return float((df['quantity'] * df['unit_price']).sum())
    price_col = next((c for c in df.columns if 'Giá' in c), None)
    if price_col:
        return float(df[price_col].astype(str).str.replace(r"[^0-9]", "", regex=True).astype(float).sum())
    return 0


def classify_email(content: str) -> str:
    inputs = TOKENIZER(content, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = MODEL(**inputs)
    return CATEGORIES[int(torch.argmax(outputs.logits))]


def choose_reply(body: str) -> str:
    tbl = extract_table_from_html(body)
    total = calculate_total_price(tbl)
    if tbl is not None:
        category = 'đổi trả' if total > 200000 else 'hủy vật lý'
    else:
        category = classify_email(body)
    return next((r['reply'] for r in RESPONSES if r['category'] == category), "Không tìm thấy câu trả lời phù hợp.")

# --- CHECK & QUEUE EMAILS ---
def check_emails():
    try:
        service = init_gmail_service()
        result = service.users().messages().list(userId='me', labelIds=['INBOX'], q="is:unread").execute()
        msgs = result.get('messages', [])
    except Exception as e:
        logger.error(f"Failed to fetch unread emails: {e}", exc_info=True)
        return

    for m in msgs:
        try:
            d = service.users().messages().get(userId='me', id=m['id'], format='full').execute()
            hdrs = {h['name']: h['value'] for h in d.get('payload', {}).get('headers', [])}
            sender = parseaddr(hdrs.get('From', ''))[1]
            if sender.split('@')[-1].lower() != ALLOWED_DOMAIN:
                continue

            body = ''
            for part in d.get('payload', {}).get('parts', []):
                if part.get('mimeType') in ['text/plain', 'text/html'] and part.get('body', {}).get('data'):
                    body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
                    break
            reply = choose_reply(body)

            os.makedirs(PENDING_DIR, exist_ok=True)
            with open(f"{PENDING_DIR}/{m['id']}.json", 'w', encoding='utf-8') as f:
                json.dump({'message_id': m['id'], 'to': hdrs.get('From', ''), 'subject': hdrs.get('Subject', ''), 'body': reply}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error processing message {m['id']}: {e}", exc_info=True)
            continue

# --- SEND EMAIL & ROUTES ---
def send_email(service, to, subject, body, thread_id):
    from email.mime.text import MIMEText
    message = MIMEText(body, 'plain', 'utf-8')
    message['To'] = to
    message['Subject'] = f"Re: {subject}"
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    service.users().messages().send(userId='me', body={'raw': raw, 'threadId': thread_id}).execute()
    service.users().messages().modify(userId='me', id=thread_id, body={'removeLabelIds': ['UNREAD']}).execute()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    pending_files = os.listdir(PENDING_DIR) if os.path.exists(PENDING_DIR) else []
    pending_emails = []
    for file in pending_files:
        with open(os.path.join(PENDING_DIR, file), 'r', encoding='utf-8') as f:
            pending_emails.append(json.load(f))
    return templates.TemplateResponse("index.html", {"request": request, "pending_emails": pending_emails})

@app.post("/confirm/{message_id}")
async def confirm(message_id: str):
    file_path = f"{PENDING_DIR}/{message_id}.json"
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            email_data = json.load(f)
        service = init_gmail_service()
        send_email(service, email_data['to'], email_data['subject'], email_data['body'], message_id)
        os.remove(file_path)
    return {"status": "success"}

# --- SCHEDULER ---
scheduler = BackgroundScheduler()
scheduler.add_job(check_emails, 'interval', minutes=5)
scheduler.start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
