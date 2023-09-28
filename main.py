import uvicorn
from fastapi import FastAPI, Request, Form, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate
from dotenv import load_dotenv, find_dotenv
from PyPDF2 import PdfReader
from pathlib import Path
from transformers import GPT2TokenizerFast
import re
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(verbose=True)

app = FastAPI()

tokenizer = GPT2TokenizerFast.from_pretrained("GPT2")

# Templates
templates = Jinja2Templates(directory="templates")

# File upload directory
upload_directory = Path("uploads")

def num_tokens_from_string(string: str) -> int:
    # Use the tokenizer to tokenize the input string
    tokens = tokenizer.encode(string, add_special_tokens=False)
    num_tokens = len(tokens)
    return num_tokens

@app.post("/uploadfile/")
async def upload_pdf_file(file: UploadFile = Form(...)):
    # Ensure the upload directory exists
    upload_directory.mkdir(parents=True, exist_ok=True)

    template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "You are a government contractor and you will be provided the content of a Contract "
                    "you would have to make a detailed and professional proposal for that contract "
                    "you would have to analyze the nature of the contract and you would make the proposal accordingly "
                    "Your proposal must sound professional and display the result in accurate format with suitable headings."
                    "You should not sound robotic and keep a professional tone throughout."
                )
            ),
            HumanMessagePromptTemplate.from_template("Following is the Content of the Contract:\n{text}"),
        ]
    )

    # Save the uploaded PDF file
    file_path = upload_directory / "contract.pdf"
    with file_path.open("wb") as pdf_file:
        pdf_file.write(file.file.read())

    reader = PdfReader(file_path)

    contract_text = ""
    for i, page in enumerate(reader.pages, 1):
        contract_text += "\n" + page.extract_text()
        num_of_tokens = num_tokens_from_string(contract_text)
        if num_of_tokens >= 3000:
            print("Max limit of Tokens Reached @{} tokens\nPages read: {}/{}".format(num_of_tokens, i, len(reader.pages)))
            break

    contract_text = re.sub(' +', ' ', contract_text)

    output = llm(template.format_messages(text=contract_text))

    with open('output.txt', 'w') as f:
        f.write(output.content)

    return {"AI_Response": output.content, "pdf_content": contract_text}

@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run(app)
