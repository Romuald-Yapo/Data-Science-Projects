# Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/
# OpenAI Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/openai/

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables  from .env
load_dotenv()

# Create a ChatMistral model
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")

# Invoke the model with a message
result = model.invoke("What is the best web browser")
print("Full result:")
print(result)
print("Content only:")
print(result.content)
