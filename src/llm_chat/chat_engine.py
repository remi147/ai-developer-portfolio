"""LangChain conversation chain with sliding-window memory."""
import os
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

SYSTEM_PROMPT = PromptTemplate(
    input_variables=["history", "input"],
    template=(
        "You are a helpful, knowledgeable AI assistant.\n\n"
        "Conversation history:\n{history}\n\n"
        "Human: {input}\nAI:"
    ),
)


class ChatEngine:
    """Manages per-session LangChain conversation chains."""

    def __init__(self, window_size: int = 10):
        self._sessions: dict[str, ConversationChain] = {}
        self._window_size = window_size

    def _get_chain(self, session_id: str) -> ConversationChain:
        if session_id not in self._sessions:
            llm = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                temperature=0.7,
                streaming=False,
            )
            memory = ConversationBufferWindowMemory(k=self._window_size)
            self._sessions[session_id] = ConversationChain(
                llm=llm, memory=memory, prompt=SYSTEM_PROMPT, verbose=False
            )
        return self._sessions[session_id]

    def chat(self, session_id: str, message: str) -> str:
        """Send a message and receive a reply."""
        chain = self._get_chain(session_id)
        return chain.predict(input=message)

    def clear_session(self, session_id: str) -> None:
        """Remove a session and its memory."""
        self._sessions.pop(session_id, None)
