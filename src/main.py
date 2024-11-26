from langchain_gigachat import GigaChat, GigaChatEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from telegram import Update, Bot
import os
import re


def escape_markdown_v2(text: str) -> str:
    """
    Экранирует все специальные символы для MarkdownV2 в Telegram.
    """
    special_chars = r'_*[]()~`>#+-=|{}.!'
    # Экранируем каждый символ с помощью обратного слэша
    return re.sub(r'([%s])' % re.escape(special_chars), r'\\\1', text)


# DEAL_INFO = input()
# DEAL_INFO = 'Компания Озон хочет взять кредит на 300 миллионов рублей'

dotenv_path = '../.env'
load_dotenv(dotenv_path)

# модели
llm = GigaChat(verify_ssl_certs=False, model='GigaChat-Pro', scope="GIGACHAT_API_PERS")
embeddings = GigaChatEmbeddings(verify_ssl_certs=False, scope="GIGACHAT_API_PERS")

vector_store = FAISS.load_local(
    "../data/faiss_index", embeddings, allow_dangerous_deserialization=True
)

# делаем ретривер и тул
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})
retriever_tool = create_retriever_tool(
    retriever,
    "ozon_financial_results_search",
    "Содержит данные о финансовых и других показателях компании Озон.")

# тул для поиска в интернете
search = TavilySearchResults()

# Ассистент
tools_assistant = [retriever_tool, search]  # тулы ассистента
prompt_assistant = hub.pull("hwchase17/openai-functions-agent")  # промпты для ассистента

# агент
assistant = create_tool_calling_agent(llm, tools_assistant,
                                      prompt_assistant)
assistant_executor = AgentExecutor(agent=assistant, tools=tools_assistant,
                                   verbose=True)

## Топ-менеджер
top_manager_system = (
    'Ты председатель кредитного комитета Сбербанка. '
    'Тебе даны сообщения из разговора о клиенте. Ты должен решить, выдавать кредит или нет. '
    'Для того, чтобы принять решение, ты должен задавать верные вопросы своему помощнику, '
    'у которого есть доступ к Интернету и финансовой отчетности. Узнай у помощника '
    'всю необходимую информацию, после чего вынеси итоговое решение. '
    'Отвечай лишь двумя способами:\n'
    '1. задай вопрос по компании, если хочешь спросить узнать что-то у ассистента;\n'
    '2. ОДОБРЕНО/НЕ ОДОБРЕНО, если ты готов вынести решение по кредиту. '
    'Твое решение должно начинаться с "КРЕДИТ ОДОБРЕН" или "КРЕДИТ НЕ ОДОБРЕН", '
    'после чего ты должен еще написать обоснование своего решения.\n'
    'Отвечай лишь указанными выше способами.'
'''
Примеры твоего ответа:
Какая выручка у клиента?
ОДОБРЕНО
Чем занимается клиент?
Какой размер клиентской базы?
НЕ ОДОБРЕНО
'''
)

# начало диалога для запуска анализа
prompts_top_manager = [
    SystemMessage(top_manager_system)
]


## Граф
class State(TypedDict):
    messages: Annotated[list, add_messages]
    num_qs: int = 0
    bot: Bot
    chat_id: int


system_prompt_decision_making = '''\
Ты топ-менеджер банка.
Тебе дана информация о клиенте, который хочет взять кредит.\
Твоя задача - взвесить все известные о нем факты и принять решение одобрять кредит или нет.
Обосновывай свой выбор.
'''


async def top_manager(state: State):
    if state["num_qs"] > 5:
        return {"messages": state["messages"]}

    message = llm.invoke(state["messages"])
    state["messages"].append(message)

    bot = state["bot"]
    chat_id = state["chat_id"]

    # md_text = message.content.copy()
    # md_text = md_text
    await bot.send_message(chat_id=chat_id,
                           text=f"*Топ-менеджер:* {message.content}",
                           parse_mode='Markdown'
                           )

    return {"messages": state["messages"], "num_qs": state["num_qs"] + 1}


async def ask_or_end(state: State):

    query = state["messages"][-1].content
    bot = state["bot"]
    chat_id = state["chat_id"]

    if '?' in query:
        return 'ask_assistant'
    else:
        await bot.send_message(chat_id=chat_id, text="*Переход к принятию решения*",
                               parse_mode='MarkdownV2')
        return 'make_decision'


async def ask_assistant(state: State):
    query = state["messages"][-1].content
    answer = assistant_executor.invoke({"input": query})

    message = f"*Ответ помощника:* {answer['output']}"
    bot = state["bot"]
    chat_id = state["chat_id"]

    await bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown')
    state["messages"].append(HumanMessage(message))

    return {"messages": state["messages"]}


async def make_decision(state: State):
    messages_for_desision = ([SystemMessage(system_prompt_decision_making)]
                             + state["messages"][1:])
    messages_for_desision.append(HumanMessage(
        'Сделай заключение по клиенту (одобрять кредит или нет), '
        'обоснуй свой выбор, указывай все пункты в формате списка. '
        # 'Пиши официально и формально. Сначала укажи основные факты, затем напиши свое решение.'
    ))
    decision = llm.invoke(messages_for_desision)
    decision_text = decision.content
    decision_text.replace('**', '*')

    bot = state["bot"]
    chat_id = state["chat_id"]

    await bot.send_message(chat_id=chat_id,
                           text='*Итоговое решение:*\n' + decision_text,
                           parse_mode='Markdown'
                           )
    return {"messages": [decision]}


graph_builder = StateGraph(State)
graph_builder.add_node("top_manager", top_manager)
graph_builder.add_node("ask_assistant", ask_assistant)
graph_builder.add_node("make_decision", make_decision)

graph_builder.add_conditional_edges(
    "top_manager",
    ask_or_end,
    {"ask_assistant": "ask_assistant",
     'make_decision': "make_decision"})
graph_builder.add_edge('ask_assistant', "top_manager")
graph_builder.add_edge(START, "top_manager")
graph_builder.add_edge("make_decision", END)
graph = graph_builder.compile()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id

    # начальное состояние графа
    state = {
        "messages": prompts_top_manager,
        "num_qs": 0,
        "bot": context.bot,
        "chat_id": chat_id
    }
    context.user_data["state"] = state

    await update.message.reply_text("Здравствуйте! Введите параметры сделки, чтобы увидеть решение."
                                    "\nПример: Компания Озон хочет взять кредит на 300 миллионов рублей")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state = context.user_data.get("state")
    if not state:
        await update.message.reply_text("Пожалуйста, начните с команды /start.")
        return

    # убедимся, что все необходимые ключи присутствуют
    required_keys = ["messages", "num_qs", "bot", "chat_id"]
    for key in required_keys:
        if key not in state:
            await update.message.reply_text(f"Ошибка: отсутствует ключ '{key}' в состоянии.")
            return

    user_input = update.message.text
    state["messages"].append(HumanMessage(user_input))

    # запуск графа
    await graph.ainvoke(state)


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Бот остановлен.")


if __name__ == "__main__":
    app = ApplicationBuilder().token(os.environ["TELEGRAM_TOKEN"]).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stop", stop))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot is running...")
    app.run_polling()
