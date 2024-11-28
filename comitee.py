from langchain_gigachat import GigaChat, GigaChatEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from typing import Annotated
from prompts import prompts_dict
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from telegram import Update, Bot
import os
import re

load_dotenv()
NUM_Q_THRESHOLD = 10  # сколько максимум вопросов задаст председатель


def escape_markdown_v2(text: str) -> str:
    """
    Экранирует все специальные символы для MarkdownV2 в Telegram.
    """
    special_chars = r'_*[]()~`>#+-=|{}.!'
    # Экранируем каждый символ с помощью обратного слэша
    return re.sub(r'([%s])' % re.escape(special_chars), r'\\\1', text)


# DEAL_INFO = input()
# DEAL_INFO = 'Компания Озон хочет взять кредит на 300 миллионов рублей'

# модели
llm = GigaChat(verify_ssl_certs=False, model='GigaChat-Pro', scope="GIGACHAT_API_PERS", top_p=0.5)
embeddings = GigaChatEmbeddings(verify_ssl_certs=False, scope="GIGACHAT_API_PERS")

vector_store = FAISS.load_local(
    "./data/faiss_comitee_bot", embeddings, allow_dangerous_deserialization=True
)

# делаем ретривер и тул
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})
retriever_tool = create_retriever_tool(
    retriever,
    "deal_docs_search",
    "Содержит данные из полезных для анализа документов о клиенте: \
    заключения служб, информация о клиенте, условиях кредитования.")

# тул для поиска в интернете
search = TavilySearchResults()

# тулы для агентов
tools = [retriever_tool, search]

# агенты-участники
agents_dict = {}
agents_names = ['law_member', 'reputation_member', 'credit_manager']

for agent_name in agents_names:
    agent_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompts_dict[f'system_{agent_name}']),
            ("human", "Вопрос председателя: {input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, agent_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    agents_dict[agent_name] = agent_executor

# пердседатель комитета
prompts_comitee_head = [
    SystemMessage(prompts_dict['system_comitee_head']),
]


# граф
class State(TypedDict):
    messages: Annotated[list, add_messages]
    num_qs: int = 0
    bot: Bot
    chat_id: int


async def comitee_head(state: State):
    """
    Председатель говорит, что делать дальше
    """
    # если уже было задано очень много вопросов
    if state['num_qs'] > NUM_Q_THRESHOLD:
        return {"messages": [HumanMessage("{'decision': 'ok'}")]}

    return {"messages": [llm.invoke(state["messages"])],
            'num_qs': state['num_qs'] + 1}


async def router(state: State):
    """
    Определяем, куда идет диалог (вопрос участнику комитета / принятие решения)
    """
    comitee_head_message = state["messages"][-1].content
    bot = state["bot"]
    chat_id = state["chat_id"]

    try:
        dict_msg = eval(comitee_head_message)
        # определяем куда идти дальше
        way_key = list(dict_msg.keys())[0]
        # если есть ключ определяет ноду
        if way_key in ['law', 'reputation', 'decision', 'credit']:
            if way_key == 'law':
                await bot.send_message(chat_id=chat_id,
                                       text='*Вопрос Председателя Юристу:* ' + dict_msg[way_key],
                                       parse_mode='Markdown')
            elif way_key == 'reputation':
                await bot.send_message(chat_id=chat_id,
                                       text='*Вопрос Председателя Представителю Подразделения Безопасности:* ' + dict_msg[way_key],
                                       parse_mode='Markdown')
            elif way_key == 'credit':
                await bot.send_message(chat_id=chat_id,
                                       text='*Вопрос Председателя Кредитному Менеджеру:* ' + dict_msg[way_key],
                                       parse_mode='Markdown')
            elif way_key == 'decision':
                await bot.send_message(chat_id=chat_id, text="*Переход к формированию заключения КО*",
                                       parse_mode='Markdown')
            return way_key
        # если непонятно, куда идти, принимаем решение
        else:
            await bot.send_message(chat_id=chat_id, text="*Переход к формированию заключения КО*",
                                   parse_mode='Markdown')
            return 'decision'
    except:
        await bot.send_message(chat_id=chat_id, text="*Переход к формированию заключения КО*",
                               parse_mode='Markdown')
        return 'decision'


async def ask_law(state: State):
    """
    Спрашиваем Юриста
    """
    bot = state["bot"]
    chat_id = state["chat_id"]
    query = eval(state["messages"][-1].content)['law']

    answer = agents_dict['law_member'].invoke({"input": query})
    message = f"*Ответ Юриста:* {answer['output']}"
    await bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown')

    return {"messages": [HumanMessage('Ответ Юриста: ' + answer['output'])]}


async def ask_reputation(state: State):
    """
    Спрашиваем Безопасника
    """
    bot = state["bot"]
    chat_id = state["chat_id"]
    query = eval(state["messages"][-1].content)['reputation']

    answer = agents_dict['reputation_member'].invoke({"input": query})
    message = f"*Ответ Представителя Подразделения Безопасности:* {answer['output']}"
    await bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown')

    return {"messages": [HumanMessage('Ответ Представителя Подразделения Безопасности: '
                                      + answer['output'])]}


async def ask_credit(state: State):
    """
    Спрашиваем Кредитного инспектора
    """
    bot = state["bot"]
    chat_id = state["chat_id"]
    query = eval(state["messages"][-1].content)['credit']

    answer = agents_dict['credit_manager'].invoke({"input": query})
    message = f"*Ответ Кредитного менеджера:* {answer['output']}"
    await bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown')

    return {"messages": [HumanMessage('Ответ Кредитного менеджера: '
                                      + answer['output'])]}


async def make_decision(state: State):
    """
    Принимаем итоговое решение на основе всего обсуждения на комитете
    """
    bot = state["bot"]
    chat_id = state["chat_id"]

    messages_for_decision = ([SystemMessage(prompts_dict['system_decision_making'])]
                             + state["messages"][1:-1])
    messages_for_decision.append(HumanMessage(prompts_dict['make_conclusion_prompt']))
    decision = llm.invoke(messages_for_decision)

    await bot.send_message(chat_id=chat_id,
                           text='*Итоговое решение:*\n' + decision.content,
                           parse_mode='Markdown'
                           )

    return {"messages": [decision]}

graph_builder = StateGraph(State)
graph_builder.add_node("comitee_head", comitee_head)
graph_builder.add_node("law_member", ask_law)
graph_builder.add_node("reputation_member", ask_reputation)
graph_builder.add_node("credit_manager", ask_credit)
graph_builder.add_node("make_decision", make_decision)

graph_builder.add_edge(START, "comitee_head")
graph_builder.add_conditional_edges(
    "comitee_head",
    router,
    {"law": "law_member",
     "reputation": "reputation_member",
     "credit": 'credit_manager',
     'decision': "make_decision"})
graph_builder.add_edge('law_member', "comitee_head")
graph_builder.add_edge('reputation_member', "comitee_head")
graph_builder.add_edge('credit_manager', "comitee_head")
graph_builder.add_edge("make_decision", END)
graph = graph_builder.compile()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id

    # начальное состояние графа
    state = {
        "messages": prompts_comitee_head,
        "num_qs": 0,
        "bot": context.bot,
        "chat_id": chat_id
    }
    context.user_data["state"] = state

    await update.message.reply_text(
'''\
Здравствуйте! Введите параметры сделки, чтобы увидеть решение.
Пример: Компания ООО Никомет хочет взять индивидуальный овердрафт на 10 миллионов рублей сроком на 12 месяцев.\
''')


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
    bot = state["bot"]
    await bot.send_message(chat_id=state['chat_id'],
                           text='*Начало заседания Кредитного Комитета*',
                           parse_mode='Markdown'
                           )
    await graph.ainvoke(state, {"recursion_limit": 100})


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Бот остановлен.")


if __name__ == "__main__":
    app = ApplicationBuilder().token(os.environ["TELEGRAM_TOKEN"]).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stop", stop))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot is running...")
    app.run_polling()
