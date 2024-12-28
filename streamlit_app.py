import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages.base import BaseMessage
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# tranduz texto usando llm
def translate_text_with_chain(text, chain):
    """
    Tradução de texto usando um chain (pipeline).

    Args:
        text (str): Texto a ser traduzido.
        chain: Pipeline para realizar a tradução.

    Returns:
        str: Texto traduzido ou o original em caso de erro.
    """
    try:
        response = chain.invoke({"text": text})
        return response.content 
    except Exception as e:
        print(f"Error translating text: {e}")
        return text 

# Carrega arquivo CSV, seleciona colunas especificas, e traduz espeficias colunas
def translate_csv_columns(file_path, columns_to_select, columns_to_translate, output_file):
    """
    Traduz colunas de um arquivo CSV e seus nomes.

    Args:
        file_path (str): Caminho do arquivo CSV.
        columns_to_select (list): Colunas a serem selecionadas do CSV.
        columns_to_translate (list): Colunas a serem traduzidas.
        output_file (str): Caminho do arquivo de saída traduzido.

    Returns:
        None
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    prompt = PromptTemplate(
        input_variables=["text"],
        template="Translate the following text into Portuguese. Don't write anything else, just the translation: {text}"
    )

    chain = prompt | llm

    df = pd.read_csv(file_path)

    df = df[columns_to_select]

    for column in columns_to_translate:
        if column in df.columns:
            df[column + '_translated'] = df[column].apply(lambda x: translate_text_with_chain(str(x), chain))
        else:
            print(f"Column '{column}' not found in the DataFrame.")

    translated_column_names = {col: translate_text_with_chain(col, chain) for col in df.columns}

    df.rename(columns=translated_column_names, inplace=True)

    df.to_csv(output_file, index=False)
    print(f"Translated CSV saved to {output_file}")

#Define uma função para dividir grandes textos em partes menores para facilitar o processamento.
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    """
    Divide grandes textos em partes menores.

    Args:
        data (list): Lista de documentos ou textos.
        chunk_size (int): Tamanho de cada parte.
        chunk_overlap (int): Quantidade de sobreposição entre partes.

    Returns:
        list: Lista de partes divididas.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ".", ",", " ", ""])
    chunks = text_splitter.split_documents(data)
    return chunks

def get_or_create_vector_store(chunks=None, persist_directory='/tmp/chroma_db_openai', force_create=False):
    """
    Cria ou carrega um armazenamento vetorial.

    Args:
        chunks (list): Documentos para gerar embeddings (necessário ao criar).
        persist_directory (str): Caminho para armazenar ou carregar.
        force_create (bool): Força a recriação mesmo se existir.

    Returns:
        Chroma: Instância do armazenamento vetorial.
    """
    # embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

    if os.path.exists(persist_directory) and not force_create:
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        if chunks is None or len(chunks) == 0:
            raise ValueError("To create a new vector store, 'chunks' must be provided and cannot be empty.")

        vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
    
    return vector_store

def get_question(input):
    """
    Extrai uma pergunta a partir do input fornecido.

    Args:
        input (Union[str, dict, BaseMessage]): O input pode ser uma string, um dicionário contendo a chave 'question' 
            ou uma instância de `BaseMessage`.

    Returns:
        Union[str, None]: A pergunta extraída como uma string, ou None se o input for vazio ou inválido.

    Raises:
        Exception: Se o input não for uma string, um dicionário com a chave 'question' ou um objeto do tipo `BaseMessage`.
    """
    if not input:
        return None
    elif isinstance(input, str):
        return input
    elif isinstance(input, dict) and 'question' in input:
        return input['question']
    elif isinstance(input, BaseMessage):
        return input.content
    else:
        raise Exception("Esperado string ou dicionário com a chave 'question' como input da RAG chain.")

def format_docs(docs):
    """
    Formata uma lista de documentos concatenando seus conteúdos com quebras de linha duplas.

    Args:
        docs (list): Uma lista de objetos de documento, cada um deve possuir o atributo `page_content`.

    Returns:
        str: Uma string única com o conteúdo de todos os documentos, separados por quebras de linha duplas.
    """
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource
def get_retriever():
    """
    Cria e armazena em cache um retriever para busca por similaridade de documentos.

    Esta função carrega os dados de um arquivo CSV, divide os dados em blocos para 
    processamento eficiente e cria uma vector store para ser usada como retriever. 
    O retriever é configurado para usar o método Maximal Marginal Relevance (MMR) na busca.

    Returns:
        retriever: Um objeto retriever configurado para busca por similaridade.

    Notas:
        - O caminho do arquivo CSV está fixado como 'data/weak_supervision_translate.csv'.
        - A criação do vector store pode ser forçada com `force_create=True` quando necessário.
    """
    loader = CSVLoader(file_path="data/weak_supervision_translate.csv")
    data = loader.load()
    chunks = chunk_data(data, chunk_size=8000, chunk_overlap=128)
    vector_store = get_or_create_vector_store(chunks,os.getenv("PERSIST_DIRECTORY"), force_create=True)
    retriever = vector_store.as_retriever(search_type="mmr")
    return retriever

# Configura uma interface Streamlit para interagir com o modelo e realizar perguntas com base em dados científicos.
def configure_streamlit_interface():

    st.set_page_config(page_title="Assistente Acadêmico")
    st.title("Assistente Acadêmico")

    msgs = StreamlitChatMessageHistory()

    if len(msgs.messages) == 0:
        msgs.add_ai_message("Como eu posso ajudar?")

    model = ChatOpenAI(model='gpt-4o-mini', temperature=0.0)
    system_prompt = """Você é um assistente de IA especializado em ajudar acadêmicos a acessar e compreender informações sobre artigos científicos. Use o contexto fornecido e o histórico do usuário para responder às perguntas de forma clara e objetiva. Sempre inclua as seguintes informações para cada artigo relevante:
        Ano de publicação
        Autor(es)
        Título
        Nota: Resumo traduzido para o português

    Contexto: {context}

    Se o contexto não tiver informações suficientes para responder à pergunta, diga 'Não sei com base nas informações fornecidas.'
    """

    prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{question}"),
            ]
        )

    ensemble_retriever = get_retriever()
    chain = {
                "context": RunnableLambda(get_question) | ensemble_retriever | format_docs,
                "question": RunnablePassthrough()
            } | prompt | model

    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: msgs, 
        input_messages_key="question",
        history_messages_key="history",
    )

    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    if prompt := st.chat_input('Envie uma mensagem para o assistente'):
        st.chat_message("human").write(prompt)
        config = {"configurable": {"session_id": "any"}}
        response = chain_with_history.invoke({"question": prompt},config)
        st.chat_message("ai").write(response.content)

def main():
    #Carrega variáveis de ambiente do arquivo `.env` para uso no notebook.
    load_dotenv()

    # traduz CSV columns
    columns = ['Key', 'Item Type', 'Publication Year', 'Author', 'Title', 'DOI', 'Url', 'Abstract Note', 'Date', 'Publisher']
    #translate_csv_columns('data/weak_supervision.csv', columns, ['Title', 'Abstract Note'], 'data/weak_supervision_translate.csv')

    # Configura Streamlit Interface
    configure_streamlit_interface()


if __name__ == "__main__":
    main()