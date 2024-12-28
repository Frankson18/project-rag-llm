# Projeto LLM-RAG: Assistente acadêmico com LLM

Este é um projeto para a disciplina de **TÓPICOS AVANÇADOS EM INTELIGÊNCIA COMPUTACIONAL 1** do Mestrado Profissional em Tecnologia da Informação da UFRN e implementa um chat acadêmico que auxilia pesquisadores a acessar e compreender informações sobre artigos científicos utilizando uma base de dados autoral de artigos retiras do sofware zotero. O assistente utiliza **Streamlit** como interface, **LangChain** para a construção da lógica, e **Chroma** para a indexação vetorial de dados. Além disso, o ambiente de desenvolvimento é configurado utilizando **DevContainer** e **Poetry**.

## Contexto

Esse projeto surgiu de uma necessidade pessoal, quando no processo de revisão de literatura científica, me deparei com um grande número de artigos salvos em ferramentas como o Zotero para serem estudados. Porém, encontrar tópicos relevantes em meio a esse mar de artigos pode ser um desafio, especialmente quando se busca agilidade sem a necessidade de buscas profundas manuais. Este projeto aborda essa problemática ao oferecer uma solução baseada em RAG, permitindo que perguntas sejam feitas diretamente a um LLM, agilizando o trabalho de revisão e recuperação de informações relevantes.

## Funcionalidades

1. **Tradução Automática:**
   - Traduz colunas específicas de arquivos CSV para português, incluindo nomes de colunas e conteúdo.
   
2. **Chunking de Textos:**
   - Divide grandes textos em partes menores para facilitar o processamento.
   
3. **Armazenamento Vetorial:**
   - Cria ou carrega um armazenamento vetorial com embeddings para busca eficiente de documentos.

4. **Interface Interativa:**
   - Permite aos usuários interagir com o modelo, fazer perguntas e receber respostas baseadas nos dados indexados.

## Estrutura do Projeto

- **`streamlit_app.py`**: Arquivo principal contendo a lógica da aplicação.
- **`data/`**: Pasta para armazenamento de arquivos CSV e bases de dados.
- **`.env`**: Arquivo para configuração de variáveis de ambiente.
- **`Dockerfile`** e **`devcontainer.json`**: Arquivos de configuração para o ambiente DevContainer.
- **`pyproject.toml`**: Arquivo de configuração do Poetry para gerenciar dependências.

## Configuração do Ambiente

### Requisitos

- Docker e DevContainer configurados.
- Python 3.10+

### Configuração Inicial

1. **Clone o Repositório:**
   ```bash
   git clone https://github.com/Frankson18/project-rag-llm
   cd project-rag-llm
   ```

2. **Abra o DevContainer:**
   Certifique-se de que o DevContainer está configurado. Abra o projeto em um editor compatível, como VS Code, e inicialize o DevContainer.

3. **Instale Dependências:**
   O Poetry será executado automaticamente no DevContainer para instalar as dependências listadas em `pyproject.toml`.

4. **Configure o `.env`:**
   Crie um arquivo `.env` na raiz do projeto com as seguintes variáveis:
   ```env
   OPENAI_API_KEY=YOUR_API_KEY
   PERSIST_DIRECTORY=/tmp/chroma_db_openai
   ```

### Execução

1. **Execute a Aplicação:**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Acesse no Navegador:**
   Abra `http://localhost:8501` para interagir com a aplicação.

## Estrutura do Código

### Principais Funções

1. **`translate_csv_columns`**:
   Tradução de colunas e seus nomes em um arquivo CSV usando um pipeline de LLM.

2. **`chunk_data`**:
   Divide textos em partes menores para processamento eficiente.

3. **`get_or_create_vector_store`**:
   Gera ou carrega um armazenamento vetorial para busca por similaridade.

4. **`configure_streamlit_interface`**:
   Configura a interface interativa no Streamlit para consultas e interações com o modelo.

5. **`main`**:
   Função principal que carrega variáveis de ambiente, traduz arquivos CSV e inicializa a interface Streamlit.

## Base de Dados

A base de dados utilizada é um arquivo CSV exportado do Zotero. Certifique-se de que o arquivo está localizado na pasta `data/` e nomeado como `weak_supervision.csv`. Para tradução, o arquivo resultante será salvo como `weak_supervision_translate.csv`.

## Notas Importantes

- **Cache no Streamlit**: A função `get_retriever` utiliza o cache para melhorar o desempenho.
- **Limitações de Tradução**: As traduções dependem de um modelo de linguagem, que pode apresentar imprecisões.
- **A função de tradução**:A função esta comentada, pois o arquivo traduzido já esta disponivel.
- **Recriação do Vector Store**: Utilize a flag `force_create=True` em `get_or_create_vector_store` caso precise forçar a recriação do armazenamento vetorial.

## Melhorias Futuras

1. Implementar **routing de LLMs** para permitir o uso de vários modelos além da OpenAI.
2. Utilizar um modelo de embeddings especializado em multilinguagem e com foco no português.
3. Aperfeiçoar a busca no sistema RAG para aumentar a precisão e relevância dos resultados.


