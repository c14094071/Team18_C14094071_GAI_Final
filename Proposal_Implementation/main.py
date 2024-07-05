import chromadb
from langchain.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain_openai import ChatOpenAI
import os
import sys


def get_summary(text_content):

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0)
    _prompt = """根據使用者的描述，生成相關的法律條文來解決使用者遇到的法律困境。請考慮相關法律，例如《合同法》、《勞動法》、《消費者權益保護法》等，並依照使用者情況舉例該法條的使用情境，並多多提及使用者輸入的關鍵字，不需要任何建議。列點控制在三個以內，列點數量越少越好，同時要維持正確性並且與使用者內容高度相關，並列出編號清單。
使用者描述如下：
  {text}

  請特別注意：
  1. 引用具體法律條文的適用情境，但不要提到法條名稱。
  2. 只要說明使用情境，不要提及任何法律的字眼。
  3. 每個列點結尾務必換行
  4. 盡量重複提及使用者所描述的關鍵字，例如車禍、偽造文書、等等

  """
    prompt = PromptTemplate.from_template(_prompt)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    mp_chain = MapReduceChain.from_params(llm, prompt, text_splitter)
    # text_content = df[col_name][0]
    # print("textcontent:",text_content)
    reduce_docs = mp_chain.run(text_content)
    # print(reduce_docs)

    import re

    def merge_points_with_newlines(text):
        # Split text into points
        points = text.strip().split("\n\n")
        merged_points = []
        for point in points:
            # Remove **, newline characters, and unnecessary spaces within each point
            point = re.sub(r"\*\*", "", point)
            point = re.sub(r"\n\s*", " ", point)
            # Merge sub-points
            point = re.sub(r"-\s*", "", point)
            # Fix numbering and colons
            point = re.sub(r"(\d\.\s[^\:]+)：", r"\1:", point)
            merged_points.append(point)
        # Join points with newlines
        return "\n".join(merged_points)

    merged_text = merge_points_with_newlines(reduce_docs)
    return_text = ""
    for text in merged_text:
        return_text += text
    print("return_text:", return_text)
    return return_text


def search_similar_case(user_text):
    client = chromadb.PersistentClient(path="my_local_data")
    collection1_name = "clean_doc_collection"
    collection1 = client.get_or_create_collection(name=collection1_name)
    # user_text = "我砍伐了自家前面的森林取用木材，但卻被政府提告。"
    ##user_text = "我偽造文書，結果被公司告，我該怎麼辦"
    user_prompt = f"""使用者輸入: {user_text}

    根據使用者的描述，生成相關的法律條文來解決使用者遇到的法律困境。請考慮相關法律，例如《合同法》、《勞動法》、《消費者權益保護法》等，並給出具體的法律條文及其解釋，不需要任何建議"""

    # user_prompt = str(PromptTemplate.from_template(user_prompt))
    # llm = ChatOpenAI(model_name="gpt-4o", temperature=0.3)
    # gen_text = llm.invoke(user_prompt)
    reduce_docs = get_summary(user_text)
    # reduce_docs_list = reduce_docs.split("\n")

    # print("reduce_doc_list:")
    # print(reduce_docs_list)
    # for doc in reduce_docs_list:
    #    docs = collection1.query(query_texts=doc, n_results=3)
    #    print(docs)
    docs = collection1.query(query_texts=(reduce_docs + reduce_docs), n_results=5)
    print("============================Retrieve Chunks===========================")
    print(docs)
    # print(gen_text.content)
    # docs = collection1.query(query_texts=gen_text.content, n_results=10)
    # 取得metadatas的source的數字，且不重複
    unique_sources = set()
    for metadata_list in docs["metadatas"]:
        for metadata in metadata_list:
            unique_sources.add(metadata["source"])

    # 將結果轉換為列表並排序
    unique_sources_list = sorted(list(unique_sources))
    print(unique_sources_list)

    print("============================摘要===========================")
    collection2_name = "sum_doc_collection"
    collection2 = client.get_or_create_collection(name=collection2_name)
    for source_id in unique_sources_list:
        doc = collection2.get(where={"source": str(source_id)})
        print(doc["ids"])
        print(doc["metadatas"])
        print(doc["documents"])
    print("============================裁判原文===========================")
    collection3_name = "ori_doc_collection"
    collection3 = client.get_or_create_collection(name=collection3_name)
    for source_id in unique_sources_list:
        doc = collection3.get(where={"source": str(source_id)})
        print(doc["ids"])
        print(doc["metadatas"])
        print(doc["documents"])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <input_value>")
        sys.exit(1)

    input_value = sys.argv[1]
    search_similar_case(input_value)
