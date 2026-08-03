import re
from markdownParser import *

import os
import gc
import logging
import numpy as np

logger = logging.getLogger(__name__)

def calc_sim(A, B, model):
    embedding_A = model.encode(A)  # Shape: (len(A), embedding_dim)
    embedding_B = model.encode(B)  # Shape: (len(B), embedding_dim)

    norm_A = np.linalg.norm(embedding_A, axis=1, keepdims=True)  # Shape: (len(A), 1)
    norm_B = np.linalg.norm(embedding_B, axis=1, keepdims=True)  # Shape: (len(B), 1)


    similarity_matrix = np.dot(embedding_A, embedding_B.T) / (norm_A * norm_B.T)

    return similarity_matrix

def normalize_sims(sims):
    return sims


def soft_heading_recall(G, P, model):
    #print(G)
    #print(P)

    def soft_cardinality(T):
        card = 0
        sims = calc_sim(T,T,model)
        
        sims = normalize_sims(sims)
        
        for i in range(len(T)):
            tmp_sum = sum(sims[i])
            #print(sum(sims[i]),"below",1/tmp_sum)
            #print(sims[i])
            card += (1 / tmp_sum) if tmp_sum != 0 else 0
        #print(f'card={card}')
        return card

    def intersection_soft_cardinality(R, G):
        R_card = soft_cardinality(R)
        G_card = soft_cardinality(G)
        RG_union = list(set(R + G))
        union_card = soft_cardinality(RG_union)  
        return R_card + G_card - union_card

    # Soft Cardinality for R and R ∩ G
    card_R = soft_cardinality(G)
    card_R_intersect_G = intersection_soft_cardinality(G, P)

    # Soft Heading Recall
    soft_recall = card_R_intersect_G / card_R if card_R != 0 else 0
    return soft_recall


def get_title_list(psg_node:MarkdownNode):
    res = []
    if "root" not in psg_node.title and "Abstract:" not in psg_node.title :
        res.append(psg_node.title[:512])
    for child in psg_node.children:
        tmp_res = get_title_list(child)
        res.extend(tmp_res)
        
    return res

def eval_SHRecall(target_survey,psg_node: MarkdownNode,model):
    
    target_titles = []
    for section in target_survey['structure']:
        if len(section['content']) < 100:
            continue
        target_titles.append(section['title'])
        #subtitles_map.append(section['title'])
        
    gen_titles = get_title_list(psg_node)
    
    if len(gen_titles) == 0:
        return 0
    
    return soft_heading_recall(target_titles,gen_titles,model)
    
    
def get_target_title_structure(target_survey,id,level):
    res = ""
    for section in target_survey['structure']:
        if section['parent_id'] == id:
            res += "#"*level + " " + section['title'] + "\n"
            res += get_target_title_structure(target_survey,section['id'],level+1)
            res += '\n'
    return res   

def get_generate_title_structure(psg_node:MarkdownNode,level):
    res = ""
    if "root" not in psg_node.title and "Abstract:" not in psg_node.title :
        res += "#"*level + " " + psg_node.title + "\n"
    for child in psg_node.children:
        tmp_res = get_generate_title_structure(child,level+1)
        res += tmp_res
    return res
    
    
def gen_title_structure_compare_prompt(target_titles, generated_titles):
    prompt = f"""You are an AI evaluator. Your task is to compare the generated titles with the target titles and assign a score from 0 to 5 based on their similarity in structure, meaning, and wording.

### Target Titles:
{target_titles}

### Generated Titles:
{generated_titles}

## **Scoring Criteria:**

- **0 – Completely Different:**  
  - Nearly no words in common.  
  - Completely different meanings.  
  - No similarity in structure or phrasing.  

- **1 – Somewhat Different:**  
  - Few words overlap, but they are not key terms.  
  - The meaning is somewhat related but mostly different.  
  - The sentence structures are significantly different.  

- **2 – Somewhat Similar:**  
  - Some key words are shared, but others are different.  
  - The general topic is the same, but the emphasis may differ.  
  - The sentence structures are different but not entirely unrelated.  

- **3 – Similar:**  
  - Several key words are shared.  
  - The meaning is largely the same with slight variations.  
  - The structure is somewhat similar, but there may be word substitutions.  

- **4 – Very Similar:**  
  - Most key words match.  
  - The meaning is nearly identical.  
  - The phrasing and structure are very close, with minor rewording.  

- **5 – Almost Identical:**  
  - Nearly all key words match exactly.  
  - The meaning is fully preserved.  
  - The phrasing and structure are identical or differ only in trivial ways.  

### **Instructions:**  
Analyze the generated titles based on the criteria above and provide a single score between 0 and 5.  
**Output only the score as a number, without any additional explanation or comments.**
"""
    return prompt

    
def chat_openai(prompt, client, try_number, model="gpt-4o", max_retries=5):
    """판정 LLM 을 호출해 0-5 정수 하나를 받아온다.

    응답이 '0'~'5' 한 글자와 정확히 일치하지 않으면 재귀적으로 재시도하며,
    max_retries 회에 도달하면 포기하고 None 을 돌려준다.
    (상한이 없으면 형식을 못 맞추는 모델에서 호출이 무한히 늘어난다.)
    """
    if try_number >= max_retries:
        logger.error(
            "SQS 판정 실패: %d회 재시도 후에도 0-5 형식 응답을 받지 못했다 (model=%s)",
            max_retries, model,
        )
        return None
    #print(f"Try {try_number} time")
    #print(prompt)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            max_tokens=100
        )
        #print(f"Answer:{response.choices[0].message.content}")
        ans = None
        content = response.choices[0].message.content
        if content is None or not re.match(r'^[0-5]$', content):
            logger.warning(
                "SQS 응답 형식 불일치 (시도 %d/%d, model=%s): %r",
                try_number + 1, max_retries, model, (content or "")[:100],
            )
            ans =  chat_openai(prompt,client,try_number + 1,model,max_retries)
        else :
            ans = int(content)
        return ans
    except Exception as e:
        logger.warning("SQS API 호출 실패 (시도 %d/%d): %s", try_number + 1, max_retries, e)
        return chat_openai(prompt,client,try_number + 1,model,max_retries)
    

def eval_structure_quality_client(target_survey,psg_node:MarkdownNode,client,model="gpt-4o",max_retries=5):
    target_titles = ""
    for section in target_survey['structure']:
        if section['title'] == "root":
            target_titles = get_target_title_structure(target_survey,section['id'],1)
            break
    
    gen_titles = get_generate_title_structure(psg_node,1)
    
    if len(gen_titles)<5:
        return 0
    
    prompt = gen_title_structure_compare_prompt(target_titles,gen_titles)

    score = chat_openai(prompt,client,0,model,max_retries)
    if score is None:
        # 판정 실패는 0점으로 떨어뜨린다. 위 logger.error 로 이미 기록됐다.
        logger.error("SQS 를 0 으로 처리한다 (survey_title=%r)", target_survey.get('survey_title'))
        return 0
    return score
    
