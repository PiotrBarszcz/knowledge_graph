import sys
from yachalk import chalk
sys.path.append("..")

import json
import ollama.client as client
import re

def extractConcepts(prompt: str, metadata={}, model="mistral-openorca:latest"):
    SYS_PROMPT = (
        "Your task is extract the key concepts (and non personal entities) mentioned in the given context. "
        "Extract only the most important and atomistic concepts, if  needed break the concepts down to the simpler concepts."
        "Categorize the concepts in one of the following categories: "
        "[event, concept, place, object, document, organisation, condition, misc]\n"
        "Format your output as a list of json with the following format:\n"
        "[\n"
        "   {\n"
        '       "entity": The Concept,\n'
        '       "importance": The concontextual importance of the concept on a scale of 1 to 5 (5 being the highest),\n'
        '       "category": The Type of Concept,\n'
        "   }, \n"
        "{ }, \n"
        "]\n"
    )
    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=prompt)
    try:
        result = json.loads(response)
        result = [dict(item, **metadata) for item in result]
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    return result


def graphPrompt(input: str, metadata={}, model="mistral-openorca:latest"):
    if model == None:
        model = "mistral-openorca:latest"

    # model_info = client.show(model_name=model)
    # print( chalk.blue(model_info))

    SYS_PROMPT = (
        "You are a network graph maker who extracts terms and their relations from a given context. "
        "You are provided with a context chunk (delimited by ```) Your task is to extract the ontology "
        "of terms mentioned in the given context. These terms should represent the key concepts as per the context. \n"
        "Thought 1: While traversing through each sentence, Think about the key terms mentioned in it.\n"
            "\tTerms may include object, entity, location, organization, person, \n"
            "\tcondition, acronym, documents, service, concept, etc.\n"
            "\tTerms should be as atomistic as possible\n\n"
        "Thought 2: Think about how these terms can have one on one relation with other terms.\n"
            "\tTerms that are mentioned in the same sentence or the same paragraph are typically related to each other.\n"
            "\tTerms can be related to many other terms\n\n"
        "Thought 3: Find out the relation between each such related pair of terms. \n\n"
        "Format your output as a list of json. Each element of the list contains a pair of terms"
        "and the relation between them, like the follwing: \n"
        "[\n"
        "   {\n"
        '       "node_1": "A concept from extracted ontology",\n'
        '       "node_2": "A related concept from extracted ontology",\n'
        '       "edge": "relationship between the two concepts, node_1 and node_2 in one or two sentences"\n'
        "   }, {...}\n"
        "]"
    )

    USER_PROMPT = f"context: ```{input}``` \n\n output: "
    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT)
    try:
        result = json.loads(response)
        result = [dict(item, **metadata) for item in result]
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    return result

MF_PROMPT_EN = """
You will score how strongly the text relates to each Moral Foundations dimension.

Important: News can express moral content IMPLICITLY (through who is harmed, blamed, protected, or restricted).

Use an integer scale 0–10:
0 = not relevant
2 = weak / background mention
5 = clearly present
8 = very strong
10 = dominant theme

Rubric (use these anchors):
- Care/Harm: injury, death, suffering, victims, protection, rescue, tragedy
- Fairness/Cheating: corruption, fraud, inequality, justice, rights violations, discrimination
- Loyalty/Betrayal: nation/group loyalty, betrayal, treason, insiders vs outsiders
- Authority/Subversion: leaders, police, courts, state, rule of law, obedience, unrest, legitimacy
- Sanctity/Degradation: purity, disgust, desecration, moral contamination, “depravity”
- Liberty/Oppression: censorship, coercion, surveillance, restrictions, authoritarian control, freedom

Calibration rule:
If the story includes death or physical harm, Care/Harm MUST be >= 5 (unless clearly irrelevant).
Avoid returning all zeros unless the text is truly morally neutral.

Return ONLY valid JSON in exactly this format:
{{
  "care_harm": 0,
  "fairness_cheating": 0,
  "loyalty_betrayal": 0,
  "authority_subversion": 0,
  "sanctity_degradation": 0,
  "liberty_oppression": 0
}}

Text:
\"\"\"{text}\"\"\"
"""




def _extract_first_json_object(s: str):
    # usuń ewentualne code fences
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)

    start = s.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    return None


MF_KEYS = [
    "care_harm",
    "fairness_cheating",
    "loyalty_betrayal",
    "authority_subversion",
    "sanctity_degradation",
    "liberty_oppression",
]

def moralFoundations(text: str, metadata=None, model="zephyr:latest"):
    if metadata is None:
        metadata = {}

    SYS_PROMPT = "Return ONLY valid JSON. No markdown. No extra text."


    USER_PROMPT = MF_PROMPT_EN.format(text=text)

    response, _ = client.generate(
        model_name=model,
        system=SYS_PROMPT,
        prompt=USER_PROMPT
    )

    # 1) spróbuj całość
    obj = None
    try:
        obj = json.loads(response)
    except Exception:
        pass

    # 2) jeśli nie, wyciągnij pierwszy obiekt JSON
    if obj is None:
        js = _extract_first_json_object(response)
        if js:
            try:
                obj = json.loads(js)
            except Exception:
                obj = None

    if not isinstance(obj, dict):
        return None

    # 3) Normalizacja: zawsze zwróć 6 kluczy (braki -> 0), int, zakres 0..10
    cleaned = {}
    for k in MF_KEYS:
        v = obj.get(k, 0)
        if isinstance(v, bool):
            v = int(v)
        if isinstance(v, (int, float)):
            v = int(v)
            if v < 0: v = 0
            if v > 10: v = 10
        else:
            # jeśli brakuje albo jest śmieć -> 0 (unikamy NaN w DataFrame)
            v = 0
        cleaned[k] = v
    
    return dict(cleaned, **metadata)
    

IDEOLOGY_FROM_MF_PROMPT_EN = """
You will classify political ideology using ONLY Moral Foundations scores.

Return ONLY valid JSON in exactly this format:
{{
  "predicted_ideology": "liberal" or "conservative",
  "confidence": 0-10
}}

Scores (0–10):
Care/Harm: {care_harm}
Fairness/Cheating: {fairness_cheating}
Loyalty/Betrayal: {loyalty_betrayal}
Authority/Subversion: {authority_subversion}
Sanctity/Degradation: {sanctity_degradation}
Liberty/Oppression: {liberty_oppression}
"""

def classifyIdeologyFromMF(
    care_harm,
    fairness_cheating,
    loyalty_betrayal,
    authority_subversion,
    sanctity_degradation,
    liberty_oppression,
    metadata=None,
    model="zephyr:latest",
):
    if metadata is None:
        metadata = {}

    SYS_PROMPT = "Return ONLY valid JSON. No markdown. No extra text."

    USER_PROMPT = IDEOLOGY_FROM_MF_PROMPT_EN.format(
        care_harm=care_harm,
        fairness_cheating=fairness_cheating,
        loyalty_betrayal=loyalty_betrayal,
        authority_subversion=authority_subversion,
        sanctity_degradation=sanctity_degradation,
        liberty_oppression=liberty_oppression,
    )

    response, _ = client.generate(
        model_name=model,
        system=SYS_PROMPT,
        prompt=USER_PROMPT
    )

    # 1) spróbuj sparsować całość
    obj = None
    try:
        obj = json.loads(response)
    except Exception:
        pass

    # 2) jeśli nie, wyciągnij pierwszy obiekt JSON
    if obj is None:
        js = _extract_first_json_object(response)
        if js:
            try:
                obj = json.loads(js)
            except Exception:
                obj = None

    if not isinstance(obj, dict):
        return None

    # normalizacja predicted_ideology
    pred = str(obj.get("predicted_ideology", "")).strip().lower()
    if pred not in ("liberal", "conservative"):
        return None

    # normalizacja confidence (0..10 int)
    conf = obj.get("confidence", 0)
    if isinstance(conf, bool):
        conf = int(conf)
    if isinstance(conf, (int, float)):
        conf = int(conf)
    else:
        conf = 0
    if conf < 0:
        conf = 0
    if conf > 10:
        conf = 10

    cleaned = {
        "predicted_ideology": pred,
        "confidence": conf
    }

    return dict(cleaned, **metadata)





