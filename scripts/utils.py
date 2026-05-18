import re
from typing import Dict, List, Optional, Tuple

# Optional deps (only required for de-identification utilities).
# Keep these optional so the rest of the repo can run without spaCy installed.
try:
    import spacy  # type: ignore
except Exception:  # pragma: no cover
    spacy = None

try:
    from pydeidentify import Deidentifier  # type: ignore
except Exception:  # pragma: no cover
    Deidentifier = None

def create_classification_prompt(comment: str, content_type: str = "reddit", few_shot_text: str = "none") -> str:
    """Creates the classification prompt for analyzing content about homelessness, with optional few-shot examples.
    
    Args:
        comment: The text content to analyze
        content_type: Type of content source ("reddit", "x", "news", "meeting_minutes")
        few_shot_text: Few-shot examples (default "none" for zero-shot, or content type name for automatic selection)
    """
    # Validate content_type
    valid_content_types = ["reddit", "x", "news", "meeting_minutes"]
    if content_type.lower() not in valid_content_types:
        raise ValueError(f"content_type must be one of {valid_content_types}, got '{content_type}'")
    
    # Define content type descriptions
    content_descriptions = {
        "reddit": "Reddit comments",
        "x": "X (Twitter) posts",
        "news": "news articles",
        "meeting_minutes": "meeting minutes"
    }
    
    # Define few-shot text mapping
    few_shot_mapping = {
        "reddit": FEW_SHOT_REDDIT_PROMPT_TEXT,
        "x": FEW_SHOT_X_PROMPT_TEXT,
        "news": FEW_SHOT_NEWS_PROMPT_TEXT,
        "meeting_minutes": FEW_SHOT_MEETING_MINUTES_PROMPT_TEXT
    }
    
    content_desc = content_descriptions.get(content_type.lower(), "content")
    
    # Handle few-shot text selection
    if few_shot_text.lower() == "none":
        few_shot_text = None  # Zero-shot
    elif few_shot_text in valid_content_types:
        # Use automatic selection for the specified content type
        few_shot_text = few_shot_mapping.get(few_shot_text.lower(), FEW_SHOT_REDDIT_PROMPT_TEXT)
    
    base_prompt = f"""You are an expert in social behavior analysis. Your task is to analyze {content_desc} about homelessness and categorize them according to specific criteria.

DEFINITIONS:

1. Comment Types (select all that apply):
   - Ask a Genuine Question: The speaker asks a sincere question about homelessness or related issues
   - Ask a Rhetorical Question: The speaker asks a question not intended to be answered, often to make a point
   - Provide a Fact or Claim: The speaker provides a factual statement or claim about homelessness
   - Provide an Observation: The speaker shares an observation about homelessness or related situations
   - Express their Opinion: The speaker expresses their own views or feelings about homelessness
   - Express Others Opinions: The speaker describes or references the views or feelings of others about homelessness

2. Critique Categories (select all that apply):
   - Money Aid Allocation: Discussion of financial resources, aid distribution, or resource allocation for homelessness
   - Government Critique: Criticism of government policies, laws, or political approaches to homelessness
   - Societal Critique: Criticism of social norms, systems, or societal attitudes toward homelessness

3. Response Categories (select all that apply):
   - Solutions/Interventions: Discussion of specific solutions, interventions, or charitable actions

4. Perception Types (select all that apply):
   - Personal Interaction: Direct personal experiences with PEH
   - Media Portrayal: Discussion of PEH as portrayed in media
   - Not in my Backyard: Opposition to local homelessness developments
   - Harmful Generalization: Negative stereotypes about PEH
   - Deserving/Undeserving: Judgments about who deserves help

5. racist Classification:
   - Yes: Contains explicit or implicit racial bias
   - No: No racial bias present

INSTRUCTIONS:
1. Read the comment carefully
2. Analyze it according to the categories above
3. Provide your analysis in the exact format below
4. Include a brief reasoning for your classification

FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:

Comment Type: [ask a genuine question, ask a rhetorical question, provide a fact or claim, provide an observation, express their opinion, express others opinions]
Critique Category: [money aid allocation, government critique, societal critique]
Response Category: [solutions/interventions]
Perception Type: [personal interaction, media portrayal, not in my backyard, harmful generalization, deserving/undeserving]
racist: [Yes/No]
Reasoning: [brief explanation]
"""
    if few_shot_text:
        base_prompt += f"\n\n{few_shot_text.strip()}\n"
    base_prompt += f"\nContent to analyze:\n\"\"\" {comment} \"\"\"\n\nAnalysis:"
    return base_prompt

def create_mitigation_prompt(comment: str, content_type: str = "reddit", classification: str = "") -> str:
    """Creates the mitigation prompt for rephrasing biased content.
    
    Args:
        comment: The text content to mitigate
        content_type: Type of content source ("reddit", "x", "news", "meeting_minutes")
        classification: Optional classification information
    """
    # Validate content_type
    valid_content_types = ["reddit", "x", "news", "meeting_minutes"]
    if content_type.lower() not in valid_content_types:
        raise ValueError(f"content_type must be one of {valid_content_types}, got '{content_type}'")
    
    # Define content type descriptions
    content_descriptions = {
        "reddit": "Reddit comment",
        "x": "X (Twitter) post",
        "news": "news article",
        "meeting_minutes": "meeting minutes"
    }
    
    content_desc = content_descriptions.get(content_type.lower(), "content")
    
    return f"""You are a social behavior expert. Rephrase the following {content_desc} to remove any bias against people experiencing homelessness. Explain your reasoning for the changes you made.

Format your response exactly as follows:

NEW CONTENT: [your rephrased content]
REASONING: [your explanation]

Content:
\"\"\"{comment}\"\"\"

Analysis:"""

def create_mitigation_y_n_classification_prompt(comment: str, content_type: str = "reddit") -> str:
    """Creates a simple yes/no prompt to check if content is biased.
    
    Args:
        comment: The text content to analyze
        content_type: Type of content source ("reddit", "x", "news", "meeting_minutes")
    """
    # Validate content_type
    valid_content_types = ["reddit", "x", "news", "meeting_minutes"]
    if content_type.lower() not in valid_content_types:
        raise ValueError(f"content_type must be one of {valid_content_types}, got '{content_type}'")
    
    # Define content type descriptions
    content_descriptions = {
        "reddit": "Reddit comment",
        "x": "X (Twitter) post",
        "news": "news article",
        "meeting_minutes": "meeting minutes"
    }
    
    content_desc = content_descriptions.get(content_type.lower(), "content")
    
    return f"""You are a social behavior expert. Analyze the following {content_desc} for bias against people experiencing homelessness (PEH).

Content:
\"\"\"{comment}\"\"\"

Is this content biased against PEH? Reply with Yes or No only.

Analysis:"""

def create_recheck_prompt(comment: str, content_type: str = "reddit") -> str:
    """Creates a prompt to check if mitigated content is still biased.
    
    Args:
        comment: The text content to recheck
        content_type: Type of content source ("reddit", "x", "news", "meeting_minutes")
    """
    # Validate content_type
    valid_content_types = ["reddit", "x", "news", "meeting_minutes"]
    if content_type.lower() not in valid_content_types:
        raise ValueError(f"content_type must be one of {valid_content_types}, got '{content_type}'")
    
    # Define content type descriptions
    content_descriptions = {
        "reddit": "Reddit comment",
        "x": "X (Twitter) post",
        "news": "news article",
        "meeting_minutes": "meeting minutes"
    }
    
    content_desc = content_descriptions.get(content_type.lower(), "content")
    
    return f"""You are a social behavior expert. Analyze the following revised {content_desc} for any remaining bias against people experiencing homelessness.

Content:
\"\"\"{comment}\"\"\"

Is this content still biased against PEH? Reply with Yes or No only.

Analysis:"""

def clean_text(text: str) -> str:
    """Cleans text by removing special characters and normalizing whitespace."""
    if not isinstance(text, str):
        return ""
    # Keep only alphanumeric and spaces, convert to lowercase
    text = re.sub(r'[^a-z0-9\s]', '', text.lower())
    return ' '.join(text.split())  # Normalize whitespace

def extract_field(text: str, field_name: str) -> str:
    """Extracts a specific field from the model's response text."""
    if not isinstance(text, str):
        return ""
    
    # Find the field in the text
    pattern = rf"{field_name}:\s*(.*?)(?:\n(?!\w+:)|$)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    
    return match.group(1).strip()

def extract_mitigation_results(mitigation_output: str) -> Tuple[str, str]:
    """Extracts the new content and reasoning from the mitigation output."""
    # Find the last NEW CONTENT: in the text
    last_new_content = mitigation_output.rfind("NEW CONTENT:")
    if last_new_content == -1:
        # Fallback to NEW COMMENT: for backward compatibility
        last_new_content = mitigation_output.rfind("NEW COMMENT:")
        if last_new_content == -1:
            return mitigation_output, "Format error in model response"
        content_label = "NEW COMMENT:"
    else:
        content_label = "NEW CONTENT:"
    
    # Get everything after the last NEW CONTENT: or NEW COMMENT:
    text_after_new_content = mitigation_output[last_new_content + len(content_label):]
    
    # Split on REASONING: if it exists
    if "REASONING:" in text_after_new_content:
        new_content, reasoning = text_after_new_content.split("REASONING:", 1)
        return new_content.strip(), reasoning.strip()
    else:
        return text_after_new_content.strip(), "No reasoning provided"

def extract_classification_results(classification_output: str) -> str:
    """Extracts the classification analysis from the model's response."""
    analysis_start = classification_output.find("Analysis:")
    if analysis_start != -1:
        return classification_output[analysis_start + len("Analysis:"):].strip()
    return classification_output

def get_model_config(model_name: str) -> Dict:
    """Returns model-specific configuration parameters."""
    configs = {
        "qwen": {
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "max_new_tokens": 500,
            "temperature": 0.1,
            "top_p": 0.95,
            "repetition_penalty": 1.1
        },
        "llama": {
            "model_id": "meta-llama/Llama-3.2-3B-Instruct",
            "max_new_tokens": 500,
            "temperature": 0.1,
            "top_p": 0.95,
            "repetition_penalty": 1.1
        },
        "gemma3": {
            "model_id": "google/gemma-3-4b-it",
            "max_new_tokens": 500,
            "temperature": 0.1,
            "top_p": 0.95,
            "repetition_penalty": 1.1
        },
        "phi4": {
            "model_id": "microsoft/Phi-4-mini-instruct",
            "max_new_tokens": 500,
            "temperature": 0.1,
            "top_p": 0.95,
            "repetition_penalty": 1.1
        },
        "gpt4": {
            "api": "openai",
            "api_key_env": "OPENAI_API_KEY",
            "model_id": "gpt-4.1",
            "max_new_tokens": 500
        },
        "gemini": {
            "api": "google",
            "api_key_env": "GEMINI_API_KEY",
            "model_id": "gemini-2.5-pro",
            "max_new_tokens": 500
        },
        "grok": {
            "api": "grok",
            "api_key_env": "GROK_API_KEY",
            "model_id": "grok-4-latest",
            "max_new_tokens": 500
        }
    }
    return configs.get(model_name.lower(), configs["qwen"])

import requests
import os
import time
import random

def handle_rate_limit(response, model_name):
    """Handle rate limit responses with exponential backoff."""
    if response.status_code == 429:  # Rate limit
        retry_after = response.headers.get('Retry-After')
        if retry_after:
            wait_time = int(retry_after)
        else:
            wait_time = 60  # Default 1 minute
        
        print(f"Rate limit hit for {model_name}. Waiting {wait_time} seconds...")
        time.sleep(wait_time)
        return True  # Should retry
    return False  # Don't retry

def call_api_llm(prompt: str, model_name: str, max_tokens: int = 500) -> str:
    """Call an API-based LLM (GPT-4.1, Gemini 2.5 Pro, Grok 4) and return the response text."""
    # Load .env if present and not already loaded
    if not getattr(call_api_llm, '_env_loaded', False):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            print("[Warning] python-dotenv not installed. Set API keys in your environment or install with 'pip install python-dotenv'.")
        call_api_llm._env_loaded = True
    config = get_model_config(model_name)
    api = config.get("api")
    api_key = os.environ.get(config.get("api_key_env", ""), None)
    model_id = config.get("model_id")
    if api == "openai":
        # Use new OpenAI SDK v1.x for gpt-4.1 and similar models
        max_retries = 5  # Increased retries for better rate limit handling
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                # Use chat.completions.create for chat models
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.1,
                )
                response_content = response.choices[0].message.content
                
                # Check if response contains rate limit error
                if "rate limit" in response_content.lower() or "tpm" in response_content.lower():
                    if attempt < max_retries - 1:
                        # Extract wait time from response if available
                        wait_time = 5.0  # Default wait time
                        if "try again in" in response_content.lower():
                            try:
                                import re
                                wait_match = re.search(r'try again in (\d+\.?\d*)s', response_content.lower())
                                if wait_match:
                                    wait_time = float(wait_match.group(1)) + 1.0  # Add buffer
                            except:
                                pass
                        
                        delay = max(wait_time, base_delay * (2 ** attempt) + random.uniform(0, 1))
                        print(f"Rate limit error in response for {model_name}. Waiting {delay:.1f} seconds and retrying... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        # If we've exhausted retries, raise an exception so it gets caught
                        raise Exception(response_content)
                
                return response_content
            except ImportError:
                # Fallback to old SDK if new one is not available
                import openai
                openai.api_key = api_key
                response = openai.ChatCompletion.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.1,
                )
                response_content = response["choices"][0]["message"]["content"]
                
                # Check if response contains rate limit error
                if "rate limit" in response_content.lower() or "tpm" in response_content.lower():
                    if attempt < max_retries - 1:
                        # Extract wait time from response if available
                        wait_time = 5.0  # Default wait time
                        if "try again in" in response_content.lower():
                            try:
                                import re
                                wait_match = re.search(r'try again in (\d+\.?\d*)s', response_content.lower())
                                if wait_match:
                                    wait_time = float(wait_match.group(1)) + 1.0  # Add buffer
                            except:
                                pass
                        
                        delay = max(wait_time, base_delay * (2 ** attempt) + random.uniform(0, 1))
                        print(f"Rate limit error in response for {model_name}. Waiting {delay:.1f} seconds and retrying... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        # If we've exhausted retries, raise an exception so it gets caught
                        raise Exception(response_content)
                
                return response_content
            except Exception as e:
                error_str = str(e).lower()
                if ("rate_limit" in error_str or "429" in error_str or "tpm" in error_str or "tokens per min" in error_str):
                    if attempt < max_retries - 1:
                        # Extract wait time from error message if available
                        wait_time = 5.0  # Default wait time
                        if "try again in" in error_str:
                            try:
                                import re
                                wait_match = re.search(r'try again in (\d+\.?\d*)s', error_str)
                                if wait_match:
                                    wait_time = float(wait_match.group(1)) + 1.0  # Add buffer
                            except:
                                pass
                        
                        delay = max(wait_time, base_delay * (2 ** attempt) + random.uniform(0, 1))
                        print(f"Rate limit hit for {model_name} (TPM limit). Waiting {delay:.1f} seconds... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                raise e
    elif api == "google":
        # Gemini 2.5 Pro (Google Generative AI SDK)
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_id)
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        print(f"Rate limit hit for {model_name}. Retrying in {delay:.1f} seconds... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                raise e
    elif api == "grok":
        # Grok 4 (xAI API)
        endpoint = "https://api.x.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {
            "model": model_id, 
            "messages": [{"role": "user", "content": prompt}], 
            "max_tokens": max_tokens,
            "stream": False,
            "temperature": 0.1
        }
        
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            resp = requests.post(endpoint, headers=headers, json=data)
            if resp.status_code == 429:  # Rate limit
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Rate limit hit for {model_name}. Retrying in {delay:.1f} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
    else:
        raise ValueError(f"Unknown or unsupported API model: {model_name}")

# Define categories
COMMENT_TYPES = [
    "ask a genuine question",
    "ask a rhetorical question",
    "provide a fact or claim",
    "provide an observation",
    "express their opinion",
    "express others opinions"
]
CRITIQUE_CATEGORIES = ["money aid allocation", "government critique", "societal critique"]
RESPONSE_CATEGORIES = ["solutions/interventions"]
PERCEPTION_TYPES = ["personal interaction", "media portrayal", "not in my backyard", "harmful generalization", "deserving/undeserving"]

def extract_flags(field_text: str, options: List[str]) -> Dict[str, int]:
    """Extracts flags from field text with strict matching."""
    flags = {opt: 0 for opt in options}

    if not field_text or not isinstance(field_text, str):
        return flags

    field_text = field_text.lower()
    if field_text.strip() in ["none", "n/a", "-", "no categories", "none applicable"]:
        return flags

    # Split on common delimiters and clean each item
    field_items = []
    for item in re.split(r'[,\n•\-]+', field_text):
        cleaned = clean_text(item)
        if cleaned:
            field_items.append(cleaned)

    # For each option, check for exact match only
    for opt in options:
        cleaned_opt = clean_text(opt)
        # Only match if the entire option is present as a complete word
        if cleaned_opt in field_items:
            flags[opt] = 1
    return flags

def create_output_row(
    comment: str,
    city: str,
    comment_text: str,
    critique_text: str,
    response_text: str,
    perception_text: str,
    racist_flag: int,
    reasoning: str,
    raw_response: str
) -> Dict:
    """Creates a standardized output row with all fields and flags."""
    output_row = {
        "Comment": comment,
        "City": city,
        "Comment Type": comment_text,
        "Critique Category": critique_text,
        "Response Category": response_text,
        "Perception Type": perception_text,
        "racist": "Yes" if racist_flag else "No",
        "Reasoning": reasoning,
        "Raw Response": raw_response
    }
    
    # Add all flag columns
    for category, flags in [
        ("Comment", extract_flags(comment_text, COMMENT_TYPES)),
        ("Critique", extract_flags(critique_text, CRITIQUE_CATEGORIES)),
        ("Response", extract_flags(response_text, RESPONSE_CATEGORIES)),
        ("Perception", extract_flags(perception_text, PERCEPTION_TYPES))
    ]:
        for flag, value in flags.items():
            output_row[f"{category}_{flag}"] = value
    
    # Add racist flag
    output_row["Racist_Flag"] = racist_flag
    
    return output_row

def load_spacy_model():
    try:
        # Load English language model
        if spacy is None:
            raise ImportError("spaCy is not installed")
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        print("Downloading spaCy model...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

def deidentify_text(text, nlp=None):
    if not isinstance(text, str):
        return ""
    # First, use pydeidentify
    if Deidentifier is None:
        raise ImportError("pydeidentify is not installed")
    deidentifier = Deidentifier()
    text = str(deidentifier.deidentify(text))  # Convert DeidentifiedText to string
    
    # Then, apply custom regex/spaCy logic for further deidentification
    if nlp is None:
        if spacy is None:
            raise ImportError("spaCy is not installed")
        nlp = spacy.load("en_core_web_sm")
    
    # Process text with spaCy
    doc = nlp(text)
    deidentified = text
    
    # Custom patterns for domain-specific terms
    location_patterns = [
        (r'\b(?:St\.|Saint)\s+[A-Za-z]+\s+(?:County|Parish|City|Town)\b', '[LOCATION]'),
        (r'\b(?:Low|High)\s+Barrier\s+(?:Homeless|Housing)\s+Shelter\b', '[INSTITUTION]'),
        (r'\b(?:Homeless|Housing)\s+Shelter\b', '[INSTITUTION]'),
        (r'\b(?:Community|Resource)\s+Center\b', '[INSTITUTION]'),
        (r'\b(?:Public|Private)\s+(?:School|University|College)\b', '[INSTITUTION]'),
        (r'\b(?:Medical|Health)\s+Center\b', '[INSTITUTION]'),
        (r'\b(?:Police|Fire)\s+Department\b', '[INSTITUTION]'),
        (r'\b(?:City|County|State)\s+Hall\b', '[INSTITUTION]'),
        (r'\b(?:Public|Private)\s+(?:Library|Park|Garden)\b', '[INSTITUTION]'),
        (r'\b(?:Shopping|Retail)\s+Mall\b', '[INSTITUTION]'),
        (r'\b(?:Bus|Train|Subway)\s+Station\b', '[INSTITUTION]'),
        (r'\b(?:Airport|Harbor|Port)\b', '[INSTITUTION]'),
        (r'\b(?:Street|Avenue|Road|Boulevard|Drive|Lane|Place|Court|Circle|Way)\b', '[STREET]'),
        (r'\b(?:North|South|East|West|N|S|E|W)\s+(?:Street|Avenue|Road|Boulevard|Drive|Lane|Place|Court|Circle|Way)\b', '[STREET]'),
        (r'\b(?:First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth)\s+(?:Street|Avenue|Road|Boulevard|Drive|Lane|Place|Court|Circle|Way)\b', '[STREET]'),
        (r'\b(?:Main|Broad|Market|Park|Church|School|College|University|Hospital|Library)\s+(?:Street|Avenue|Road|Boulevard|Drive|Lane|Place|Court|Circle|Way)\b', '[STREET]'),
    ]
    
    # Apply location patterns first
    for pattern, replacement in location_patterns:
        deidentified = re.sub(pattern, replacement, deidentified, flags=re.IGNORECASE)
    
    # Replace named entities
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'GPE', 'LOC', 'ORG', 'DATE', 'TIME']:
            if ent.label_ == 'PERSON':
                replacement = '[PERSON]'
            elif ent.label_ in ['GPE', 'LOC']:
                replacement = '[LOCATION]'
            elif ent.label_ == 'ORG':
                replacement = '[ORGANIZATION]'
            elif ent.label_ == 'DATE':
                replacement = '[DATE]'
            elif ent.label_ == 'TIME':
                replacement = '[TIME]'
            deidentified = deidentified.replace(ent.text, replacement)
    
    # Additional patterns for emails, phones, etc.
    patterns = {
        r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}': '[PHONE]',
        r'\+\d{1,2}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}': '[PHONE]',
        r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}': '[PHONE]',
        r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}': '[PHONE]',
        r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[^\s]*)?': '[URL]',
        r'www\.(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[^\s]*)?': '[URL]',
        r'(?:[-\w.]|(?:%[\da-fA-F]{2}))+\.(?:com|org|net|edu|gov|mil|biz|info|mobi|name|aero|asia|jobs|museum)(?:/[^\s]*)?': '[URL]',
        r'\[URL\](?:/[^\s]*)?': '[URL]',
        r'\[URL\]/search\?[^\s]*': '[URL]',
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b': '[EMAIL]',
        r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b': '[IP]',
        r'\b\d{5}(?:-\d{4})?\b': '[ZIP]',
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b': '[DATE]',
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b': '[DATE]',
    }
    
    # Apply additional patterns
    for pattern, replacement in patterns.items():
        deidentified = re.sub(pattern, replacement, deidentified)
    
    # Clean up any remaining URL-like or location patterns
    deidentified = re.sub(r'\[URL\]/[^\s]+', '[URL]', deidentified)
    deidentified = re.sub(r'\[URL\]\[URL\]', '[URL]', deidentified)
    deidentified = re.sub(r'\[LOCATION\]/[^\s]+', '[LOCATION]', deidentified)
    deidentified = re.sub(r'\[LOCATION\]\[LOCATION\]', '[LOCATION]', deidentified)
    deidentified = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', deidentified)
    
    return deidentified

FEW_SHOT_REDDIT_PROMPT_TEXT = '''Sentance: Are you implying that local police beat panhandlers with batons?  Because they don't .
Comment Type: [ask a genuine question, provide a fact or claim]
Critique Category: [societal critique]
Response Category: []
Perception Type: []
racist: [No]

Sentance: Most comments are saying how great it is to homeless (and it usually is) but are ignoring or unaware of the ***type*** of homeless they plan to [STREET] here.  *Drug addicts and people with mental issues.*  If it were more homes for homeless and/or low income families, I wouldn't think twice about it but I'm very concerned about a facility housing drug addicts and people with mental issues just a couple hundred feet from a school in the middle of a residential neighborhood.
Comment Type: [express their opinion, express others opinions]
Critique Category: []
Response Category: []
Perception Type: [not in my backyard, harmful generalization, deserving/undeserving]
racist: [No]

Sentance: What is up with the pots and pans?  What homeless or trafficked person needs those?  Oh wait!  She needs some.  Send her 50 sets.  She can keep one and sell the rest!  What a piece of 💩
Comment Type: [ask a rhetorical question, express their opinion]
Critique Category: [societal critique]
Response Category: []
Perception Type: [not in my backyard, harmful generalization]
racist: [No]

Sentance: "I live here too [ORGANIZATION][ORGANIZATION][ORGANIZATION]
Fuck the homeless"
Comment Type: [express their opinion]
Critique Category: []
Response Category: []
Perception Type: [not in my backyard, harmful generalization, deserving/undeserving]
racist: [No]

Sentance: "I won't support organizations that are homophobic personally. I clearly stated that others can make their own choices. I then brought up a very real issue in [ORGANIZATION]. I'm a Social Worker. I've worked directly with [ORGANIZATION] in the past. They are very religious. It is what it is. You overreacted to my post IMO. I'm not that important. Its just my opinion. But yeah, I'm not okay with discrimination, so personally I would not work for nor support [PERSON]. I know far too many GLBTQIA+ and Trans individuals that have struggled in [ORGANIZATION] because of discrimination from places like this. Trans houseless individuals in particular are often sexually assaulted around here when they start engaging in services. Its a problem. >""Get over it"" **No.**"
Comment Type: [provide a fact or claim, provide an observation, express their opinion]
Critique Category: [societal critique]
Response Category: []
Perception Type: []
racist: [No]
'''

FEW_SHOT_X_PROMPT_TEXT = '''Post: [PERSON] awarded $100,000 to [PERSON] (ORG3) to enhance employment and education-related skills for [DATE] and migrant farmworkers. The award was part of a $300,000 discretionary fund award under the CSBG Program. [PERSON]
Comment Type: [provide a fact or claim]
Critique Category: [money aid allocation]
Response Category: [solutions/interventions]
Perception Type: []
racist: [No]

Post: "Did your Black flunky mayor get the🐀[ORGANIZATION]'s memo 2 stick it 2 Rump instead of serving you by refusing 2 deport migrants + give them Black taxpayers'💰 4 shelter+food while Black citizens go homeless? [ORGANIZATION] mayors did. Charity starts at 🏠. [URL]"
Comment Type: [ask a rhetorical question, provide a fact or claim, express their opinion]
Critique Category: [money aid allocation, government critique]
Response Category: []
Perception Type: [harmful generalization, deserving/undeserving]
racist: [Yes]

Post: PERSON0 Instead of peacocking on social media for your next job, how about you concentrate on the gaggles of homeless people in [ORGANIZATION]?
Comment Type: [ask a rhetorical question, provide a fact or claim, express their opinion]
Critique Category: [societal critique]
Response Category: [solutions/interventions]
Perception Type: []
racist: [No]

Post: [ORGANIZATION] Just what [ORGANIZATION] needs...another beggar.
Comment Type: [express their opinion]
Critique Category: []
Response Category: []
Perception Type: [not in my backyard, harmful generalization, deserving/undeserving]
racist: [No]

Post: [ORGANIZATION] area in [ORGANIZATION] is facing a housing crisis. 40% of people in this area live in poverty, and the city lacks 20,000 affordable housing units. Initiatives like [ORGANIZATION] to fix old housing, but progress depends on securing funding. [URL]
Comment Type: [provide a fact or claim]
Critique Category: [money aid allocation]
Response Category: [solutions/interventions]
Perception Type: []
racist: [No]
'''

FEW_SHOT_NEWS_PROMPT_TEXT = '''Article: We applaud this important first step to assure the long-term resolution of homelessness.
Comment Type: [express their opinion]
Critique Category: []
Response Category: [solutions/interventions]
Perception Type: []
racist: [No]

Article: 60 million for programs to support homeless veterans including 20 million for [ORGANIZATION]. The President proposed to eliminate the program.
Comment Type: [provide a fact or claim]
Critique Category: []
Response Category: [solutions/interventions]
Perception Type: []
racist: [No]

Article: [ORGANIZATION] county commissioners on [ORGANIZATION] weighed options for creating a migrant support services center while city emergency managers opened a busing hub, as dozens of migrants remained in homeless conditions [LOCATION].
Comment Type: [provide a fact or claim]
Critique Category: []
Response Category: [solutions/interventions]
Perception Type: []
racist: [No]

Article: About 1 in 3 people who are homeless in [ORGANIZATION] report having a mental illness or a substance use disorder, and the combination of homelessness and substance use or untreated mental illness has led to very public tragedies.
Comment Type: [provide a fact or claim, express their opinion]
Critique Category: []
Response Category: []
Perception Type: []
racist: [No]

Article: I would imagine she is not being delusional about being unsafe on the streets, [ORGANIZATION], executive director of [ORGANIZATION], told [ORGANIZATION]. [PERSON] specializes in treating mentally ill homeless people. Somewhere in all of this is a hook around the fear she has of being unsafe, especially as a woman who is homeless, and that is not uncommon. There should be a real conversation about that, and it could be very useful for figuring out whats going on with her.
Comment Type: [provide a fact or claim, provide an observation, express their opinion]
Critique Category: []
Response Category: [solutions/interventions]
Perception Type: []
racist: [No]
'''

FEW_SHOT_MEETING_MINUTES_PROMPT_TEXT = '''Meeting Minutes: Okay. But on that area, one corner is the location that is on dispute right now. And the reason that we are, that I am questioning that is because we got a lot of homeless people there.
Comment Type: [provide a fact or claim, express their opinion]
Critique Category: []
Response Category: []
Perception Type: [not in my backyard]
racist: [No]

Meeting Minutes: but they stuck with us, they got all the permissions they needed, and we would not have made the functional end of veteran homelessness in [ORGANIZATION] without them, so thank you. PERSON0? Well, thank you for this honor.
Comment Type: [provide a fact or claim, express their opinion]
Critique Category: []
Response Category: [solutions/interventions]
Perception Type: []
racist: [No]

Meeting Minutes: I've seen it all, like certain people being removed out of there. And I'm down there [ORGANIZATION]. And all the [ORGANIZATION] and all the residents there, like, I know them all, you know, and I love them because I was dropped off to be homeless [DATE].
Comment Type: [provide a fact or claim, provide an observation, express their opinion]
Critique Category: []
Response Category: []
Perception Type: []
racist: [No]

Meeting Minutes: Yeah. Yeah, I just think that there's a different ROI for chronically homeless folks. And there's a deeper impact, but a more narrow impact.
Comment Type: [provide a fact or claim, express their opinion]
Critique Category: [money aid allocation]
Response Category: [solutions/interventions]
Perception Type: [deserving/undeserving]
racist: [No]

Meeting Minutes: Simple delays, be they [ORGANIZATION] or [LOCATION], have been a feeble solution. Demolitions of up to 370 affordable houses a year valued at $100 million dwarfs the city's efforts at spending $20 million to support affordable housing. Can we really achieve affordable housing through demolition?
Comment Type: [ask a rhetorical question, provide a fact or claim, express their opinion]
Critique Category: [money aid allocation, government critique]
Response Category: [solutions/interventions]
Perception Type: []
racist: [No]'''

CITY_MAP = {
    'south bend': 'southbend',
    'rockford': 'rockford',
    'kalamazoo': 'kzoo',
    'scranton': 'scranton',
    'fayetteville': 'fayetteville',
    'san francisco': 'sanfrancisco',
    'portland': 'portland',
    'buffalo': 'buffalo',
    'baltimore': 'baltimore',
    'el paso': 'elpaso',
}

