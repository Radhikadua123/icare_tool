# src/mcq_generation_ollama.py
import requests
from requests.exceptions import Timeout
import json
import pandas as pd
import os
from datetime import datetime
import re
import argparse
import random
import copy
import requests
import json
import os
from typing import Dict, Any, Optional, List
import numpy as np

def ensure_dir(directory: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)
    
def make_ollama_request(
    prompt: str,
    url: str = "http://localhost:11434/api/generate",
    model: str = "llama3.2:1b",
    max_tokens: int = 2048,
    temperature: float = 0.7,
    timeout: int = 300,
    seed: int = 123,
    top_p: float = 0.9,
    stream: bool = False
) -> Optional[Dict[str, Any]]:
    """Make a request to the Ollama API."""
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "num_predict": max_tokens
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=timeout)
        print(response.json())
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error in Ollama API call: {e}")
        return None

def parse_mcq(mcq_text):
    """Parse MCQ text into structured format."""
    questions = []
    
    # Split the text into individual questions
    raw_questions = mcq_text.split('\n\n')
    
    for i, raw_q in enumerate(raw_questions, start=0):
        if not raw_q.strip():
            continue
        
        try:
            # Initialize question dictionary
            question = {
                "question_id": i,
                "question_text": None,
                "options": {
                    "A": None, "B": None, "C": None, "D": None
                },
                "correct_answer": None
            }
            
            # Extract question number and text
            q_match = re.search(r'\*\*(\d+):\s+(.+?)\*\*', raw_q)
            if q_match:
                question["question_text"] = q_match.group(2).strip()
            
            # Extract options
            options = re.findall(r'([A-D])\)\s+(.+?)(?=\n[A-D]\)|$|\nAnswer:)', raw_q, re.DOTALL)
            for opt_letter, opt_text in options:
                question["options"][opt_letter] = opt_text.strip()
            
            # Extract answer
            answer_match = re.search(r'Answer:\s+([A-D])', raw_q)
            if answer_match:
                question["correct_answer"] = answer_match.group(1)
            
            # Only add complete questions
            if (question["question_text"] and 
                all(question["options"].values()) and 
                question["correct_answer"]):
                questions.append(question)
        except Exception as e:
            print(f"Error parsing question: {e}")
            continue
    
    return questions

def generate_and_write_mcqs(reports, num_ques, output_file, url="http://localhost:11434/api/generate",
    model="llama3.2:1b", timeout=300, max_tokens=2048, 
    temperature=0.7, top_p=0.9, seed=123):
    """Generate MCQs for given reports and write them to file."""
    try:
        # Set base random seed
        total_mcqs = 0
        total_reports_processed = 0

        with open(output_file, 'w') as file:
            # Write the opening of the JSON structure
            file.write('{\n"metadata": {\n')
            file.write(f'"generation_timestamp": "{datetime.now().isoformat()}",\n')
            file.write(f'"total_reports": {len(reports)},\n')
            file.write('"total_mcqs": 0\n},\n')
            file.write('"mcq_data": [\n')

            for i, report in enumerate(reports):
                # Generate MCQs until we have num_ques in the desired format
                formatted_mcqs = []
                while len(formatted_mcqs) < num_ques:
                    prompt = (
                        f"Please generate {num_ques} different multiple choice question answer pairs for the following radiology report: {report}. "
                        "The questions should be based on report and cannot be answered without the report."
                        "Please use the following format exactly as your life depends on sticking to these formats.:\n\n"
                        "**1: [Question text]**\n"
                        "A) [Option A]\n"
                        "B) [Option B]\n"
                        "C) [Option C]\n"
                        "D) [Option D]\n"
                        "Answer: [Correct answer]\n\n"
                    )
                    
                    try:
                        response = make_ollama_request(
                            prompt=prompt,
                            url=url,
                            model=model,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            timeout=timeout,
                            seed=seed,
                            top_p=top_p,
                            stream=False
                        )
                        
                        if response and "response" in response:
                            mcq_data = response["response"]
                            parsed_mcqs = parse_mcq(mcq_data)
                            
                            # Only add MCQs that are in the desired format
                            for mcq in parsed_mcqs:
                                if (mcq["question_text"] and 
                                    all(mcq["options"].values()) and 
                                    mcq["correct_answer"]):
                                    formatted_mcqs.append(mcq)
                                    
                                if len(formatted_mcqs) >= num_ques:
                                    print(f"Generated {len(formatted_mcqs)} MCQs for this report")
                                    break

                    except Timeout:
                        print(f"Request timed out for report: {report[:50]}...")
                        continue
                    except Exception as e:
                        print(f"Error processing report: {e}")
                        continue
                
                # Check if we have num_ques valid MCQs
                if len(formatted_mcqs) == num_ques:
                    report_data = {
                        "report": report,
                        "questions": formatted_mcqs[:num_ques]  # Ensure we only take num_ques questions
                    }
                    # Write the report data to file
                    json.dump(report_data, file)
                    file.write(',\n' if i < len(reports) - 1 else '\n')
                    file.flush()
                    total_mcqs += num_ques
                    total_reports_processed += 1

            # Write the closing of the JSON structure
            file.write(']\n}')

        # Update metadata
        with open(output_file, 'r+') as file:
            content = file.read()
            file.seek(0)
            content = content.replace('"total_mcqs": 0', f'"total_mcqs": {total_mcqs}')
            content = content.replace(
                f'"total_reports": {len(reports)}', 
                f'"total_reports": {total_reports_processed}'
            )
            file.write(content)
            file.truncate()

        return total_reports_processed, total_mcqs
    except Exception as e:
        print(f"Error in generate_and_write_mcqs: {e}")
        return 0, 0

def swap_answer_choices(question, rng):
    """Swap the correct answer with another choice in a question."""
    # Create a deep copy of the question to avoid modifying the original
    modified_question = copy.deepcopy(question)

    # Ensure the question has the required keys
    if 'options' not in modified_question or 'correct_answer' not in modified_question:
        return modified_question

    current_correct = modified_question['correct_answer']
    other_options = [opt for opt in modified_question['options'].keys() if opt != current_correct]
    
    if other_options:
        swap_option = rng.choice(other_options)
        modified_question['options'][current_correct], modified_question['options'][swap_option] = \
            modified_question['options'][swap_option], modified_question['options'][current_correct]
        modified_question['correct_answer'] = swap_option

    return modified_question

def process_json_file(input_file, output_file, seed=123):
    """Read JSON file, swap answer choices, and save to new file."""
    # Create RNG once at the start
    rng = random.Random(seed)
    
    # Read the original JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Track the number of changes
    total_questions = 0
    modified_questions = 0

    # Process the mcq_data
    mcq_data = data['mcq_data']

    # Iterate through each report/item in mcq_data
    for i, report in enumerate(mcq_data):
        # Swap answer choices for each question in the report
        for j, question in enumerate(report['questions']):
            total_questions += 1
            # Pass the rng instance instead of seed
            modified_question = swap_answer_choices(question, rng)
            
            # Only replace if the question was actually modified
            if modified_question != question:
                data['mcq_data'][i]['questions'][j] = modified_question
                modified_questions += 1

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Total questions processed: {total_questions}")
    print(f"Questions modified: {modified_questions}")
    print(f"Processed file saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate and Shuffle MCQs using Ollama')
    parser.add_argument('--input_csv', default=os.getenv("RRGEVAL_INPUT_CSV_PATH", "sample_reports.csv"), help='Input CSV file path')
    parser.add_argument('--output_dir', default=os.getenv("RRGEVAL_OUTPUT_DIR", "./output"), help='Output directory')
    parser.add_argument('--reference', choices=['gt', 'gen'], default='gt', help='Reference type')
    parser.add_argument('--num_questions', type=int, default=40, help='Number of questions per report')
    parser.add_argument('--seed', type=int, default=123, help='Random seed for reproducibility')
    parser.add_argument('--ollama_url', default="http://localhost:11434/api/generate", help='Ollama API URL')
    parser.add_argument('--model', default="llama3.1", help='Ollama model name')
    parser.add_argument('--max_tokens', type=int, default=10000, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0, help='Temperature for generation')
    parser.add_argument('--top_p', type=float, default=1, help='Top-p for generation')
    parser.add_argument('--timeout', type=int, default=300, help='Request timeout in seconds')

    args = parser.parse_args()

    seed = args.seed
    num_ques = args.num_questions
    reference = args.reference
    input_csv = args.input_csv
    output_dir = args.output_dir
    
    random.seed(seed)
    np.random.seed(seed)
    
    # import pdb; pdb.set_trace()
    df = pd.read_csv(input_csv)
    reports = df['ground_truth_report' if reference == 'gt' else 'generated_report'].tolist()
    print(f"Total unique reports: {len(reports)}")
    
    print(f"generating for {reference} reports")

    # Generate original MCQs
    output_dir = f"{output_dir}/orig_data/{reference}_reports_as_ref"
    ensure_dir(output_dir)
    json_output_file = f"{output_dir}/mcqa_data.json"
    
    total_reports, total_mcqs = generate_and_write_mcqs(
        reports, 
        num_ques, 
        json_output_file, 
        url=args.ollama_url,
        model=args.model,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=seed
    )
    
    print(f"MCQs saved to {json_output_file}")
    print(f"Total reports processed: {total_reports}")
    print(f"Total MCQs generated: {total_mcqs}")
    
    # Generate shuffled version
    shuffled_output_dir = f"{args.output_dir}/shuffled_ans_choices_data/{args.reference}_reports_as_ref"
    ensure_dir(shuffled_output_dir)
    shuffled_output_file = f"{shuffled_output_dir}/mcqa_data.json"
    
    # Process and save shuffled version
    process_json_file(json_output_file, shuffled_output_file, seed=seed)

if __name__ == "__main__":
    main()
