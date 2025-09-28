import os
import json
import csv
import requests
from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import argparse
import random

def ensure_dir(directory: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)

def make_ollama_request(
    prompt: str,
    url: str = "http://localhost:11434/api/generate",
    model: str = "llama3.1",
    max_tokens: int = 2048,
    temperature: float = 0.0,
    timeout: int = 300,
    seed: int = 123,
    top_p: float = 1.0,
    stream: bool = False
) -> Optional[str]:
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
            "num_predict": max_tokens,
            "num_gpu": -1,
            "num_ctx": 2048
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "")
    except Exception as e:
        print(f"Error in Ollama API call: {e}")
        return None

def get_model_prediction(
    report: str, 
    question: str,
    options: Dict[str, str],
    pred_using_report_setting: str,
    url: str = "http://localhost:11434/api/generate",
    model: str = "llama3.1",
    timeout: int = 300,
    max_tokens: int = 10,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 123
) -> Optional[str]:
    """Get model's prediction for a question."""

    if pred_using_report_setting == "using_report":
        prompt = f"""Given the following radiology report:
        "{report}"
    
        Answer the following question:
        {question}
    
        Options:
        A) {options['A']}
        B) {options['B']}
        C) {options['C']}
        D) {options['D']}
        
        Your life depends on providing ONLY a single letter (A, B, C, or D) as your answer. 
        Do not include any other text, punctuation, or explanation.
        Format: Just the letter.
        Example correct format: A
        Example incorrect formats: A., The answer is A, Option A"""
    else:
        prompt = f"""Answer the following question:
        {question}
    
        Options:
        A) {options['A']}
        B) {options['B']}
        C) {options['C']}
        D) {options['D']}
    
        Your life depends on providing ONLY a single letter (A, B, C, or D) as your answer. 
        Do not include any other text, punctuation, or explanation.
        Format: Just the letter.
        Example correct format: A
        Example incorrect formats: A., The answer is A, Option A"""

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
    
    # Extract just the letter from response
    if response:
        # Clean the response to get just A, B, C, or D
        response = response.strip().upper()
        for letter in ['A', 'B', 'C', 'D']:
            if letter in response:
                return letter
    return None

def load_mcq_data_from_json(json_file: str, csv_file: str) -> List[Dict]:
    """Load MCQ data from JSON file and reports from CSV file."""
    # Load MCQ questions from JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Load reports from CSV
    df = pd.read_csv(csv_file)
    gt_report = df['ground_truth_report'].iloc[0]
    gen_report = df['generated_report'].iloc[0]
    
    mcq_data = []
    for report_data in data['mcq_data']:
        for question in report_data['questions']:
            mcq_data.append({
                'Report_ID': 0,
                'Question_ID': question['question_id'],
                'Question_Text': question['question_text'],
                'Options': question['options'],
                'Correct_Answer': question['correct_answer'],
                'GT_Report': gt_report,
                'Gen_Report': gen_report
            })
    
    return mcq_data

def predict_answers_for_mcq_data(
    mcq_data: List[Dict],
    output_csv_file: str,
    seed: int = 123
) -> List[Dict]:
    """
    Process MCQ data and generate predictions using both GT and generated reports.
    """
    # Prepare CSV output
    csv_headers = ['Index', 'Report_ID', 'Question_ID', 'Correct_Answer', 
                  'Predicted_Answer_Using_GT', 'Predicted_Answer_Using_Gen', 'Options']
    csvfile = open(output_csv_file, 'w', newline='')
    csv_writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
    csv_writer.writeheader()

    results = []
    idx = 0

    # Process each question
    for mcq in tqdm(mcq_data, desc="Processing MCQs"):
        report_id = mcq['Report_ID']
        question_id = mcq['Question_ID']
        question_text = mcq['Question_Text']
        options = mcq['Options']
        correct_answer = mcq['Correct_Answer']
        
        # Get model predictions using both GT and Gen reports
        predicted_answer_using_gt = get_model_prediction(mcq['GT_Report'], question_text, options, "using_report", seed=seed)
        predicted_answer_using_gen = get_model_prediction(mcq['Gen_Report'], question_text, options, "using_report", seed=seed)

        result = {
            'Index': idx,
            'Report_ID': report_id,
            'Question_ID': question_id,
            'Correct_Answer': correct_answer,
            'Predicted_Answer_Using_GT': predicted_answer_using_gt,
            'Predicted_Answer_Using_Gen': predicted_answer_using_gen,
            'Options': options
        }
        results.append(result)
        csv_writer.writerow(result)
        csvfile.flush()
        idx += 1

    csvfile.close()
    return results

def calculate_dataset_level_agreement(csv_file: str) -> Dict[str, float]:
    """
    Calculate agreement and disagreement percentages at the dataset level.
    """
    try:
        df = pd.read_csv(csv_file)
        total_questions = len(df)
        
        # Calculate agreement
        agreements = df['Predicted_Answer_Using_GT'] == df['Predicted_Answer_Using_Gen']
        agreement_count = agreements.sum()
        disagreement_count = total_questions - agreement_count
        
        agreement_pct = (agreement_count / total_questions) * 100
        disagreement_pct = (disagreement_count / total_questions) * 100
        
        return {
            'total_questions': total_questions,
            'agreement_count': int(agreement_count),
            'disagreement_count': int(disagreement_count),
            'agreement_percentage': round(agreement_pct, 2),
            'disagreement_percentage': round(disagreement_pct, 2)
        }
    except Exception as e:
        print(f"Error calculating dataset agreement: {e}")
        return None

def plot_report_level_agreement(csv_file: str, output_dir: str, reference: str):
    """
    Plot report-level agreement statistics.
    """
    try:
        df = pd.read_csv(csv_file)
        report_agreements = []
        
        # Calculate agreement percentage for each report
        for report_id in df['Report_ID'].unique():
            report_df = df[df['Report_ID'] == report_id]
            total = len(report_df)
            matches = len(report_df[report_df['Predicted_Answer_Using_GT'] == 
                                  report_df['Predicted_Answer_Using_Gen']])
            agreement_pct = (matches / total) * 100 if total > 0 else 0
            report_agreements.append(agreement_pct)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.hist(report_agreements, bins=20, edgecolor='black')
        plt.xlabel('Agreement Percentage', fontsize=24, fontweight='bold', labelpad=15)
        plt.ylabel('Number of Reports', fontsize=24, fontweight='bold', labelpad=15)
        plt.xticks(fontsize=22, weight='bold')
        plt.yticks(fontsize=22, weight='bold')
        
        # Add mean and std dev lines
        mean_agreement = np.mean(report_agreements)
        std_agreement = np.std(report_agreements)
        plt.axvline(mean_agreement, color='r', linestyle='dashed', linewidth=2, 
                   label=f'Mean: {mean_agreement:.1f}%')
        plt.axvline(mean_agreement + std_agreement, color='g', linestyle=':', linewidth=2,
                   label=f'SD: {std_agreement:.1f}%')
        plt.axvline(mean_agreement - std_agreement, color='g', linestyle=':', linewidth=2)
        
        plt.legend(fontsize=24, prop={'weight': 'bold', 'size': 16})
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_file = os.path.join(output_dir, f'mcq_eval_report_level_agreement_hist.png')
        plt.savefig(plot_file, dpi=600, bbox_inches='tight')
        plt.close()
        
        # Save report-level statistics
        report_stats = pd.DataFrame({
            'Report_ID': df['Report_ID'].unique(),
            'Agreement_Percentage': report_agreements
        })
        stats_file = os.path.join(output_dir, f'mcq_eval_report_level_stats.csv')
        report_stats.to_csv(stats_file, index=False)

        aggregated_stats_file = os.path.join(output_dir, f'mcq_eval_report_level_stats_aggregated.csv')
        aggregated_stats = pd.DataFrame({
            'Mean_Agreement': [mean_agreement],
            'Std_Deviation': [std_agreement]
        })
        aggregated_stats.to_csv(aggregated_stats_file, index=False)
        
        print(f"\nReport-level statistics:")
        print(f"Mean agreement: {mean_agreement:.1f}%")
        print(f"Standard deviation: {std_agreement:.1f}%")
        print(f"Plot saved to {plot_file}")
        print(f"Report-level statistics saved to {stats_file}")
        
    except Exception as e:
        print(f"Error plotting report-level agreement: {e}")

def main():
    parser = argparse.ArgumentParser(description='MCQ Evaluation Script using Ollama')
    parser.add_argument('--input_csv', type=str, default='./sample_reports.csv',
                      help='Input CSV file with GT and Gen reports (default: ./sample_reports.csv)')
    parser.add_argument('--output_dir', type=str, default='./output/mcqa_eval',
                      help='Output directory (default: ./output/mcqa_eval)')
    parser.add_argument('--seed', type=int, default=123,
                      help='Random seed (default: 123)')
    parser.add_argument('--ollama_url', type=str, default='http://localhost:11434/api/generate',
                      help='Ollama API URL (default: http://localhost:11434/api/generate)')
    parser.add_argument('--model', type=str, default='llama3.1',
                      help='Ollama model name (default: llama3.1)')
    parser.add_argument('--max_tokens', type=int, default=10,
                      help='Maximum tokens for prediction (default: 10)')
    parser.add_argument('--temperature', type=float, default=0.0,
                      help='Temperature for generation (default: 0.0)')
    parser.add_argument('--timeout', type=int, default=300,
                      help='Request timeout in seconds (default: 300)')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    ensure_dir(args.output_dir)
    
    # Load reports from CSV
    print(f"Loading reports from {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    gt_report = df['ground_truth_report'].iloc[0]
    gen_report = df['generated_report'].iloc[0]
    print(f"Loaded GT and Gen reports")
    
    # Process both GT and Gen question scenarios
    for question_type in ['gt', 'gen']:
        print(f"\n{'='*50}")
        print(f"EVALUATING {question_type.upper()} QUESTIONS")
        print(f"{'='*50}")
        
        # Load questions generated from this report type
        if question_type == 'gt':
            json_file = './output/shuffled_ans_choices_data/gt_reports_as_ref/mcqa_data.json'
        else:
            json_file = './output/shuffled_ans_choices_data/gen_reports_as_ref/mcqa_data.json'
        
        print(f"Loading {question_type} questions from {json_file}")
        mcq_data = load_mcq_data_from_json(json_file, args.input_csv)
        print(f"Loaded {len(mcq_data)} questions generated from {question_type} reports")
        
        if len(mcq_data) == 0:
            print(f"No questions found for {question_type} scenario. Skipping...")
            continue
        
        # Output file for this scenario
        output_csv_file = os.path.join(args.output_dir, f"mcqa_eval_{question_type}_questions_predictions.csv")
        
        print(f"Processing MCQ evaluation for {question_type} questions...")
        results = predict_answers_for_mcq_data( 
            mcq_data,
            output_csv_file,
            seed=args.seed
        )
        print(f"Results saved to {output_csv_file}")
        
        # Calculate dataset-level agreement
        agreement_stats = calculate_dataset_level_agreement(output_csv_file)
        if agreement_stats:
            print(f"\n{question_type.upper()} Questions - Dataset-level statistics:")
            print(f"Total questions: {agreement_stats['total_questions']}")
            print(f"Agreement: {agreement_stats['agreement_percentage']}%")
            print(f"Disagreement: {agreement_stats['disagreement_percentage']}%")
            
            # Save agreement stats
            stats_file = os.path.join(args.output_dir, f'mcq_eval_{question_type}_questions_agreement_stats.csv')
            pd.DataFrame([agreement_stats]).to_csv(stats_file, index=False)
            print(f"Agreement statistics saved to {stats_file}")
        
        # Generate report-level analysis
        plot_report_level_agreement(output_csv_file, args.output_dir, question_type)
    
    print(f"\n{'='*50}")
    print("EVALUATION COMPLETE")
    print(f"{'='*50}")
    print("Summary:")
    print("- GT Questions: Questions generated from Ground Truth reports")
    print("- Gen Questions: Questions generated from Generated reports")
    print("- Both evaluated using both GT and Gen reports for answering")
    print("- Agreement measures consistency between GT and Gen report usage")

if __name__ == "__main__":
    main()
