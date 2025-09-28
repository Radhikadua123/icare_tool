import streamlit as st
import pandas as pd
import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from mcqa_evaluation_ollama import (
    load_mcq_data_from_json, 
    predict_answers_for_mcq_data,
    calculate_dataset_level_agreement,
    make_ollama_request,
    get_model_prediction
)
from mcq_generation_ollama import generate_and_write_mcqs, ensure_dir
import tempfile

# Page config
st.set_page_config(
    page_title="ICARE",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<div class="main-header">🏥 ICARE</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This demo shows how we generate multiple choice questions from radiology reports and evaluate 
    their quality by testing agreement between different report versions.
    """)
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    model_name = st.sidebar.selectbox("Ollama Model", ["llama3.1", "gpt-oss:20b"])
    num_questions = st.sidebar.slider("Number of Questions", 1, 30, 5)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    
    # # GPU Info
    # st.sidebar.header("🚀 Performance")
    # st.sidebar.info("""
    # **GPU Optimized Settings:**
    # - Using gpt-oss:20b model
    # - GPU acceleration enabled
    # - Optimized batch processing
    # - Enhanced context window
    # """)
    
    # Main interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">👨‍⚕️ Ground Truth Report</div>', unsafe_allow_html=True)
        gt_report = st.text_area(
            "Enter Ground Truth Report:",
            value="the cardiomediastinal silhouette is normal. the lung parenchyma is clear. there are no pleural or significant bony abnormalities.",
            height=200,
            help="Enter the original, ground truth radiology report"
        )
    
    with col2:
        st.markdown('<div class="section-header">📝 Generated Report</div>', unsafe_allow_html=True)
        gen_report = st.text_area(
            "Enter Generated Report:",
            value="The lungs are clear. No pneumothorax or effusion. Unremarkable cardiomediastinal silhouette.",
            height=200,
            help="Enter the AI-generated or modified radiology report"
        )
    
    # Compare button
    if st.button("🔍 Compare Reports", type="primary", use_container_width=True):
        with st.spinner("Processing... This may take a few minutes."):
            # Create temporary files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save reports to CSV
                reports_df = pd.DataFrame({
                    'ground_truth_report': [gt_report],
                    'generated_report': [gen_report]
                })
                csv_path = os.path.join(temp_dir, 'reports.csv')
                reports_df.to_csv(csv_path, index=False)
                
                # Generate MCQs for both report types
                results = {}
                
                for report_type in ['gt', 'gen']:
                    st.markdown(f'<div class="section-header">🔄 Generating MCQs from {report_type.upper()} Report</div>', unsafe_allow_html=True)
                    
                    # Create output directory
                    output_dir = os.path.join(temp_dir, f'{report_type}_output')
                    ensure_dir(output_dir)
                    
                    # Generate MCQs
                    reports = [gt_report if report_type == 'gt' else gen_report]
                    json_file = os.path.join(output_dir, 'mcqa_data.json')
                    
                    with st.status(f"Generating {num_questions} questions from {report_type.upper()} report...", expanded=True):
                        st.write(f"🤖 Calling Ollama API with {num_questions} questions...")
                        total_reports, total_mcqs = generate_and_write_mcqs(
                            reports, 
                            num_questions, 
                            json_file,
                            model=model_name,
                            temperature=temperature,
                            timeout=120  # Increased timeout for GPU processing
                        )
                        st.write(f"✅ Generated {total_mcqs} MCQs")
                    
                    # Load and display questions
                    if os.path.exists(json_file):
                        with open(json_file, 'r') as f:
                            mcq_data = json.load(f)
                        
                        if mcq_data['mcq_data'] and mcq_data['mcq_data'][0]['questions']:
                            st.markdown(f"**Generated Questions from {report_type.upper()} Report:**")
                            
                            for i, question in enumerate(mcq_data['mcq_data'][0]['questions'][:num_questions]):  # Show selected number
                                with st.expander(f"Question {i+1}: {question['question_text'][:50]}..."):
                                    st.write(f"**Question:** {question['question_text']}")
                                    st.write(f"**A)** {question['options']['A']}")
                                    st.write(f"**B)** {question['options']['B']}")
                                    st.write(f"**C)** {question['options']['C']}")
                                    st.write(f"**D)** {question['options']['D']}")
                                    st.write(f"**Correct Answer:** {question['correct_answer']}")
                            
                            # Evaluate questions
                            st.markdown(f'<div class="section-header">📊 Evaluating {report_type.upper()} Questions</div>', unsafe_allow_html=True)
                            
                            with st.status(f"Evaluating questions using both reports...", expanded=True):
                                st.write("🤖 Getting predictions using GT report...")
                                st.write("🤖 Getting predictions using Gen report...")
                                
                                # Load MCQ data for evaluation
                                mcq_data_list = load_mcq_data_from_json(json_file, csv_path)
                                
                                if mcq_data_list:
                                    # Create evaluation results
                                    evaluation_results = []
                                    
                                    for mcq in mcq_data_list:
                                        # Get predictions using both reports
                                        pred_gt = get_model_prediction(
                                            gt_report, 
                                            mcq['Question_Text'], 
                                            mcq['Options'], 
                                            "using_report",
                                            model=model_name,
                                            temperature=temperature
                                        )
                                        pred_gen = get_model_prediction(
                                            gen_report, 
                                            mcq['Question_Text'], 
                                            mcq['Options'], 
                                            "using_report",
                                            model=model_name,
                                            temperature=temperature
                                        )
                                        
                                        evaluation_results.append({
                                            'Question': mcq['Question_Text'],
                                            'Correct_Answer': mcq['Correct_Answer'],
                                            'Pred_GT': pred_gt,
                                            'Pred_Gen': pred_gen,
                                            'Agreement': pred_gt == pred_gen if pred_gt and pred_gen else False
                                        })
                                    
                                    # Calculate comprehensive metrics based on ICARE framework
                                    total_questions = len(evaluation_results)
                                    agreements = sum(1 for r in evaluation_results if r['Agreement'])
                                    agreement_pct = (agreements / total_questions) * 100 if total_questions > 0 else 0
                                    
                                    # Calculate GT vs Gen performance metrics
                                    gt_correct = sum(1 for r in evaluation_results if r['Pred_GT'] == r['Correct_Answer'])
                                    gen_correct = sum(1 for r in evaluation_results if r['Pred_Gen'] == r['Correct_Answer'])
                                    
                                    # ICARE Framework: Different metrics based on question source
                                    if report_type == 'gt':
                                        # GT Questions → ICARE-GT → Precision + Omission
                                        precision = (gt_correct / total_questions) * 100 if total_questions > 0 else 0
                                        recall = None  # Not applicable for GT questions
                                        
                                        # Omission: GT correct but Gen wrong (Gen report failed to capture GT info)
                                        omission = sum(1 for r in evaluation_results 
                                                      if r['Pred_GT'] == r['Correct_Answer'] and r['Pred_Gen'] != r['Correct_Answer'])
                                        omission_pct = (omission / total_questions) * 100 if total_questions > 0 else 0
                                        hallucination_pct = None  # Not applicable for GT questions
                                        
                                    else:  # report_type == 'gen'
                                        # Gen Questions → ICARE-GEN → Recall + Hallucination
                                        precision = None  # Not applicable for Gen questions
                                        recall = (gen_correct / total_questions) * 100 if total_questions > 0 else 0
                                        
                                        # Hallucination: Gen correct but GT wrong (unsupported addition in Gen)
                                        hallucination = sum(1 for r in evaluation_results 
                                                         if r['Pred_Gen'] == r['Correct_Answer'] and r['Pred_GT'] != r['Correct_Answer'])
                                        hallucination_pct = (hallucination / total_questions) * 100 if total_questions > 0 else 0
                                        omission_pct = None  # Not applicable for Gen questions
                                    
                                    # Display basic results
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Total Questions", total_questions)
                                    with col2:
                                        st.metric("Agreements", agreements)
                                    with col3:
                                        st.metric("Agreement %", f"{agreement_pct:.1f}%")
                                    
                                    
                                    if report_type == 'gt':
                                        # GT Questions → ICARE-GT → Precision + Omission
                                        st.markdown("#### 🎯 ICARE-GT: Precision Analysis")
                                        st.markdown("*Measures whether the generated report preserved clinically important findings from the ground-truth.*")
                                        
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("GT Correct Answers", gt_correct, f"{gt_correct}/{total_questions}")
                                        with col2:
                                            st.metric("Precision", f"{precision:.1f}%", 
                                                    "How well GT report preserves findings")
                                        with col3:
                                            st.metric("Omission", f"{omission_pct:.1f}%", 
                                                    f"{omission} cases where Gen missed info")
                                        
                                        # Explanation
                                        st.info("""
                                        **ICARE-GT Interpretation:**
                                        - **Precision**: How well the generated report preserves clinically important findings from ground-truth
                                        - **Omission**: Cases where GT info was not captured in the Gen report
                                        """)
                                        
                                    else:  # report_type == 'gen'
                                        # Gen Questions → ICARE-GEN → Recall + Hallucination
                                        st.markdown("#### 🤖 ICARE-GEN: Recall Analysis")
                                        st.markdown("*Measures whether extra content in the generated report is clinically consistent with the ground-truth.*")
                                        
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Gen Correct Answers", gen_correct, f"{gen_correct}/{total_questions}")
                                        with col2:
                                            st.metric("Recall", f"{recall:.1f}%", 
                                                    "How well Gen report content is consistent")
                                        with col3:
                                            st.metric("Hallucination", f"{hallucination_pct:.1f}%", 
                                                    f"{hallucination} unsupported additions by Gen")
                                        
                                        # Explanation
                                        st.info("""
                                        **ICARE-GEN Interpretation:**
                                        - **Recall**: How well the generated report's extra content is clinically consistent with ground-truth
                                        - **Hallucination**: Cases where the Gen report introduced unsupported content
                                        """)
                                    
                                    # Performance comparison
                                    st.markdown("#### 📈 Report Performance Comparison")
                                    if report_type == 'gt':
                                        perf_data = {
                                            'Metric': ['GT Correct', 'Gen Correct', 'Precision', 'Omission'],
                                            'Value': [f"{gt_correct}/{total_questions}", f"{gen_correct}/{total_questions}", 
                                                    f"{precision:.1f}%", f"{omission_pct:.1f}%"]
                                        }
                                    else:
                                        perf_data = {
                                            'Metric': ['GT Correct', 'Gen Correct', 'Recall', 'Hallucination'],
                                            'Value': [f"{gt_correct}/{total_questions}", f"{gen_correct}/{total_questions}", 
                                                    f"{recall:.1f}%", f"{hallucination_pct:.1f}%"]
                                        }
                                    perf_df = pd.DataFrame(perf_data)
                                    st.dataframe(perf_df, use_container_width=True)
                                    
                                    # Show sample results
                                    st.markdown("**Sample Results:**")
                                    results_df = pd.DataFrame(evaluation_results)
                                    
                                    # High agreement examples
                                    high_agreement = results_df[results_df['Agreement'] == True].head(2)
                                    if not high_agreement.empty:
                                        st.markdown("✅ **High Agreement Examples:**")
                                        for _, row in high_agreement.iterrows():
                                            st.write(f"**Q:** {row['Question'][:60]}...")
                                            st.write(f"**GT Answer:** {row['Pred_GT']} | **Gen Answer:** {row['Pred_Gen']} | **Correct:** {row['Correct_Answer']}")
                                            st.write("---")
                                    
                                    # Low agreement examples
                                    low_agreement = results_df[results_df['Agreement'] == False].head(2)
                                    if not low_agreement.empty:
                                        st.markdown("❌ **Low Agreement Examples:**")
                                        for _, row in low_agreement.iterrows():
                                            st.write(f"**Q:** {row['Question'][:60]}...")
                                            st.write(f"**GT Answer:** {row['Pred_GT']} | **Gen Answer:** {row['Pred_Gen']} | **Correct:** {row['Correct_Answer']}")
                                            st.write("---")
                                    
                                    results[report_type] = {
                                        'total_questions': total_questions,
                                        'agreements': agreements,
                                        'agreement_pct': agreement_pct,
                                        'gt_correct': gt_correct,
                                        'gen_correct': gen_correct,
                                        'precision': precision,
                                        'recall': recall,
                                        'hallucination_pct': hallucination_pct,
                                        'omission_pct': omission_pct,
                                        'results_df': results_df
                                    }
                                else:
                                    st.warning(f"No questions found in {report_type} report")
                        else:
                            st.warning(f"No questions generated from {report_type} report")
                    else:
                        st.error(f"Failed to generate questions from {report_type} report")
                
                # Summary comparison
                if 'gt' in results and 'gen' in results:
                    st.markdown('<div class="section-header">📈 Summary Comparison</div>', unsafe_allow_html=True)
                    
                    comparison_data = {
                        'Question Type': ['GT Questions (ICARE-GT)', 'Gen Questions (ICARE-GEN)'],
                        'Total Questions': [results['gt']['total_questions'], results['gen']['total_questions']],
                        'Agreements': [results['gt']['agreements'], results['gen']['agreements']],
                        'Agreement %': [f"{results['gt']['agreement_pct']:.1f}%", f"{results['gen']['agreement_pct']:.1f}%"],
                        'Precision': [f"{results['gt']['precision']:.1f}%" if results['gt']['precision'] else "N/A", "N/A"],
                        'Recall': ["N/A", f"{results['gen']['recall']:.1f}%" if results['gen']['recall'] else "N/A"],
                        'Omission %': ["N/A", f"{results['gt']['omission_pct']:.1f}%"],
                        'Hallucination %': [f"{results['gen']['hallucination_pct']:.1f}%", "N/A"]
                    }
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Visualization
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Agreement comparison
                    report_types = ['GT Questions', 'Gen Questions']
                    agreement_pcts = [results['gt']['agreement_pct'], results['gen']['agreement_pct']]
                    
                    bars = ax1.bar(report_types, agreement_pcts, color=['#2ecc71', '#e74c3c'])
                    ax1.set_ylabel('Agreement %')
                    ax1.set_title('Agreement Comparison')
                    ax1.set_ylim(0, 100)
                    
                    for bar, pct in zip(bars, agreement_pcts):
                        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                               f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
                    
                    # Question count comparison
                    question_counts = [results['gt']['total_questions'], results['gen']['total_questions']]
                    bars2 = ax2.bar(report_types, question_counts, color=['#3498db', '#f39c12'])
                    ax2.set_ylabel('Number of Questions')
                    ax2.set_title('Questions Generated')
                    
                    for bar, count in zip(bars2, question_counts):
                        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                               str(count), ha='center', va='bottom', fontweight='bold')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Insights
                    st.markdown("### 🔍 Key Insights:")

                    gt_agreement = results['gt']['agreement_pct']
                    gen_agreement = results['gen']['agreement_pct']

                    # Compare agreement
                    if gt_agreement > gen_agreement:
                        st.success(f"✅ GT-generated questions show higher agreement ({gt_agreement:.1f}% vs {gen_agreement:.1f}%)")
                        st.info("This suggests the generated report may have introduced **hallucinations** (extra content not supported by GT).")
                    elif gen_agreement > gt_agreement:
                        st.success(f"✅ Gen-generated questions show higher agreement ({gen_agreement:.1f}% vs {gt_agreement:.1f}%)")
                        st.info("This suggests the generated report may have more **omissions** (missing content from GT).")
                    else:
                        st.info("ℹ️ Both question types show similar agreement levels.")

                    # Absolute benchmarks
                    if gt_agreement > 90:
                        st.success("🎯 GT questions show excellent consistency (>90% agreement).")
                    if gen_agreement > 90:
                        st.success("🎯 Gen questions show excellent consistency (>90% agreement).")

                    # If there’s a big gap (>15%) highlight likely error type
                    if abs(gt_agreement - gen_agreement) > 15:
                        if gt_agreement > gen_agreement:
                            st.warning("⚠️ The large gap indicates the generated report likely contains **hallucinations**.")
                        else:
                            st.warning("⚠️ The large gap indicates the generated report likely contains **omissions**.")
if __name__ == "__main__":
    main()
