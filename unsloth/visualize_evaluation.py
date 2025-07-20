#!/usr/bin/env python3
"""
Script visualize kết quả evaluation
Tạo biểu đồ so sánh model trước và sau finetuning
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style cho matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EvaluationVisualizer:
    """Class để visualize kết quả evaluation"""
    
    def __init__(self):
        self.results_dir = Path("./evaluation_results")
        self.output_dir = Path("./evaluation_visualizations")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load results
        self.simple_results = self.load_simple_results()
        self.full_results = self.load_full_results()
    
    def load_simple_results(self) -> Dict[str, Any]:
        """Load kết quả từ simple evaluation"""
        try:
            with open(self.results_dir / "simple_evaluation.json", 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Simple evaluation results not found")
            return None
    
    def load_full_results(self) -> Dict[str, Any]:
        """Load kết quả từ full evaluation"""
        try:
            base_results = {}
            finetuned_results = {}
            comparison_results = {}
            
            # Load base model results
            if (self.results_dir / "base_model_evaluation.json").exists():
                with open(self.results_dir / "base_model_evaluation.json", 'r', encoding='utf-8') as f:
                    base_results = json.load(f)
            
            # Load finetuned model results
            if (self.results_dir / "finetuned_model_evaluation.json").exists():
                with open(self.results_dir / "finetuned_model_evaluation.json", 'r', encoding='utf-8') as f:
                    finetuned_results = json.load(f)
            
            # Load comparison results
            if (self.results_dir / "model_comparison.json").exists():
                with open(self.results_dir / "model_comparison.json", 'r', encoding='utf-8') as f:
                    comparison_results = json.load(f)
            
            return {
                "base": base_results,
                "finetuned": finetuned_results,
                "comparison": comparison_results
            }
        except Exception as e:
            logger.warning(f"Error loading full results: {e}")
            return {}
    
    def create_metrics_comparison_chart(self):
        """Tạo biểu đồ so sánh metrics"""
        if not self.simple_results:
            logger.warning("No simple results available for visualization")
            return
        
        # Extract metrics data
        base_metrics = self.simple_results["base_model"]["metrics"]
        finetuned_metrics = self.simple_results["finetuned_model"]["metrics"]
        
        metrics_names = list(base_metrics.keys())
        base_scores = [base_metrics[m]["mean"] for m in metrics_names]
        finetuned_scores = [finetuned_metrics[m]["mean"] for m in metrics_names]
        
        # Create DataFrame
        df = pd.DataFrame({
            'Metric': metrics_names * 2,
            'Score': base_scores + finetuned_scores,
            'Model': ['Base Model'] * len(metrics_names) + ['Finetuned Model'] * len(metrics_names)
        })
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Bar plot
        ax = sns.barplot(data=df, x='Metric', y='Score', hue='Model', palette=['#ff7f0e', '#2ca02c'])
        
        # Customize plot
        plt.title('Model Performance Comparison by Metrics', fontsize=16, fontweight='bold')
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.ylim(0, 1.0)
        
        # Add value labels on bars
        for i, p in enumerate(ax.patches):
            ax.annotate(f'{p.get_height():.3f}', 
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='bottom', fontsize=10)
        
        plt.legend(title='Model', title_fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(self.output_dir / "metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Metrics comparison chart saved")
    
    def create_weighted_score_chart(self):
        """Tạo biểu đồ so sánh weighted score"""
        if not self.simple_results:
            return
        
        base_score = self.simple_results["base_model"]["weighted_score"]
        finetuned_score = self.simple_results["finetuned_model"]["weighted_score"]
        improvement = self.simple_results["comparison"]["improvement"]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar chart
        models = ['Base Model', 'Finetuned Model']
        scores = [base_score, finetuned_score]
        colors = ['#ff7f0e', '#2ca02c']
        
        bars = ax1.bar(models, scores, color=colors, alpha=0.8)
        ax1.set_title('Weighted Score Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Weighted Score', fontsize=12)
        ax1.set_ylim(0, max(scores) * 1.2)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Improvement chart
        ax2.bar(['Improvement'], [improvement], 
               color='green' if improvement > 0 else 'red', alpha=0.8)
        ax2.set_title('Performance Improvement', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Score Improvement', fontsize=12)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value label
        ax2.text(0, improvement + (0.01 if improvement > 0 else -0.01),
                f'{improvement:+.3f}', ha='center', 
                va='bottom' if improvement > 0 else 'top', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "weighted_score_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Weighted score comparison chart saved")
    
    def create_question_response_comparison(self):
        """Tạo biểu đồ so sánh response cho từng câu hỏi"""
        if not self.simple_results:
            return
        
        base_responses = self.simple_results["base_model"]["responses"]
        finetuned_responses = self.simple_results["finetuned_model"]["responses"]
        
        # Create DataFrame for each metric
        metrics = ['relevancy', 'faithfulness', 'correctness']
        
        for metric in metrics:
            questions = [f"Q{i+1}" for i in range(len(base_responses))]
            base_scores = [r[metric] for r in base_responses]
            finetuned_scores = [r[metric] for r in finetuned_responses]
            
            df = pd.DataFrame({
                'Question': questions * 2,
                'Score': base_scores + finetuned_scores,
                'Model': ['Base'] * len(questions) + ['Finetuned'] * len(questions)
            })
            
            # Create plot
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(data=df, x='Question', y='Score', hue='Model', palette=['#ff7f0e', '#2ca02c'])
            
            plt.title(f'{metric.title()} Scores by Question', fontsize=14, fontweight='bold')
            plt.xlabel('Questions', fontsize=12)
            plt.ylabel(f'{metric.title()} Score', fontsize=12)
            plt.ylim(0, 1.0)
            
            # Add value labels
            for i, p in enumerate(ax.patches):
                ax.annotate(f'{p.get_height():.3f}', 
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='bottom', fontsize=9)
            
            plt.legend(title='Model')
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{metric}_by_question.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("Question response comparison charts saved")
    
    def create_radar_chart(self):
        """Tạo biểu đồ radar để so sánh metrics"""
        if not self.simple_results:
            return
        
        base_metrics = self.simple_results["base_model"]["metrics"]
        finetuned_metrics = self.simple_results["finetuned_model"]["metrics"]
        
        metrics_names = list(base_metrics.keys())
        base_scores = [base_metrics[m]["mean"] for m in metrics_names]
        finetuned_scores = [finetuned_metrics[m]["mean"] for m in metrics_names]
        
        # Close the plot by appending first value
        base_scores += base_scores[:1]
        finetuned_scores += finetuned_scores[:1]
        metrics_names += metrics_names[:1]
        
        # Calculate angles
        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
        angles += angles[:1]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Plot data
        ax.plot(angles, base_scores, 'o-', linewidth=2, label='Base Model', color='#ff7f0e')
        ax.fill(angles, base_scores, alpha=0.25, color='#ff7f0e')
        
        ax.plot(angles, finetuned_scores, 'o-', linewidth=2, label='Finetuned Model', color='#2ca02c')
        ax.fill(angles, finetuned_scores, alpha=0.25, color='#2ca02c')
        
        # Customize plot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_names[:-1])
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar Chart', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # Add grid
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "radar_chart.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Radar chart saved")
    
    def create_summary_report(self):
        """Tạo báo cáo tóm tắt"""
        if not self.simple_results:
            return
        
        report = f"""
# Báo cáo đánh giá Model

## Thông tin chung
- Thời gian đánh giá: {self.simple_results.get('evaluation_time', 'N/A')}
- Số lượng câu hỏi test: {len(self.simple_results['base_model']['responses'])}

## Kết quả tổng quan
- **Base Model Score**: {self.simple_results['base_model']['weighted_score']:.3f}
- **Finetuned Model Score**: {self.simple_results['finetuned_model']['weighted_score']:.3f}
- **Cải thiện**: {self.simple_results['comparison']['improvement']:.3f} ({self.simple_results['comparison']['improvement_percentage']:.1f}%)

## Chi tiết metrics

### Base Model
"""
        
        for metric, data in self.simple_results['base_model']['metrics'].items():
            report += f"- **{metric.replace('_', ' ').title()}**: {data['mean']:.3f}\n"
        
        report += "\n### Finetuned Model\n"
        for metric, data in self.simple_results['finetuned_model']['metrics'].items():
            report += f"- **{metric.replace('_', ' ').title()}**: {data['mean']:.3f}\n"
        
        report += "\n## Chi tiết câu trả lời\n"
        for i, (base_resp, finetuned_resp) in enumerate(zip(
            self.simple_results['base_model']['responses'],
            self.simple_results['finetuned_model']['responses']
        )):
            report += f"""
### Câu hỏi {i+1}: {base_resp['question']}
**Expected**: {base_resp['expected']}

**Base Model Response**: {base_resp['response'][:200]}...
- Relevancy: {base_resp['relevancy']:.3f}
- Faithfulness: {base_resp['faithfulness']:.3f}
- Correctness: {base_resp['correctness']:.3f}

**Finetuned Model Response**: {finetuned_resp['response'][:200]}...
- Relevancy: {finetuned_resp['relevancy']:.3f}
- Faithfulness: {finetuned_resp['faithfulness']:.3f}
- Correctness: {finetuned_resp['correctness']:.3f}
"""
        
        # Save report
        with open(self.output_dir / "evaluation_report.md", 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info("Summary report saved")
    
    def run_visualization(self):
        """Chạy toàn bộ visualization"""
        logger.info("Starting evaluation visualization...")
        
        try:
            # Create all charts
            self.create_metrics_comparison_chart()
            self.create_weighted_score_chart()
            self.create_question_response_comparison()
            self.create_radar_chart()
            self.create_summary_report()
            
            logger.info("All visualizations completed successfully!")
            logger.info(f"Output saved to: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error during visualization: {e}")
            raise

def main():
    """Main function"""
    visualizer = EvaluationVisualizer()
    visualizer.run_visualization()

if __name__ == "__main__":
    main() 