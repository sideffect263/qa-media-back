from transformers import pipeline
import sys
import json
import re
from typing import Dict, Optional, List
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QASystem:
    def __init__(self):
        """Initialize QA system with proper config."""
        try:
            self.qa_pipeline = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                device=-1  # Force CPU to avoid CUDA issues
            )
        except Exception as e:
            logger.error(f"Failed to initialize QA system: {str(e)}")
            raise

    def _split_context(self, context: str, max_length: int = 512) -> List[str]:
        """Split context into chunks that fit within model's max token limit"""
        words = context.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            # Rough estimate of tokens (words + some padding)
            word_length = len(word.split()) + 1
            if current_length + word_length > max_length:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def answer_question(self, question: str, context: str) -> Dict:
        """Get answer with improved processing."""
        start_time = datetime.now()
        
        try:
            # Clean inputs
            question = question.strip()
            context = context.strip()
            
            if not question or not context:
                return {
                    "answer": "Question or context is empty",
                    "confidence": 0.0,
                    "start": 0,
                    "end": 0,
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }

            # Split context into manageable chunks
            chunks = self._split_context(context)
            
            # Get answers from each chunk
            answers = []
            for chunk in chunks:
                try:
                    result = self.qa_pipeline(
                        question=question,
                        context=chunk,
                        max_answer_length=100,
                        handle_impossible_answer=True
                    )
                    answers.append({
                        "answer": result["answer"],
                        "confidence": float(result["score"]),
                        "start": result["start"],
                        "end": result["end"],
                        "chunk": chunk
                    })
                except Exception as e:
                    logger.warning(f"Error processing chunk: {str(e)}")
                    continue
            
            if not answers:
                return {
                    "answer": "Could not find a reliable answer in the text",
                    "confidence": 0.0,
                    "start": 0,
                    "end": 0,
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }
            
            # Get best answer based on confidence
            best_answer = max(answers, key=lambda x: x["confidence"])
            
            return {
                "answer": best_answer["answer"],
                "confidence": round(best_answer["confidence"] * 100, 2),
                "start": best_answer["start"],
                "end": best_answer["end"],
                "processing_time": (datetime.now() - start_time).total_seconds()
            }

        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {
                "answer": f"Error processing question: {str(e)}",
                "confidence": 0.0,
                "start": 0,
                "end": 0,
                "processing_time": (datetime.now() - start_time).total_seconds()
            }

def main():
    if len(sys.argv) != 3:
        print(json.dumps({
            "error": "Please provide both question and context file path"
        }))
        sys.exit(1)

    question = sys.argv[1]
    context_path = sys.argv[2]

    try:
        with open(context_path, 'r', encoding='utf-8') as f:
            context = f.read()

        if not context.strip():
            raise ValueError("Empty context provided")

        qa = QASystem()
        result = qa.answer_question(question, context)
        print(json.dumps(result))

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print(json.dumps({
            "answer": f"Error: {str(e)}",
            "confidence": 0.0,
            "start": 0,
            "end": 0,
            "processing_time": 0.0
        }))
        sys.exit(1)

if __name__ == "__main__":
    main()