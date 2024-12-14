from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import torch
import sys
import json
import re
import logging
from typing import Dict, Optional, List, Union
from dataclasses import dataclass
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class QAResult:
    """Data class for QA results with structured output"""
    answer: str
    confidence: float
    start: int
    end: int
    context: str
    processing_time: float

class AdvancedQA:
    def __init__(self, model_name: str = "distilbert-base-cased-distilled-squad"):
        """
        Initialize QA system with advanced configuration.
        
        Args:
            model_name: Name of the HuggingFace model to use
        """
        try:
            logger.info(f"Initializing QA system with model: {model_name}")
            
            # Set device with proper error handling
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            # Initialize tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            self.model = self.model.to(self.device)
            
            # Initialize pipeline with advanced configuration
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1 if self.device == "cpu" else 0,
                handle_long_sequences=True
            )
            
            # Initialize regex patterns
            self._compile_regex_patterns()
            
        except Exception as e:
            logger.error(f"Failed to initialize QA system: {str(e)}")
            raise

    def _compile_regex_patterns(self):
        """Compile regex patterns for efficient reuse"""
        self.patterns = {
            'week': re.compile(r'week\s+(\d+)', re.IGNORECASE),
            'date': re.compile(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b'),
            'time': re.compile(r'\b\d{1,2}:\d{2}\s*(?:AM|PM)?\b', re.IGNORECASE),
            'number': re.compile(r'\b\d+(?:\.\d+)?\b'),
            'money': re.compile(r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?'),
            'percentage': re.compile(r'\b\d+(?:\.\d+)?%\b'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        }

    def _preprocess_text(self, text: str) -> str:
        """
        Clean and normalize text for processing.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned and normalized text
        """
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Normalize quotes and dashes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace('–', '-').replace('—', '-')
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char == '\n')
        
        return text

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract various entities from text using regex patterns.
        
        Args:
            text: Input text to process
            
        Returns:
            Dictionary of extracted entities by type
        """
        entities = {}
        for name, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                entities[name] = matches
        return entities

    def _get_best_segment(self, question: str, text: str, window_size: int = 512) -> str:
        """
        Find the most relevant text segment for the question.
        
        Args:
            question: Question to answer
            text: Full context text
            window_size: Size of text window to consider
            
        Returns:
            Most relevant text segment
        """
        # Tokenize question and text
        question_tokens = self.tokenizer.tokenize(question)
        text_tokens = self.tokenizer.tokenize(text)
        
        if len(text_tokens) <= window_size:
            return text
            
        # Create windows of text
        windows = []
        for i in range(0, len(text_tokens) - window_size + 1, window_size // 2):
            window = text_tokens[i:i + window_size]
            windows.append(window)
            
        # Score each window based on token overlap
        scores = []
        for window in windows:
            overlap = len(set(question_tokens) & set(window))
            scores.append(overlap)
            
        # Get best window
        best_window_idx = np.argmax(scores)
        best_window = windows[best_window_idx]
        
        return self.tokenizer.convert_tokens_to_string(best_window)

    def answer_question(self, question: str, context: str) -> QAResult:
        """
        Get answer for a question with improved processing and handling.
        
        Args:
            question: Question to answer
            context: Context text containing the answer
            
        Returns:
            QAResult object containing answer and metadata
        """
        start_time = datetime.now()
        
        try:
            # Clean inputs
            question = self._preprocess_text(question)
            context = self._preprocess_text(context)
            
            if not question or not context:
                raise ValueError("Question or context is empty")
                
            # Extract entities for special handling
            entities = self._extract_entities(context)
            
            # Handle special question types
            lower_question = question.lower()
            
            # Date/time questions
            if any(word in lower_question for word in ['when', 'what time', 'what date']):
                if 'date' in entities:
                    return QAResult(
                        answer=entities['date'][0],
                        confidence=0.95,
                        start=context.find(entities['date'][0]),
                        end=context.find(entities['date'][0]) + len(entities['date'][0]),
                        context=context,
                        processing_time=(datetime.now() - start_time).total_seconds()
                    )
                if 'time' in entities:
                    return QAResult(
                        answer=entities['time'][0],
                        confidence=0.95,
                        start=context.find(entities['time'][0]),
                        end=context.find(entities['time'][0]) + len(entities['time'][0]),
                        context=context,
                        processing_time=(datetime.now() - start_time).total_seconds()
                    )
            
            # Numeric questions
            if any(word in lower_question for word in ['how many', 'how much', 'price', 'cost']):
                if 'money' in entities:
                    return QAResult(
                        answer=entities['money'][0],
                        confidence=0.95,
                        start=context.find(entities['money'][0]),
                        end=context.find(entities['money'][0]) + len(entities['money'][0]),
                        context=context,
                        processing_time=(datetime.now() - start_time).total_seconds()
                    )
                if 'number' in entities:
                    return QAResult(
                        answer=entities['number'][0],
                        confidence=0.90,
                        start=context.find(entities['number'][0]),
                        end=context.find(entities['number'][0]) + len(entities['number'][0]),
                        context=context,
                        processing_time=(datetime.now() - start_time).total_seconds()
                    )
            
            # Get most relevant context segment
            relevant_context = self._get_best_segment(question, context)
            
            # Use QA pipeline
            result = self.qa_pipeline(
                question=question,
                context=relevant_context,
                max_answer_len=100,
                handle_impossible_answer=True,
                top_k=3
            )
            
            # Validate answer quality
            if not result['answer'].strip() or result['score'] < 0.2:
                return QAResult(
                    answer="I could not find a reliable answer to this question in the given context.",
                    confidence=0.0,
                    start=0,
                    end=0,
                    context=relevant_context,
                    processing_time=(datetime.now() - start_time).total_seconds()
                )
            
            return QAResult(
                answer=result['answer'].strip(),
                confidence=round(float(result['score']) * 100, 2),
                start=result['start'],
                end=result['end'],
                context=relevant_context,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return QAResult(
                answer=f"An error occurred while processing the question: {str(e)}",
                confidence=0.0,
                start=0,
                end=0,
                context=context,
                processing_time=(datetime.now() - start_time).total_seconds()
            )

def main():
    """Main function to handle command line usage"""
    if len(sys.argv) != 3:
        print(json.dumps({
            "error": "Please provide both question and context file path"
        }))
        sys.exit(1)
        
    question = sys.argv[1]
    context_path = sys.argv[2]
    
    try:
        # Read context file
        with open(context_path, 'r', encoding='utf-8') as f:
            context = f.read()
            
        if not context.strip():
            raise ValueError("Empty context provided")
            
        # Initialize QA system
        qa = AdvancedQA()
        
        # Get answer
        result = qa.answer_question(question, context)
        
        # Convert to dict for JSON serialization
        output = {
            "answer": result.answer,
            "confidence": result.confidence,
            "start": result.start,
            "end": result.end,
            "processing_time": result.processing_time
        }
        
        print(json.dumps(output))
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print(json.dumps({
            "error": str(e)
        }))
        sys.exit(1)

if __name__ == "__main__":
    main()