# qa.py
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import sys
import json
import re
from typing import Dict, List, Tuple

class ImprovedQASystem:
    def __init__(self, model_name: str = "bert-large-uncased-whole-word-masking-finetuned-squad"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.qa_pipeline = pipeline(
                "question-answering",
                model=model_name,
                device=self.device
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(json.dumps({"error": f"Failed to initialize QA system: {str(e)}"}))
            sys.exit(1)

    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text

    def create_chunks(self, text: str, max_length: int = 512) -> List[str]:
        """Split text into overlapping chunks."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            tokens = self.tokenizer.tokenize(sentence)
            if current_length + len(tokens) > max_length:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    # Keep last sentence for overlap
                    current_chunk = [current_chunk[-1]]
                    current_length = len(self.tokenizer.tokenize(current_chunk[0]))
            current_chunk.append(sentence)
            current_length += len(tokens)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def process_question(self, question: str) -> str:
        """Optimize question format."""
        question = question.strip()
        question_lower = question.lower()

        # Handle common question patterns
        if any(word in question_lower for word in ['subject', 'topic', 'about']):
            return "What is the main topic being discussed in this text?"
        if "when" in question_lower and "time" in question_lower:
            return "At what specific time or date " + question.rstrip('?') + "?"
        if "where" in question_lower and "location" in question_lower:
            return "At what location " + question.rstrip('?') + "?"

        # Ensure question ends with question mark
        if not question.endswith('?'):
            question += '?'

        return question

    def get_best_answer(self, answers: List[Dict]) -> Dict:
        """Select the best answer from multiple candidates."""
        if not answers:
            return {
                "answer": "Unable to find an answer",
                "confidence": 0,
                "start": 0,
                "end": 0
            }

        # Filter out low confidence answers
        valid_answers = [a for a in answers if a['score'] > 0.2]
        if not valid_answers:
            return {
                "answer": "No confident answer found. Try rephrasing the question.",
                "confidence": 0,
                "start": 0,
                "end": 0
            }

        # Return the highest confidence answer
        best_answer = max(valid_answers, key=lambda x: x['score'])
        return {
            "answer": best_answer['answer'].strip(),
            "confidence": round(best_answer['score'] * 100, 2),
            "start": best_answer['start'],
            "end": best_answer['end']
        }

    def answer_question(self, question: str, context: str) -> Dict:
        """Process question and get best answer from context."""
        try:
            # Preprocess inputs
            question = self.process_question(question)
            context = self.preprocess_text(context)

            # Split into chunks if text is long
            if len(self.tokenizer.tokenize(context)) > 512:
                chunks = self.create_chunks(context)
                all_answers = []
                
                for chunk in chunks:
                    try:
                        result = self.qa_pipeline(
                            question=question,
                            context=chunk,
                            max_answer_len=100,
                            handle_impossible_answer=True
                        )
                        if isinstance(result, list):
                            all_answers.extend(result)
                        else:
                            all_answers.append(result)
                    except Exception as e:
                        print(f"Warning: Error processing chunk: {str(e)}", file=sys.stderr)
                        continue
                
                return self.get_best_answer(all_answers)
            else:
                # Process short text directly
                result = self.qa_pipeline(
                    question=question,
                    context=context,
                    max_answer_len=100,
                    handle_impossible_answer=True
                )
                return self.get_best_answer([result] if not isinstance(result, list) else result)

        except Exception as e:
            print(json.dumps({"error": f"Failed to process question: {str(e)}"}))
            sys.exit(1)

def main():
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Please provide both question and context file path"}))
        sys.exit(1)

    question = sys.argv[1]
    context_path = sys.argv[2]

    try:
        with open(context_path, 'r', encoding='utf-8') as f:
            context = f.read()

        if not context.strip():
            raise ValueError("Context file is empty")

        qa_system = ImprovedQASystem()
        result = qa_system.answer_question(question, context)
        print(json.dumps(result))

    except FileNotFoundError:
        print(json.dumps({"error": "Context file not found"}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": f"Failed to process question: {str(e)}"}))
        sys.exit(1)

if __name__ == "__main__":
    main()