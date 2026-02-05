
# Sentence Autocomplete

This project implements a sentence autocompletion system similar to Gmail Smart Compose.
Given an incomplete sentence, the system generates 3 meaningful autocomplete suggestions.

---

## a) Model Used

- **GPT-Neo (125M parameters)**
  - Decoder-only causal language model
  - Selected based on evaluation using **Perplexity (fluency)** and **Distinct-2 (diversity)**
  - Provides a good balance between fluent and non-repetitive suggestions

- **GPT-2 (125M parameters)**
  - Decoder-only causal language model
  - Selected based on evaluation using **Perplexity (fluency)** and **Distinct-2 (diversity)**
  - Produces highly fluent and conservative autocomplete suggestions with lower perplexity

---

## b) How to Run the Code

### 1. Install dependencies
'''bash
pip install requirements.txt
'''

### 2. Run the python notebook from start to bottom or run the python script 
'''bash
python autocomplete.py
'''


### 3. Provide input (Example input)
'''
Enter a partial sentence: AI can help improve
'''

### Example Output
'''
1. AI can help improve efficiency across business operations.
2. AI can help improve decision-making through data analysis.
3. AI can help improve customer experience using personalization.
'''

## Logs
'''
1. Logs the model name used for generation
2. Records the input prompt provided by the users
3. Stores generation parameters (Top-K, temperature, and maximum tokens)
4. Captures all generated autocomplete outputs
5. Enables traceability, debugging, and reproducibility of results
''


---

## c) Key Challenges, Assumptions, and Improvements

### Challenges
- No single ground-truth completion exists for sentence autocomplete
- Need to balance fluency (low perplexity) with diversity (non-repetitive outputs)
- Preventing over-generation while ensuring sentence completeness

### Assumptions
- CPU-only execution is sufficient for inference
- End-of-sentence token (.) is a reliable stopping criterion, with max_new_tokens set to 40
- Top-K sampling with temperature provides adequate variation

### Improvement Ideas
- Dynamic decoding parameters: Adjust Top-K and temperature based on prompt length or context to balance fluency and diversity more effectively.
- Latency optimization: Measure and optimize inference time, and introduce batching or model quantization for faster responses.
- Evaluation enhancement: Extend evaluation beyond perplexity and diversity by incorporating human feedback or acceptance-rate–based metrics.
- Structured logging: Store logs in JSON format with run IDs and timestamps to enable easier analysis and experiment tracking.
- API deployment: Expose the autocomplete functionality via a REST API (e.g., FastAPI) for easier integration into applications.
- Fallback strategy: Introduce a fallback model or decoding configuration when generation quality degrades or confidence is low.
- Scalability: Add support for batch processing and asynchronous requests to handle higher throughput in production.

  
---

**Final Model Selected:** GPT-Neo  
Chosen for its strong fluency–diversity trade-off.
