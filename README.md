🧠 Finance Analyst — Domain-Specific Qwen2.5 LoRA + RAG System

A domain-specialized Finance Q&A Agent that combines LoRA fine-tuned Qwen2.5-3B with RAG (Retrieval-Augmented Generation) to deliver context-grounded, concise, and reliable financial answers.
It can classify finance-related questions, retrieve real-time financial data from curated sources, and generate expert-level summaries.

🚀 Features

✅ Fine-tuned Qwen2.5-3B on financial datasets (Dolly + FIQA + synthetic finance QA) using LoRA
✅ 4-bit quantized inference for efficient deployment
✅ Integrated Retriever + Reranker (Chroma + Cross-Encoder)
✅ Contextual multi-query expansion for better document recall
✅ Dynamic finance question classifier
✅ End-to-end RAG pipeline with clean post-processing
✅ Interactive CLI: “Finance Analyst ready. Type ‘exit’ to quit”

🏗️ Architecture Overview
                ┌─────────────────────────────┐
                │  User Question (Finance?)   │
                └──────────────┬──────────────┘
                               │
                    Yes        ▼
                   ┌────────────────────┐
                   │ Multi-Query Expander│
                   └────────────┬────────┘
                                ▼
                 ┌────────────────────────────┐
                 │ Web + Stock Retriever (RAG)│
                 └────────────┬───────────────┘
                              ▼
                   ┌────────────────────────┐
                   │ Fine-Tuned Qwen2.5 LLM │
                   └────────────┬───────────┘
                                ▼
                         Final Answer

🧩 Components
1. Fine-Tuning (LoRA)

Model: Qwen2.5-3B
Quantization: 4-bit (bnb_4bit_compute_dtype=float16)
Adapters: LoRA with r=16, alpha=32, target_modules=['q_proj','v_proj']

trainer.train()
results = trainer.evaluate()
print("Perplexity:", math.exp(results["eval_loss"]))


Training converged smoothly — see the training_loss.png graph for stable loss reduction.

2. Data Preparation
Fine-tuning Dataset:

🧾 Dolly 15k — Instruction-based dialogue

💰 FIQA — Financial sentiment + question-answering

🧠 Synthetic Finance QA — Generated with GPT-4 style templates

RAG Corpus:

📊 CSV-based stock data (Apple, Tesla, etc.)

🌐 Curated finance URLs:

Investopedia (ETF, compound interest, inflation, credit score)

Federal Reserve policy pages

Kiplinger (investing mistakes)

Tax awareness and Fed meeting summaries

3. Retrieval System (RAG)

Embeddings: all-MiniLM-L6-v2

Vectorstore: Chroma (persistent)

Reranker: cross-encoder/ms-marco-TinyBERT-L-2-v2

Multi-query expansion for semantic recall improvement

Chunking: 1000 chars per chunk, 100 overlap

4. Finance Classification

Before retrieval, every question is validated:

You are a financial domain classifier.
Classify the following question as 'finance' or 'not'.


Non-finance queries are filtered automatically:

“I can only answer finance-related questions.”

5. Post-Processing

Outputs are trimmed to 3 concise sentences and cleaned from verbose model artifacts.

💻 Example Run
Finance Analyst ready. Type 'exit' to quit

Enter your question: What could be the market impact if President Trump fires Fed Chair Jerome Powell?

=== Final Answer ===
The market impact of firing Fed Chair Jerome Powell could be significant, potentially wiping out $1.5 trillion from the stock market...

Enter your question: Who is Cristiano Ronaldo?
I can only answer finance related questions.

Enter your question: What are the stock prices of apple?

=== Final Answer ===
Apple's stock price is $231.83 USD. It has a P/E ratio of 30.3 and a market capitalization of $3010.0 billion USD.

⚙️ Run Locally
1. Clone & Setup
git clone https://github.com/yourname/finance-analyst.git
cd finance-analyst
pip install -r requirements.txt

2. Fine-tune Model
python src/train/finetune_lora.py --config configs/finetune.yaml

3. Run RAG Assistant
python src/rag/main.py --config configs/rag.yaml

📊 Evaluation

Perplexity: 2.95

📉 Training Behavior

The training and validation loss curves indicate rapid convergence within the first few hundred steps, followed by a stable plateau:

Initial loss: ~11.3 → Final loss: ~1.08

Validation closely tracks training loss throughout

No overfitting or instability observed

This behavior confirms that the model successfully adapted to the financial instruction domain while maintaining strong generalization.

<div align="center"> <img src="assets/training_loss.png" alt="Training and Validation Loss" width="500"> </div>



Loss curve shows strong convergence with smooth validation behavior.

🔮 Future Improvements

🧩 Fine-tuning Enhancements

Use larger financial corpora (EDGAR filings, SEC 10-K, financial news)

Apply LoRA rank search or QLoRA for memory-efficient multi-domain tuning

Introduce instruction-following refinement with finance experts’ Q&A datasets

🌐 RAG & Knowledge Expansion

Add more trusted finance domains: Bloomberg, Yahoo Finance, IMF, World Bank, CNBC, etc.

Integrate live data APIs for real-time market updates

Implement hybrid retrieval (vector + keyword BM25)

🧠 Pipeline Intelligence

Add reasoning feedback loops (self-reflection on factual consistency)

Memory caching of prior user sessions

Evaluate on benchmark datasets (e.g., FinancialQA, BankExamQA)

🏁 Summary

This project demonstrates a domain-specialized financial analyst assistant that blends:

Efficient parameter-efficient fine-tuning (PEFT)

Intelligent retrieval and reranking

Accurate financial domain filtering

It’s an end-to-end example of how to combine Qwen2.5 LoRA + RAG + LangChain for enterprise-grade domain QA.