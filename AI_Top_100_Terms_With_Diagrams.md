# Top 100 AI Terms with Meanings and Diagram Explanations

This guide lists 100 essential AI terms. Each term includes:
- **Meaning**: What it is.
- **Diagram explanation**: A quick visual flow (ASCII) to understand how it works in practice.

---

## 1) Artificial Intelligence (AI)
**Meaning:** The broad field of building systems that perform tasks requiring human-like intelligence (reasoning, perception, decision-making).

**Diagram explanation:**
```text
Data + Rules/Models -> AI System -> Prediction/Decision/Action
```

## 2) Machine Learning (ML)
**Meaning:** A subset of AI where systems learn patterns from data instead of being explicitly programmed.

**Diagram explanation:**
```text
Training Data -> Learning Algorithm -> Trained Model -> New Input -> Output
```

## 3) Deep Learning (DL)
**Meaning:** A subset of ML using multi-layer neural networks to learn complex patterns.

**Diagram explanation:**
```text
Input -> Hidden Layer 1 -> Hidden Layer 2 -> ... -> Output
```

## 4) Neural Network
**Meaning:** A model made of connected nodes (neurons) that transforms inputs to outputs via weighted connections.

**Diagram explanation:**
```text
x1,x2,x3 -> [Neuron Layer] -> [Neuron Layer] -> y
```

## 5) Large Language Model (LLM)
**Meaning:** A deep learning model trained on huge text corpora to understand and generate language.

**Diagram explanation:**
```text
Prompt -> Tokenization -> LLM -> Next-token probabilities -> Response
```

## 6) Transformer
**Meaning:** Neural architecture using attention to process entire sequences efficiently; backbone of modern LLMs.

**Diagram explanation:**
```text
Tokens -> Embeddings -> Self-Attention + Feed-Forward blocks -> Output tokens
```

## 7) Token
**Meaning:** The unit of text the model processes (word piece, character chunk, or symbol).

**Diagram explanation:**
```text
"unbelievable" -> ["un", "believ", "able"]
```

## 8) Tokenization
**Meaning:** The process of converting raw text into tokens.

**Diagram explanation:**
```text
Raw Text -> Tokenizer -> Token IDs -> Model
```

## 9) Embedding
**Meaning:** A dense numeric vector representing semantic meaning of text/image/audio.

**Diagram explanation:**
```text
"cat" -> [0.12, -0.44, ...]  (vector in semantic space)
```

## 10) Vector Database
**Meaning:** A database optimized to store/search embeddings via similarity.

**Diagram explanation:**
```text
Docs -> Embeddings -> Vector DB
Query -> Embedding -> Similarity Search -> Top matches
```

## 11) Similarity Search
**Meaning:** Retrieving nearest vectors to a query vector (semantic matching).

**Diagram explanation:**
```text
Query Vector -> kNN/ANN index -> Closest Vectors -> Retrieved items
```

## 12) Retrieval-Augmented Generation (RAG)
**Meaning:** Combining retrieval from external knowledge with LLM generation.

**Diagram explanation:**
```text
Question -> Retriever -> Relevant docs -> LLM + docs -> Grounded answer
```

## 13) Prompt
**Meaning:** The instruction/context provided to an AI model.

**Diagram explanation:**
```text
System Prompt + User Prompt + Context -> Model -> Response
```

## 14) Prompt Engineering
**Meaning:** Crafting prompts for better reliability, structure, and quality.

**Diagram explanation:**
```text
Goal -> Prompt design -> Test outputs -> Refine prompt -> Better results
```

## 15) Zero-shot Learning
**Meaning:** Model performs a task without seeing examples for that exact task.

**Diagram explanation:**
```text
Instruction only -> Model prior knowledge -> Output
```

## 16) One-shot Learning
**Meaning:** Model gets one example to infer task format.

**Diagram explanation:**
```text
1 example + instruction -> Model -> Output in same pattern
```

## 17) Few-shot Learning
**Meaning:** Model gets a small set of examples in prompt before solving.

**Diagram explanation:**
```text
Few examples -> Pattern induction -> New input -> Output
```

## 18) Chain-of-Thought (CoT)
**Meaning:** A reasoning style where intermediate steps are generated to solve complex tasks.

**Diagram explanation:**
```text
Problem -> Intermediate reasoning steps -> Final answer
```

## 19) ReAct
**Meaning:** Prompting style where model alternates between reasoning and acting (e.g., tool calls).

**Diagram explanation:**
```text
Think -> Act(tool) -> Observe -> Think -> Final answer
```

## 20) Agent
**Meaning:** An AI system that can plan, use tools, and take iterative actions toward goals.

**Diagram explanation:**
```text
Goal -> Plan -> Tool use -> Feedback -> Revised plan -> Completion
```

## 21) Tool Calling / Function Calling
**Meaning:** Letting model invoke external functions/APIs in structured format.

**Diagram explanation:**
```text
User request -> LLM selects function + args -> Tool result -> LLM final response
```

## 22) Context Window
**Meaning:** Maximum tokens a model can consider at once (prompt + output constraints).

**Diagram explanation:**
```text
[Past messages + docs + instructions] <= Max context tokens
```

## 23) Hallucination
**Meaning:** Confidently generated but incorrect or unsupported model output.

**Diagram explanation:**
```text
Prompt lacking facts -> Model guesses -> Plausible but false answer
```

## 24) Grounding
**Meaning:** Constraining outputs to trusted evidence (docs/databases/tools).

**Diagram explanation:**
```text
Question + evidence -> Model -> Evidence-linked answer
```

## 25) Fine-tuning
**Meaning:** Additional training of a base model on domain/task-specific data.

**Diagram explanation:**
```text
Base model + labeled domain data -> fine-tuning -> specialized model
```

## 26) Instruction Tuning
**Meaning:** Fine-tuning model on instruction-response pairs to improve followability.

**Diagram explanation:**
```text
("Do X" -> "Expected answer") pairs -> tuned model
```

## 27) RLHF (Reinforcement Learning from Human Feedback)
**Meaning:** Aligning model behavior using human preference data and reward optimization.

**Diagram explanation:**
```text
Model outputs -> Human rankings -> Reward model -> Policy optimization
```

## 28) Supervised Learning
**Meaning:** Learning from labeled input-output pairs.

**Diagram explanation:**
```text
(x, y) labeled data -> Model training -> Predict y for new x
```

## 29) Unsupervised Learning
**Meaning:** Learning structure from unlabeled data (clustering, compression, etc.).

**Diagram explanation:**
```text
Unlabeled data -> Pattern discovery -> groups/representations
```

## 30) Self-Supervised Learning
**Meaning:** Learning from automatically created labels from the data itself.

**Diagram explanation:**
```text
Raw data -> proxy task generation -> representation learning
```

## 31) Reinforcement Learning (RL)
**Meaning:** Agent learns by taking actions and receiving rewards in an environment.

**Diagram explanation:**
```text
State -> Action -> Reward + Next state -> Policy update
```

## 32) Policy
**Meaning:** Strategy mapping states/observations to actions.

**Diagram explanation:**
```text
Observation -> Policy π(a|s) -> Action
```

## 33) Reward Function
**Meaning:** Signal defining success in RL.

**Diagram explanation:**
```text
Action outcome -> Reward score -> Learning direction
```

## 34) Q-Learning
**Meaning:** RL method learning expected value of action in a state (Q-value).

**Diagram explanation:**
```text
Q(s,a) table/model <- reward + max future Q
```

## 35) Exploration vs Exploitation
**Meaning:** RL tradeoff between trying new actions and using known best actions.

**Diagram explanation:**
```text
Explore (new) <-> Exploit (best known)
```

## 36) Loss Function
**Meaning:** Numeric measure of model error to minimize during training.

**Diagram explanation:**
```text
Prediction vs truth -> Loss -> Backprop updates weights
```

## 37) Gradient Descent
**Meaning:** Optimization method that updates parameters in direction reducing loss.

**Diagram explanation:**
```text
Current weights -> gradient -> small step downhill -> lower loss
```

## 38) Backpropagation
**Meaning:** Algorithm to propagate output error backward through network layers.

**Diagram explanation:**
```text
Forward pass -> Loss -> Backward gradients -> Weight updates
```

## 39) Learning Rate
**Meaning:** Step size used during parameter updates.

**Diagram explanation:**
```text
Large LR: fast/unstable | Small LR: slow/stable
```

## 40) Epoch
**Meaning:** One full pass through the entire training dataset.

**Diagram explanation:**
```text
Dataset scanned once = 1 epoch
```

## 41) Batch Size
**Meaning:** Number of samples processed before one optimization update.

**Diagram explanation:**
```text
Dataset -> mini-batches -> update per batch
```

## 42) Overfitting
**Meaning:** Model memorizes training data; poor generalization to unseen data.

**Diagram explanation:**
```text
Train accuracy high; validation accuracy drops
```

## 43) Underfitting
**Meaning:** Model too simple to learn patterns even in training data.

**Diagram explanation:**
```text
Train and validation accuracy both low
```

## 44) Regularization
**Meaning:** Techniques to reduce overfitting (L1/L2, dropout, early stopping).

**Diagram explanation:**
```text
Base training + constraints/noise -> better generalization
```

## 45) Dropout
**Meaning:** Randomly disabling neurons during training to improve robustness.

**Diagram explanation:**
```text
Layer neurons -> randomly dropped -> reduced co-dependency
```

## 46) Cross-Validation
**Meaning:** Repeated train/validation splits to estimate model reliability.

**Diagram explanation:**
```text
Data -> K folds -> rotate validation fold -> average performance
```

## 47) Train/Validation/Test Split
**Meaning:** Data partition for training, tuning, and final unbiased evaluation.

**Diagram explanation:**
```text
Data -> Train | Validation | Test
```

## 48) Hyperparameter
**Meaning:** Configuration chosen before training (LR, batch size, layers).

**Diagram explanation:**
```text
Choose hyperparameters -> Train -> Evaluate -> Tune
```

## 49) Hyperparameter Tuning
**Meaning:** Searching best hyperparameter combinations for model performance.

**Diagram explanation:**
```text
Grid/Random/Bayesian search -> best validation score
```

## 50) Inference
**Meaning:** Using a trained model to make predictions on new input.

**Diagram explanation:**
```text
New input -> trained model -> prediction/response
```

## 51) Latency
**Meaning:** Time taken by model/system to produce output.

**Diagram explanation:**
```text
Request time ----> Response time = latency
```

## 52) Throughput
**Meaning:** Number of requests/tokens processed per unit time.

**Diagram explanation:**
```text
Requests per second (RPS) / tokens per second (TPS)
```

## 53) Quantization
**Meaning:** Reducing numerical precision (e.g., FP16/INT8/INT4) for faster/cheaper inference.

**Diagram explanation:**
```text
High-precision weights -> lower-bit weights -> smaller/faster model
```

## 54) Distillation
**Meaning:** Training a smaller student model to mimic a larger teacher model.

**Diagram explanation:**
```text
Teacher outputs -> Student training -> compact model
```

## 55) Pruning
**Meaning:** Removing low-importance weights/neurons to shrink model.

**Diagram explanation:**
```text
Trained model -> remove weak connections -> sparse model
```

## 56) Checkpoint
**Meaning:** Saved model state during or after training.

**Diagram explanation:**
```text
Training progress -> periodic save -> resume/evaluate/deploy
```

## 57) Foundation Model
**Meaning:** Large pretrained model adaptable to many tasks.

**Diagram explanation:**
```text
Massive pretraining -> broad capability -> downstream adaptation
```

## 58) Multimodal Model
**Meaning:** Model that handles multiple data types (text, image, audio, video).

**Diagram explanation:**
```text
Text/Image/Audio -> shared model -> unified output
```

## 59) Computer Vision
**Meaning:** AI field focused on understanding images/video.

**Diagram explanation:**
```text
Image -> CV model -> class/box/mask/description
```

## 60) Natural Language Processing (NLP)
**Meaning:** AI field focused on understanding and generating human language.

**Diagram explanation:**
```text
Text -> NLP model -> sentiment/summary/translation/answer
```

## 61) Speech Recognition (ASR)
**Meaning:** Converting speech audio into text.

**Diagram explanation:**
```text
Audio waveform -> ASR model -> transcript
```

## 62) Text-to-Speech (TTS)
**Meaning:** Generating natural speech audio from text.

**Diagram explanation:**
```text
Text -> TTS model -> speech waveform
```

## 63) OCR (Optical Character Recognition)
**Meaning:** Extracting text from images/scanned documents.

**Diagram explanation:**
```text
Document image -> OCR -> editable/searchable text
```

## 64) Named Entity Recognition (NER)
**Meaning:** Detecting entities like person, company, date, location in text.

**Diagram explanation:**
```text
Sentence -> token tagging -> [PERSON], [ORG], [DATE]
```

## 65) Sentiment Analysis
**Meaning:** Predicting opinion polarity (positive/negative/neutral).

**Diagram explanation:**
```text
Review text -> classifier -> sentiment label + score
```

## 66) Topic Modeling
**Meaning:** Discovering hidden thematic structure in text collections.

**Diagram explanation:**
```text
Corpus -> model -> Topic A, Topic B, Topic C ...
```

## 67) Clustering
**Meaning:** Grouping similar data points without labels.

**Diagram explanation:**
```text
Data points -> similarity grouping -> clusters
```

## 68) Classification
**Meaning:** Predicting discrete labels (spam/not spam, fraud/not fraud).

**Diagram explanation:**
```text
Input features -> classifier -> class label
```

## 69) Regression
**Meaning:** Predicting continuous numeric values (price, demand, risk score).

**Diagram explanation:**
```text
Input features -> regressor -> numeric value
```

## 70) Anomaly Detection
**Meaning:** Finding unusual patterns/events differing from normal behavior.

**Diagram explanation:**
```text
Normal baseline -> incoming event -> anomaly score
```

## 71) Recommender System
**Meaning:** Suggesting items based on user behavior/content similarity.

**Diagram explanation:**
```text
User history + item features -> ranking model -> recommendations
```

## 72) Collaborative Filtering
**Meaning:** Recommending using patterns of similar users/items interactions.

**Diagram explanation:**
```text
User-item matrix -> latent factors -> predicted preferences
```

## 73) Attention Mechanism
**Meaning:** Letting model focus on most relevant parts of input.

**Diagram explanation:**
```text
Query-Key matching -> attention weights -> weighted value sum
```

## 74) Self-Attention
**Meaning:** Attention where tokens attend to other tokens in same sequence.

**Diagram explanation:**
```text
Token i -> compares with all tokens -> context-aware representation
```

## 75) Positional Encoding
**Meaning:** Injecting token order information into transformer inputs.

**Diagram explanation:**
```text
Token embedding + position vector -> ordered representation
```

## 76) Decoder-only Model
**Meaning:** Transformer architecture that predicts next token autoregressively.

**Diagram explanation:**
```text
Prompt tokens -> masked self-attention -> next token -> repeat
```

## 77) Encoder-Decoder Model
**Meaning:** Architecture with encoder for input understanding and decoder for generation.

**Diagram explanation:**
```text
Input -> Encoder states -> Decoder -> output sequence
```

## 78) Beam Search
**Meaning:** Decoding strategy keeping top-k candidate sequences each step.

**Diagram explanation:**
```text
Step t: keep best k paths -> expand -> keep best k again
```

## 79) Temperature (Sampling)
**Meaning:** Controls randomness in text generation.

**Diagram explanation:**
```text
Low T -> deterministic; High T -> creative/diverse
```

## 80) Top-k Sampling
**Meaning:** Sample next token only from k highest-probability candidates.

**Diagram explanation:**
```text
Probabilities -> keep top k -> sample one
```

## 81) Top-p (Nucleus) Sampling
**Meaning:** Sample from smallest token set whose cumulative probability >= p.

**Diagram explanation:**
```text
Sort probs -> accumulate until p -> sample in that subset
```

## 82) Logits
**Meaning:** Raw model output scores before softmax probability conversion.

**Diagram explanation:**
```text
Model head -> logits -> softmax -> probabilities
```

## 83) Softmax
**Meaning:** Function converting scores into a probability distribution.

**Diagram explanation:**
```text
[2.1, 0.3, -1.2] -> [0.79, 0.16, 0.05]
```

## 84) Perplexity
**Meaning:** Language-model metric of uncertainty; lower is generally better.

**Diagram explanation:**
```text
Better next-token confidence -> lower perplexity
```

## 85) BLEU Score
**Meaning:** Metric for text generation quality based on n-gram overlap (often translation).

**Diagram explanation:**
```text
Generated text vs reference text -> overlap score
```

## 86) ROUGE
**Meaning:** Overlap-based metric often used for summarization quality.

**Diagram explanation:**
```text
Summary candidate vs reference summary -> recall-focused overlap
```

## 87) F1 Score
**Meaning:** Harmonic mean of precision and recall in classification.

**Diagram explanation:**
```text
Precision + Recall -> F1 balances both
```

## 88) Precision
**Meaning:** Of predicted positives, how many are truly positive.

**Diagram explanation:**
```text
Precision = TP / (TP + FP)
```

## 89) Recall
**Meaning:** Of actual positives, how many were correctly found.

**Diagram explanation:**
```text
Recall = TP / (TP + FN)
```

## 90) Confusion Matrix
**Meaning:** Table showing TP, FP, FN, TN for classifier outcomes.

**Diagram explanation:**
```text
            Pred +   Pred -
Actual +      TP       FN
Actual -      FP       TN
```

## 91) Model Drift
**Meaning:** Performance degradation over time due to changing real-world data patterns.

**Diagram explanation:**
```text
Train distribution != current production distribution
```

## 92) Data Drift
**Meaning:** Input data distribution changes from training baseline.

**Diagram explanation:**
```text
Old feature histogram -> New shifted histogram
```

## 93) Concept Drift
**Meaning:** Relationship between input and target changes over time.

**Diagram explanation:**
```text
Same x, but true y mapping evolves
```

## 94) MLOps
**Meaning:** Practices/tools for deploying, monitoring, and maintaining ML systems.

**Diagram explanation:**
```text
Data -> Train -> Validate -> Deploy -> Monitor -> Retrain loop
```

## 95) Feature Engineering
**Meaning:** Creating/improving input variables to help learning algorithms.

**Diagram explanation:**
```text
Raw data -> cleaned/derived features -> better model learning
```

## 96) Feature Store
**Meaning:** Centralized system to manage reusable, consistent ML features.

**Diagram explanation:**
```text
Feature pipelines -> feature store -> training + serving reuse
```

## 97) Explainable AI (XAI)
**Meaning:** Methods to make model decisions interpretable to humans.

**Diagram explanation:**
```text
Prediction -> explanation method -> important features/reasons
```

## 98) SHAP
**Meaning:** Explanation technique assigning contribution values to features for predictions.

**Diagram explanation:**
```text
Prediction = baseline + Σ(feature contributions)
```

## 99) LIME
**Meaning:** Local explanation method fitting simple interpretable model around one prediction.

**Diagram explanation:**
```text
Perturb neighborhood -> local surrogate model -> explanation
```

## 100) AI Alignment
**Meaning:** Ensuring AI behavior matches human values, intent, and safety requirements.

**Diagram explanation:**
```text
Human goals -> constraints + feedback + evaluation -> aligned behavior
```

---

## Quick Visual Map (How many terms connect)
```text
Data -> Preprocessing -> Features/Embeddings -> Model Training
     -> Evaluation -> Deployment (Inference) -> Monitoring (Drift)
     -> Retrieval/Tools/Agents -> Safer, Grounded AI Outputs
```

