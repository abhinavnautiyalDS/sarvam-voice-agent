# AURA AI Tutor — Machine Learning & AI Learning Handbook

**Audience:** Beginner to intermediate learners  
**Purpose:** Reference curriculum for teaching Python, data science, machine learning, deep learning, NLP and generative AI.

## 1. Teaching approach
Teach in small connected steps:
1. Identify the learner's current level and goal.
2. Explain one concept in plain language before formal terminology.
3. Give a concrete example or everyday analogy.
4. Explain intuition, then mathematical or implementation detail.
5. Ask a short check-for-understanding question.
6. Offer a small exercise or coding task.
7. Review the learner's attempt and explain mistakes constructively.
8. Summarise the key idea and suggest the next prerequisite topic.

Do not assume advanced mathematics or programming knowledge. Ask before jumping to advanced detail. If the learner asks for only a short answer, respect that request.

## 2. Learning roadmap
### Stage A — Python foundations
Variables, data types, operators, conditions, loops, functions, scope, exceptions, lists, tuples, dictionaries, sets, comprehensions, modules, packages, virtual environments, dependency management, CSV/JSON/text files, debugging, logging and object-oriented programming.

Practice: build a CSV summary tool, a simple expense tracker and a command-line quiz.

### Stage B — Data analysis
NumPy arrays, dimensions, indexing, broadcasting and vectorisation; Pandas Series/DataFrames, filtering, grouping, joins and missing values; data cleaning, duplicates, inconsistent types and outlier inspection; exploratory data analysis; Matplotlib; mean, median, mode, variance, standard deviation, quantiles, sampling, distributions and correlation versus causation.

Practice: explore a retail sales dataset, document data-quality issues and communicate three evidence-based findings.

### Stage C — Mathematics for ML
Algebra, functions, logarithms and exponentials; vectors, matrices, dot products, matrix multiplication and norms; derivatives, partial derivatives, gradients and the chain rule; probability, conditional probability, Bayes' theorem and common distributions; expectation, variance and covariance; hypothesis testing, confidence intervals and statistical significance.

Teach intuition first. Define every symbol when using equations.

## 3. Machine Learning foundations
Machine learning uses data to learn patterns that support predictions or decisions. A typical supervised-learning workflow:
1. Define the target and problem type.
2. Collect and inspect representative data.
3. Split data into training, validation and test sets before fitting transformations.
4. Build a simple baseline.
5. Preprocess features using training data only.
6. Train candidate models.
7. Tune hyperparameters using validation or cross-validation.
8. Evaluate once on the held-out test set.
9. Inspect errors, limitations and potential data leakage.
10. Package, monitor and maintain the model.

### Main learning paradigms
- Supervised learning: examples have target labels; tasks include regression and classification.
- Unsupervised learning: no target labels are supplied; tasks include clustering and dimensionality reduction.
- Semi-supervised learning: a small labelled set is combined with a larger unlabelled set.
- Self-supervised learning: training targets are derived from the data itself.
- Reinforcement learning: an agent learns actions from rewards and interaction with an environment.

## 4. Regression
Regression predicts a numeric quantity.

### Linear regression
A linear model predicts a target as an intercept plus a weighted sum of input features. Ordinary least squares chooses coefficients to minimise the sum of squared residuals.

Explain assumptions and limitations: linearity, influential outliers, correlated predictors, residual behaviour and extrapolation risk.

### Other regression models
- Polynomial regression adds transformed features to represent curved relationships.
- Ridge regression uses an L2 penalty to shrink coefficients.
- Lasso regression uses an L1 penalty and can set some coefficients to zero.
- Decision-tree regression learns piecewise rules.
- Random-forest regression averages predictions from many trees.
- Gradient boosting builds an additive sequence of models to reduce errors.

### Regression metrics
- MAE: average absolute prediction error.
- MSE: average squared error; penalises large errors more strongly.
- RMSE: square root of MSE, in the target's units.
- R-squared: proportion of target variance explained relative to a mean baseline; it can be negative on test data.

## 5. Classification
Classification predicts a category.

### Logistic regression
For binary classification, logistic regression maps a linear score through the sigmoid function to estimate a probability. The predicted probability is compared with a decision threshold. A threshold of 0.5 is common but not universally optimal.

### Common algorithms
- K-nearest neighbours: predicts from nearby training examples; feature scaling is important.
- Decision tree: learns feature-based splits; can overfit if unconstrained.
- Random forest: combines many decorrelated trees to improve robustness.
- Support vector machine: finds a separating boundary with a large margin; kernels can model non-linear boundaries.
- Naive Bayes: applies Bayes' rule with a conditional-independence assumption.
- Gradient boosting: combines weak learners sequentially.

### Classification metrics
- Confusion matrix: true positives, true negatives, false positives and false negatives.
- Accuracy: fraction of all predictions that are correct; can mislead on imbalanced data.
- Precision: among predicted positives, the fraction that are truly positive.
- Recall: among actual positives, the fraction detected.
- F1: harmonic mean of precision and recall.
- ROC-AUC: ranking quality across thresholds.
- PR-AUC: often informative when the positive class is rare.

Choose metrics according to error costs and class balance, not because one metric is universally best.

## 6. Unsupervised learning
### Clustering
- K-Means assigns points to k centroids and iteratively updates assignments and centroids. Scale numeric features and choose k thoughtfully.
- Hierarchical clustering builds nested clusters and can be visualised with a dendrogram.
- DBSCAN groups dense regions and can identify noise; it is sensitive to distance scale and parameters.

### Dimensionality reduction
- PCA projects data onto orthogonal directions that capture decreasing amounts of variance.
- t-SNE and UMAP can help visualise local structure, but visual distances and cluster shapes are not proof of true groups.

## 7. Data preparation and feature engineering
Separate target, features and identifiers. Prevent target leakage and time leakage. Impute missing values using statistics learned from training data. Encode categorical variables with one-hot, ordinal or suitable alternatives. Scale features when the algorithm is distance- or gradient-sensitive. Create domain-informed features while preserving reproducibility. Use pipelines so preprocessing is fitted only on training folds. Handle imbalance with suitable metrics, class weights or carefully applied resampling. Document data provenance, assumptions and rejected records.

## 8. Model validation and responsible evaluation
- A training score estimates fit to seen examples; it is not a reliable estimate of generalisation.
- A validation set supports model selection and hyperparameter tuning.
- A test set should remain untouched until final evaluation.
- Use stratified splitting for classification when appropriate.
- Use time-based splitting for forecasting or temporal data.
- Cross-validation estimates variability across splits.
- Check subgroup performance, data quality, calibration and operational constraints.
- Never claim a model is unbiased or production-ready from one metric alone.

## 9. Deep learning foundations
A neural network consists of layers of parameterised transformations and non-linear activations.
- Neurons compute weighted sums plus biases.
- Activation functions include ReLU, sigmoid, tanh and softmax.
- A loss function measures prediction error.
- Backpropagation computes gradients using the chain rule.
- An optimiser such as SGD or Adam updates parameters.
- Epoch: one pass through the training dataset.
- Batch: a subset used for one parameter update.
- Learning rate: controls update size.

Overfitting means training performance is much better than validation performance. Underfitting means a model cannot capture the pattern. Vanishing or exploding gradients occur when gradients become too small or too large. Possible remedies include better data, regularisation, dropout, early stopping, normalisation and architecture changes.

## 10. Computer vision
Image tensors represent height, width and channels. Convolution filters detect local patterns. Pooling/downsampling reduces spatial dimensions. CNNs combine local feature extraction and learned representations. Transfer learning starts from pretrained weights and adapts a model to a target dataset. Common architectures include ResNet, EfficientNet and MobileNet. Evaluate with suitable classification metrics and inspect per-class errors; for detection or segmentation use task-specific metrics such as IoU or mAP.

## 11. Natural language processing
Text normalisation and tokenisation turn raw text into model inputs. Vocabulary and subword tokenisation handle rare or unseen words. Bag-of-words and TF-IDF provide sparse lexical features. Word embeddings represent tokens as dense vectors. Sequence models include RNNs, LSTMs and GRUs. Attention lets a model weight relevant positions in a sequence. Transformers use attention and feed-forward layers to process token representations. Encoder models are commonly used for classification/understanding; decoder models are commonly used for autoregressive generation.

## 12. Generative AI and LLMs
- An LLM predicts tokens based on preceding context and learned parameters.
- A prompt supplies task instructions and context; it does not guarantee factual correctness.
- Temperature influences sampling variability, not factual accuracy.
- Context window limits how much input/output context can be processed.
- Embeddings map text into vectors for semantic comparison.
- Retrieval-augmented generation (RAG) retrieves relevant source passages and provides them to an LLM as context.
- A vector index supports similarity search; metadata filters can isolate collections or tenants.
- Chunk size and overlap affect retrieval granularity.
- RAG quality depends on source quality, chunking, embedding alignment, retrieval, prompt construction and answer evaluation.
- Agents combine model reasoning with tools or actions; tool permissions and outputs must be validated.
- Fine-tuning changes model parameters; it is not the same as adding documents to a RAG knowledge base.

### RAG workflow
1. Prepare trusted source documents.
2. Split documents into meaningful chunks.
3. Generate document embeddings.
4. Store vectors with document and access metadata.
5. Embed the user's question with the matching embedding model.
6. Retrieve top relevant chunks, applying metadata filters.
7. Pass only relevant context to the LLM.
8. Ask the LLM to distinguish supported facts from uncertainty.
9. Evaluate retrieval relevance and answer faithfulness.

## 13. MLOps and deployment
Track datasets, code, configuration and model versions. Use experiment tracking for parameters and metrics. Validate schemas and input distributions. Package inference behind a stable API. Monitor latency, errors, drift and resource use. Keep secrets out of source control. Add tests for preprocessing, inference contracts and failure cases. Plan rollback and model replacement. Treat model outputs as probabilistic and monitor real-world performance.

## 14. Suggested hands-on projects
1. House-price regression with an honest train/validation/test workflow.
2. Customer churn classification with precision/recall trade-offs.
3. Customer segmentation with K-Means and cluster interpretation.
4. Image classifier using transfer learning.
5. Sentiment classifier with TF-IDF and a transformer baseline.
6. Document question-answering assistant using RAG.
7. Small end-to-end ML API with input validation, logging and monitoring.

For every project, require a problem statement, dataset description, baseline, evaluation, error analysis, limitations and reproducible instructions.

## 15. Quick knowledge checks
- Regression predicts a number; classification predicts a category.
- A model should not be evaluated only on its training data.
- Data leakage occurs when information unavailable at prediction time influences training or evaluation.
- Precision measures correctness among predicted positives; recall measures coverage of actual positives.
- K-Means requires a chosen number of clusters and is affected by feature scale.
- Embeddings are vector representations; a vector database does not itself understand or teach a concept.
- RAG retrieves external context; fine-tuning updates model parameters.
- High accuracy can hide poor performance on a minority class.

## 16. Tutor response policy
Do not fabricate benchmark results, library behaviour, citations, code execution or course certificates. When giving code, explain dependencies and expected input/output. If code has not been executed, say so. Prefer small runnable examples and build complexity gradually. Encourage learners to understand and adapt code rather than blindly copy it. For a question outside this handbook, answer from general model knowledge when appropriate, but state uncertainty and do not pretend the handbook contains the answer.